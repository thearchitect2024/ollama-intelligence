"""
Optimized embedded GPU LLM engine for Qwen 2.5 7B Instruct (Q4_0).

Features:
- FlashAttention-2 with SDPA fallback
- Micro-batching with length-aware bucketing
- Pinned memory + dual CUDA streams
- torch.compile for graph optimization
- Controlled concurrency with semaphores
- Comprehensive batch-level logging

ARCHITECTURE NOTE:
- app.py sends ALL prompts at once to generate_batch()
- This engine handles ALL micro-batching, bucketing, and concurrency internally
- No iteration batching happens at the app layer
- This allows optimal length-aware bucketing across all prompts
"""
import os
import time
import queue
import logging
import threading
from typing import List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration (from environment or defaults)
# =============================================================================
EMBED_MODEL = os.getenv("EMBED_MODEL", "Qwen/Qwen2.5-7B-Instruct")
EMBED_4BIT = os.getenv("EMBED_4BIT", "1") == "1"
INFER_CONCURRENCY = int(os.getenv("INFER_CONCURRENCY", "3"))
MICRO_BATCH_SIZE = int(os.getenv("MICRO_BATCH_SIZE", "6"))
BATCH_LATENCY_MS = int(os.getenv("BATCH_LATENCY_MS", "100"))
PREFILL_BATCH_TOKENS = int(os.getenv("PREFILL_BATCH_TOKENS", "4096"))
DECODE_CONCURRENCY = int(os.getenv("DECODE_CONCURRENCY", "8"))
USE_FLASH_ATTENTION = os.getenv("USE_FLASH_ATTENTION", "True").lower() == "true"
ENABLE_COMPILE = os.getenv("ENABLE_COMPILE", "True").lower() == "true"
MAX_INPUT_TOK = int(os.getenv("MAX_INPUT_TOK", "4096"))


@dataclass
class BatchMetrics:
    """Metrics for a single batch inference."""
    batch_size: int
    total_input_tokens: int
    avg_input_len: float
    prefill_time_ms: float
    decode_time_ms: float
    tokens_per_sec: float
    vram_current_gb: float
    vram_peak_gb: float


# =============================================================================
# Model Initialization
# =============================================================================
_model = None
_tokenizer = None
_copy_stream = None
_compute_stream = None
_lock = threading.Lock()
_semaphore = threading.BoundedSemaphore(INFER_CONCURRENCY)


def _enable_gpu_optimizations():
    """Enable GPU performance optimizations."""
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        logger.info("✅ GPU optimizations enabled (TF32, cuDNN benchmark)")
    except Exception as e:
        logger.warning(f"Failed to enable GPU optimizations: {e}")


def _check_flash_attention():
    """Check FlashAttention-2 availability."""
    if not USE_FLASH_ATTENTION:
        logger.info("ℹ️  FlashAttention disabled via config")
        return "sdpa"
    
    try:
        import flash_attn
        logger.info("✅ FlashAttention-2 available")
        return "flash_attention_2"
    except ImportError:
        logger.info("ℹ️  FlashAttention-2 not installed, using SDPA")
        return "sdpa"


def _init_model(temperature: float, top_p: float, max_tokens: int):
    """Initialize the model singleton with optimizations."""
    global _model, _tokenizer, _copy_stream, _compute_stream
    
    if _model is not None:
        return
    
    with _lock:
        if _model is not None:
            return
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if device == "cpu":
            logger.warning("⚠️  No CUDA available, running on CPU (slow!)")
        
        # GPU optimizations
        _enable_gpu_optimizations()
        
        # Determine attention implementation
        attn_impl = _check_flash_attention()
        
        # Model loading kwargs
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "use_cache": True,
        }
        
        # Add attention implementation
        try:
            model_kwargs["attn_implementation"] = attn_impl
        except Exception:
            logger.warning("attn_implementation not supported, skipping")
        
        # 4-bit quantization (if enabled)
        if EMBED_4BIT and device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
                logger.info("🧱 Loading with 4-bit quantization")
            except Exception as e:
                logger.warning(f"4-bit quantization failed: {e}, using FP16")
        
        logger.info(f"🚀 Loading {EMBED_MODEL} ({attn_impl})...")
        
        # Load model
        _model = AutoModelForCausalLM.from_pretrained(EMBED_MODEL, **model_kwargs)
        _model.eval()
        
        # Try torch.compile (PyTorch 2.0+)
        if ENABLE_COMPILE and device == "cuda":
            try:
                logger.info("⚙️  Compiling model with torch.compile...")
                _model = torch.compile(
                    _model,
                    mode="reduce-overhead",
                    fullgraph=False  # More robust
                )
                logger.info("✅ Model compiled")
            except Exception as e:
                logger.info(f"ℹ️  torch.compile not available: {e}")
        
        # Load tokenizer
        _tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, trust_remote_code=True)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        
        # Initialize CUDA streams for overlap
        if device == "cuda":
            _copy_stream = torch.cuda.Stream()
            _compute_stream = torch.cuda.Stream()
            logger.info("✅ Dual CUDA streams initialized")
        
        # Warmup
        try:
            with torch.inference_mode():
                dummy = _tokenizer("test", return_tensors="pt")
                if device == "cuda":
                    dummy = {k: v.to(device) for k, v in dummy.items()}
                _ = _model.generate(**dummy, max_new_tokens=1)
            logger.info("✅ Model warmed up")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
        
        logger.info(f"✅ Embedded LLM ready | VRAM: {_get_vram_usage():.1f} GB")


# =============================================================================
# Memory & Performance Utilities
# =============================================================================
def _get_vram_usage() -> float:
    """Get current VRAM usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0


def _get_vram_peak() -> float:
    """Get peak VRAM usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
    return 0.0


def _reset_vram_peak():
    """Reset peak VRAM counter."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# =============================================================================
# Length-Aware Bucketing
# =============================================================================
def _bucket_by_length(prompts: List[str], bucket_size: int = 8) -> List[List[Tuple[int, str]]]:
    """
    Sort prompts by length and group into buckets.
    Returns list of buckets, each bucket is [(index, prompt), ...]
    """
    # Get lengths
    indexed = [(i, p, len(_tokenizer.encode(p, add_special_tokens=False))) 
               for i, p in enumerate(prompts)]
    
    # Sort by length
    indexed.sort(key=lambda x: x[2])
    
    # Group into buckets
    buckets = []
    for i in range(0, len(indexed), bucket_size):
        bucket = [(idx, prompt) for idx, prompt, _ in indexed[i:i + bucket_size]]
        buckets.append(bucket)
    
    return buckets


# =============================================================================
# Optimized Batch Generation
# =============================================================================
def _generate_batch_optimized(
    texts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float
) -> Tuple[List[str], BatchMetrics]:
    """
    Generate with GPU optimizations and metrics.
    """
    batch_size = len(texts)
    _reset_vram_peak()
    
    # Tokenize
    start_time = time.time()
    enc = _tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_TOK
    )
    
    total_input_tokens = enc['input_ids'].numel()
    avg_input_len = total_input_tokens / batch_size
    
    # Transfer with pinned memory + non-blocking
    if torch.cuda.is_available():
        # Pin memory
        for k in enc:
            if not enc[k].is_pinned():
                enc[k] = enc[k].pin_memory()
        
        # Transfer with copy stream
        with torch.cuda.stream(_copy_stream):
            enc = {k: v.to(_model.device, non_blocking=True) for k, v in enc.items()}
        
        # Wait for copy
        torch.cuda.current_stream().wait_stream(_copy_stream)
    else:
        enc = {k: v.to(_model.device) for k, v in enc.items()}
    
    prefill_time = (time.time() - start_time) * 1000
    
    # Generate with compute stream
    decode_start = time.time()
    
    with _semaphore:
        if torch.cuda.is_available():
            with torch.cuda.stream(_compute_stream):
                with torch.inference_mode():
                    out = _model.generate(
                        **enc,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        pad_token_id=_tokenizer.eos_token_id,
                        eos_token_id=_tokenizer.eos_token_id,
                        use_cache=True,
                    )
            torch.cuda.current_stream().wait_stream(_compute_stream)
            torch.cuda.synchronize()
        else:
            with torch.inference_mode():
                out = _model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=_tokenizer.eos_token_id,
                    eos_token_id=_tokenizer.eos_token_id,
                    use_cache=True,
                )
    
    decode_time = (time.time() - decode_start) * 1000
    total_time = prefill_time + decode_time
    
    # Decode
    decoded = _tokenizer.batch_decode(out, skip_special_tokens=True)
    
    # Clean up echo
    cleaned = []
    for text, prompt in zip(decoded, texts):
        text = text.strip()
        if text.startswith(prompt[:64]):
            text = text[len(prompt):].strip()
        cleaned.append(text)
    
    # Metrics
    total_output_tokens = out.numel() - total_input_tokens
    tokens_per_sec = total_output_tokens / (total_time / 1000) if total_time > 0 else 0
    
    metrics = BatchMetrics(
        batch_size=batch_size,
        total_input_tokens=total_input_tokens,
        avg_input_len=avg_input_len,
        prefill_time_ms=prefill_time,
        decode_time_ms=decode_time,
        tokens_per_sec=tokens_per_sec,
        vram_current_gb=_get_vram_usage(),
        vram_peak_gb=_get_vram_peak()
    )
    
    # Log metrics
    logger.info(
        f"📊 Batch: size={batch_size}, input_tokens={total_input_tokens}, "
        f"avg_len={avg_input_len:.0f}, prefill={prefill_time:.1f}ms, "
        f"decode={decode_time:.1f}ms, tok/s={tokens_per_sec:.1f}, "
        f"vram={metrics.vram_current_gb:.1f}GB (peak={metrics.vram_peak_gb:.1f}GB)"
    )
    
    return cleaned, metrics


# =============================================================================
# Public API
# =============================================================================
class EmbeddedLLMEngine:
    """Optimized embedded LLM engine."""
    
    def __init__(self, temperature: float, top_p: float, max_tokens: int):
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        _init_model(temperature, top_p, max_tokens)
        
        logger.info(
            f"🚀 EmbeddedLLMEngine initialized | "
            f"concurrency={INFER_CONCURRENCY}, batch_size={MICRO_BATCH_SIZE}, "
            f"latency={BATCH_LATENCY_MS}ms, flash_attn={USE_FLASH_ATTENTION}"
        )
    
    def generate_single(self, prompt: str) -> str:
        """Generate for a single prompt."""
        results, _ = _generate_batch_optimized(
            [prompt],
            self.max_tokens,
            self.temperature,
            self.top_p
        )
        return results[0]
    
    def generate_batch(self, prompts: List[str], progress_callback=None) -> Tuple[List[str], List[BatchMetrics]]:
        """
        Generate for multiple prompts with length-aware bucketing and concurrent processing.
        
        Args:
            prompts: List of prompts to process
            progress_callback: Optional callback(completed, total) for progress updates
            
        Returns:
            (results, metrics_per_bucket)
        """
        if not prompts:
            return [], []
        
        if len(prompts) == 1:
            results, metrics = _generate_batch_optimized(
                prompts, self.max_tokens, self.temperature, self.top_p
            )
            if progress_callback:
                progress_callback(1, 1)
            return results, [metrics]
        
        # Bucket by length
        buckets = _bucket_by_length(prompts, bucket_size=MICRO_BATCH_SIZE)
        logger.info(f"📦 Split {len(prompts)} prompts into {len(buckets)} buckets")
        
        # Process buckets concurrently
        all_results = [None] * len(prompts)
        all_metrics = []
        completed_buckets = [0]  # Mutable for closure
        
        def process_bucket(bucket):
            """Process a single bucket with semaphore control."""
            indices, bucket_prompts = zip(*bucket)
            
            # Semaphore controls GPU concurrency
            with self._semaphore:
                bucket_results, metrics = _generate_batch_optimized(
                    list(bucket_prompts),
                    self.max_tokens,
                    self.temperature,
                    self.top_p
                )
            
            return indices, bucket_results, metrics
        
        # Use ThreadPoolExecutor for concurrent bucket processing
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        with ThreadPoolExecutor(max_workers=self.infer_concurrency) as executor:
            # Submit all buckets
            futures = {executor.submit(process_bucket, bucket): i for i, bucket in enumerate(buckets)}
            
            # Collect results as they complete
            for future in as_completed(futures):
                bucket_idx = futures[future]
                try:
                    indices, bucket_results, metrics = future.result()
                    
                    # Map back to original indices
                    for idx, result in zip(indices, bucket_results):
                        all_results[idx] = result
                    
                    all_metrics.append(metrics)
                    
                    # Update progress
                    completed_buckets[0] += 1
                    if progress_callback:
                        progress_callback(completed_buckets[0], len(buckets))
                        
                except Exception as e:
                    logger.error(f"Bucket {bucket_idx} failed: {e}")
                    # Fill with error messages for this bucket
                    bucket = buckets[bucket_idx]
                    for idx, _ in bucket:
                        all_results[idx] = f"Error: {e}"
        
        return all_results, all_metrics
    
    def is_available(self) -> bool:
        """Check if engine is ready."""
        return _model is not None and _tokenizer is not None

