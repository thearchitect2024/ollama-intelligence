"""
Embedded GPU LLM client (Qwen2.5 7B) — no REST.
Implements the SAME API as the former OllamaClient:
 - generate(prompt)
 - is_available()
 - generate_async(prompt)
 - generate_batch(prompts, max_concurrent)

Features:
 - 4-bit load via bitsandbytes when available (approximate Ollama q4_0 quantization)
 - SDPA by default; FlashAttention2 if installed
 - Concurrency control via semaphore
 - Micro-batching with short latency window
 - Pinned memory + non_blocking H2D to overlap CPU/GPU
"""
import os
import time
import queue
import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import Settings

logger = logging.getLogger(__name__)

# -----------------------------
# Defaults driven by Settings
# -----------------------------
# NOTE: We read temperature, top_p, max_tokens, and max_concurrent_llm from Settings (unchanged).
# You can optionally override model/quant + throughput knobs via env without touching code.
EMBED_MODEL = os.getenv("EMBED_MODEL", "Qwen/Qwen2.5-7B-Instruct")
EMBED_DTYPE = os.getenv("EMBED_DTYPE", "auto")   # "auto" | "float16" | "bfloat16" | "float32"
USE_4BIT    = os.getenv("EMBED_4BIT", "1") == "1"

# Throughput knobs (OPTIONAL overrides; sensible defaults mirror your tuned style)
INFER_CONCURRENCY = int(os.getenv("INFER_CONCURRENCY", "8"))   # semaphore slots
MICRO_BATCH_SIZE  = int(os.getenv("MICRO_BATCH_SIZE", "32"))   # target batch size
BATCH_LATENCY_MS  = int(os.getenv("BATCH_LATENCY_MS", "120"))  # max wait to form batch

MAX_INPUT_TOK     = int(os.getenv("MAX_INPUT_TOK", "4096"))    # tokenizer truncation

# -----------------------------
# Singleton model state
# -----------------------------
_model = None
_tokenizer = None
_lock = threading.Lock()
_semaphore = threading.BoundedSemaphore(INFER_CONCURRENCY)

def _enable_sdpa_flash():
    # Prefer FlashAttention2 if available, otherwise SDPA.
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        from torch.backends.cuda import sdp_kernel
        sdp_kernel.enable_flash_sdp(True)
        sdp_kernel.enable_mem_efficient_sdp(True)
        sdp_kernel.enable_math_sdp(False)
        logger.info("⚡ Flash/Memory-efficient SDPA enabled")
    except Exception:
        logger.info("ℹ️ Using SDPA (FlashAttention2 not available)")

def _init_model(settings: Settings):
    global _model, _tokenizer
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _enable_sdpa_flash()

        quant_args = {}
        dtype = None
        
        # Try FlashAttention2, fallback to SDPA if not available
        attn_impl = "sdpa"  # default to SDPA
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
            logger.info("⚡ FlashAttention2 available, will use it")
        except ImportError:
            logger.info("ℹ️  FlashAttention2 not installed, using SDPA (still fast!)")

        if USE_4BIT:
            # Try bitsandbytes 4-bit (closest embedded analogue to Ollama q4_0)
            try:
                from transformers import BitsAndBytesConfig
                quant_args["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",     # performant default
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
                dtype = None  # dtype comes from quant config on GPU shards
                logger.info("🧱 Loading model in 4-bit (bitsandbytes)")
            except Exception as e:
                logger.warning(f"bitsandbytes not available ({e}); falling back to FP16")
                dtype = torch.float16
        else:
            # Non-quantized path (FP16 by default)
            dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
            dtype = dtype_map.get(EMBED_DTYPE, torch.float16)

        logger.info(f"🚀 Loading {EMBED_MODEL} on {device} (dtype={dtype or 'auto-4bit'})")
        _model = AutoModelForCausalLM.from_pretrained(
            EMBED_MODEL,
            device_map=device,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            **quant_args
        ).eval()

        _tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, trust_remote_code=True)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        # Try a light warmup to initialize kernels
        try:
            _ = _tokenizer("hi", return_tensors="pt")
            logger.info("✅ Embedded Qwen2.5 initialized (GPU)")
        except Exception:
            logger.info("✅ Embedded Qwen2.5 initialized")

class OllamaClient:
    """
    Embedded replacement with the SAME interface as your previous Ollama client.
    No REST; runs Qwen2.5-7B inside the process on GPU with concurrency + micro-batching.
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        self._exec = ThreadPoolExecutor(max_workers=settings.max_concurrent_llm)
        _init_model(settings)
        logger.info(
            f"Embedded LLM ready | concurrency={INFER_CONCURRENCY} "
            f"| micro_batch={MICRO_BATCH_SIZE} | latency={BATCH_LATENCY_MS}ms "
            f"| 4bit={USE_4BIT}"
        )

        # Mirror your original generation knobs (unchanged)
        self._temperature = settings.temperature        # 0.05
        self._top_p = settings.top_p                   # 0.9
        self._max_new_tokens = settings.max_tokens     # 320

    # --------- single prompt ----------
    def generate(self, prompt: str, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                return self._generate_one(prompt)
            except Exception as e:
                logger.warning(f"Generation attempt {attempt+1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(min(2 ** attempt, 4))

    def _generate_one(self, prompt: str) -> str:
        enc = _tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOK
        )
        # Pinned memory + non_blocking H2D
        if torch.cuda.is_available():
            for k, v in enc.items():
                if not v.is_pinned():
                    enc[k] = v.pin_memory()
            enc = {k: v.to(_model.device, non_blocking=True) for k, v in enc.items()}
        else:
            enc = {k: v.to(_model.device) for k, v in enc.items()}

        with _semaphore, torch.inference_mode():
            out = _model.generate(
                **enc,
                max_new_tokens=self._max_new_tokens,
                temperature=self._temperature,
                top_p=self._top_p,
                do_sample=True,
                pad_token_id=_tokenizer.eos_token_id,
                eos_token_id=_tokenizer.eos_token_id,
                use_cache=True,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        text = _tokenizer.decode(out[0], skip_special_tokens=True).strip()
        # Remove possible echo
        if text.startswith(prompt[:64]):
            text = text[len(prompt):].strip()
        return text

    def is_available(self) -> bool:
        try:
            _ = self._generate_one("ping")
            return True
        except Exception:
            return False

    # --------- async wrappers ----------
    async def generate_async(self, prompt: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._exec, self.generate, prompt)

    async def generate_batch(self, prompts: List[str], max_concurrent: int = 5) -> List[str]:
        """
        Concurrency-limited, micro-batched batch generate. Returns results in the same order as inputs.
        """
        # We'll do producer/consumer with micro-batching on each worker.
        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_with_limit(p: str) -> str:
            async with semaphore:
                return await self.generate_async(p)

        # Fast path: if batching many prompts at once, use internal batch kernel
        # to reduce decode overhead. We keep the same public method signature.
        # We'll group work in background threads to preserve API compatibility with your app.
        if len(prompts) <= 1:
            return [await run_with_limit(prompts[0])] if prompts else []

        # Micro-batch implementation using a local queue + threads (preserves ordering)
        indices = list(range(len(prompts)))
        q: queue.Queue[Tuple[int, str]] = queue.Queue()
        for i, p in zip(indices, prompts):
            q.put((i, p))

        results: List[Optional[str]] = [None] * len(prompts)
        stop = object()

        def worker():
            while True:
                batch = []
                try:
                    i, p = q.get(timeout=0.05)
                except queue.Empty:
                    break
                batch.append((i, p))

                t0 = time.time()
                while len(batch) < MICRO_BATCH_SIZE:
                    remain = BATCH_LATENCY_MS / 1000 - (time.time() - t0)
                    if remain <= 0:
                        break
                    try:
                        i2, p2 = q.get(timeout=remain)
                        batch.append((i2, p2))
                    except queue.Empty:
                        break
                idxs, texts = zip(*batch)
                try:
                    outs = self._generate_batch_now(list(texts))
                    for j, out in zip(idxs, outs):
                        results[j] = out
                finally:
                    for _ in batch:
                        q.task_done()

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(max_concurrent)]
        [t.start() for t in threads]
        q.join()
        [t.join() for t in threads]

        return [r if r is not None else "" for r in results]

    def _generate_batch_now(self, texts: List[str]) -> List[str]:
        enc = _tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_TOK
        )
        if torch.cuda.is_available():
            for k, v in enc.items():
                if not v.is_pinned():
                    enc[k] = v.pin_memory()
            enc = {k: v.to(_model.device, non_blocking=True) for k, v in enc.items()}
        else:
            enc = {k: v.to(_model.device) for k, v in enc.items()}

        with _semaphore, torch.inference_mode():
            out = _model.generate(
                **enc,
                max_new_tokens=self._max_new_tokens,
                temperature=self._temperature,
                top_p=self._top_p,
                do_sample=True,
                pad_token_id=_tokenizer.eos_token_id,
                eos_token_id=_tokenizer.eos_token_id,
                use_cache=True,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        decoded = _tokenizer.batch_decode(out, skip_special_tokens=True)
        cleaned = []
        for t, prompt in zip(decoded, texts):
            t = t.strip()
            if t.startswith(prompt[:64]):
                t = t[len(prompt):].strip()
            cleaned.append(t)
        return cleaned
