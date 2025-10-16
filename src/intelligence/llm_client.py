"""
Embedded LLM Client - GPU-accelerated intelligence extraction.

Uses in-process GPU inference with FlashAttention-2, micro-batching,
and concurrent bucket processing for optimal throughput.
"""
import asyncio
import logging
from typing import List
from concurrent.futures import ThreadPoolExecutor

from src.config import Settings
from src.intelligence.embedded_llm import EmbeddedLLMEngine

logger = logging.getLogger(__name__)


class EmbeddedLLMClient:
    """
    Embedded GPU LLM client for intelligence extraction.
    
    Uses in-process GPU inference with:
    - 4-bit quantization (Q4_0)
    - FlashAttention-2 / SDPA
    - Length-aware micro-batching
    - Concurrent bucket processing
    - Pinned memory + dual CUDA streams
    - torch.compile optimization
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize the embedded LLM engine.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self._exec = ThreadPoolExecutor(max_workers=settings.max_concurrent_llm)
        
        # Initialize optimized embedded engine
        self.engine = EmbeddedLLMEngine(
            temperature=settings.temperature,
            top_p=settings.top_p,
            max_tokens=settings.max_tokens
        )
        
        logger.info(
            f"✅ OllamaClient initialized with embedded GPU engine | "
            f"workers={settings.max_concurrent_llm}"
        )
    
    def generate(self, prompt: str, max_retries: int = 3) -> str:
        """
        Generate text for a single prompt (synchronous, streaming).
        
        Submits to dispatcher queue and blocks until result ready.
        Perfect for per-request processing with ThreadPoolExecutor.
        
        Args:
            prompt: Input prompt
            max_retries: Number of retries on failure
            
        Returns:
            Generated text
        """
        import time
        
        for attempt in range(max_retries):
            try:
                return self.engine.generate(prompt)  # Uses streaming dispatcher
            except Exception as e:
                logger.warning(f"Generation attempt {attempt+1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(min(2 ** attempt, 4))
    
    def is_available(self) -> bool:
        """
        Check if the LLM engine is available.
        
        Returns:
            True if engine is ready
        """
        try:
            return self.engine.is_available()
        except Exception:
            return False
    
    async def generate_async(self, prompt: str) -> str:
        """
        Generate text asynchronously.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._exec, self.generate, prompt)
    
    async def generate_batch(self, prompts: List[str], max_concurrent: int = 5, progress_callback=None) -> List[str]:
        """
        Generate text for multiple prompts with optimized batching.
        
        This method now uses the embedded engine's length-aware bucketing
        and micro-batching for optimal GPU utilization.
        
        Args:
            prompts: List of input prompts
            max_concurrent: Max concurrent operations (preserved for API compatibility,
                          but actual batching is controlled by engine config)
            progress_callback: Optional callback(completed, total) for progress updates
            
        Returns:
            List of generated texts in same order as input
        """
        if not prompts:
            return []
        
        if len(prompts) == 1:
            result = await self.generate_async(prompts[0])
            if progress_callback:
                progress_callback(1, 1)
            return [result]
        
        # Use the optimized batch generation with progress callback
        loop = asyncio.get_event_loop()
        
        # Wrapper to pass progress_callback
        def generate_with_progress():
            return self.engine.generate_batch(prompts, progress_callback=progress_callback)
        
        results, metrics = await loop.run_in_executor(
            self._exec,
            generate_with_progress
        )
        
        # Log aggregate metrics
        if metrics:
            total_batch_size = sum(m.batch_size for m in metrics)
            total_time = sum(m.prefill_time_ms + m.decode_time_ms for m in metrics)
            avg_tok_per_sec = sum(m.tokens_per_sec for m in metrics) / len(metrics)
            
            logger.info(
                f"🎯 Batch complete: {total_batch_size} prompts, "
                f"{len(metrics)} buckets, {total_time:.0f}ms total, "
                f"avg {avg_tok_per_sec:.1f} tok/s"
            )
        
        return results


# Singleton instance
_client = None


def get_llm_client(settings: Settings = None) -> EmbeddedLLMClient:
    """
    Get or create LLM client singleton.
    
    Args:
        settings: Application settings (optional)
        
    Returns:
        EmbeddedLLMClient instance
    """
    global _client
    if _client is None:
        if settings is None:
            from src.config import get_settings
            settings = get_settings()
        _client = EmbeddedLLMClient(settings)
    return _client


# Backward compatibility alias
OllamaClient = EmbeddedLLMClient
