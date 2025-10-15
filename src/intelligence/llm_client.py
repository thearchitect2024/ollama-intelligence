"""
Ollama LLM client wrapper for intelligence extraction.
Provides simplified interface for text generation with async batch processing support.
"""
import logging
import time
import asyncio
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor
from langchain_ollama import OllamaLLM

from src.config import Settings

logger = logging.getLogger(__name__)


class OllamaClient:
    """Wrapper for Ollama LLM with LangChain and async batch processing support."""

    def __init__(self, settings: Settings):
        """
        Initialize Ollama client with thread pool for async operations.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self._llm: Optional[OllamaLLM] = None
        self._executor = ThreadPoolExecutor(max_workers=settings.max_concurrent_llm)
        self._initialize()

    def _initialize(self):
        """Initialize Ollama LLM connection."""
        try:
            logger.info(f"Initializing Ollama model: {self.settings.ollama_model}")

            self._llm = OllamaLLM(
                model=self.settings.ollama_model,
                base_url=self.settings.ollama_base_url,
                temperature=self.settings.temperature,
                top_p=self.settings.top_p,
                num_predict=self.settings.max_tokens
            )

            # Test connection
            test_response = self._llm.invoke("Hello")
            logger.info(f"Ollama connected successfully. Test: {test_response[:50]}...")

        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")
            raise

    def generate(self, prompt: str, max_retries: int = 3) -> str:
        """
        Generate text from prompt with retry logic.

        Args:
            prompt: Input prompt
            max_retries: Maximum retry attempts

        Returns:
            str: Generated text
        """
        if not self._llm:
            raise RuntimeError("Ollama client not initialized")

        for attempt in range(max_retries):
            try:
                start_time = time.time()

                response = self._llm.invoke(prompt)
                elapsed = time.time() - start_time

                logger.info(f"Generation completed in {elapsed:.2f}s ({len(response.split())} words)")
                return response.strip()

            except Exception as e:
                logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff

        raise RuntimeError("Max retries exceeded")

    def is_available(self) -> bool:
        """Check if Ollama service is available."""
        try:
            self._llm.invoke("test")
            return True
        except Exception:
            return False

    async def generate_async(self, prompt: str) -> str:
        """
        Async wrapper for generate() method.

        Args:
            prompt: Input prompt

        Returns:
            str: Generated text
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.generate, prompt)

    async def generate_batch(self, prompts: List[str], max_concurrent: int = 5) -> List[str]:
        """
        Process multiple prompts concurrently with controlled concurrency.

        Args:
            prompts: List of input prompts
            max_concurrent: Maximum number of concurrent generations

        Returns:
            List[str]: List of generated texts (same order as input)
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_limit(prompt: str) -> str:
            async with semaphore:
                try:
                    return await self.generate_async(prompt)
                except Exception as e:
                    logger.error(f"Batch generation failed for prompt: {e}")
                    return f"Error: {str(e)}"

        # Process all prompts concurrently (with semaphore limiting)
        tasks = [process_with_limit(p) for p in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        logger.info(f"Batch generation completed: {len(results)} summaries")
        return results
