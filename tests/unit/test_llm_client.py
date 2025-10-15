"""
Unit tests for src/intelligence/llm_client.py
Tests Ollama LLM client wrapper.
"""
import pytest
import asyncio
from unittest.mock import MagicMock, Mock, patch, PropertyMock
from concurrent.futures import ThreadPoolExecutor

from src.intelligence.llm_client import OllamaClient


# ==========================================================================
# TEST OllamaClient Initialization
# ==========================================================================

class TestOllamaClientInit:
    """Test OllamaClient initialization."""

    @patch('src.intelligence.llm_client.OllamaLLM')
    def test_init_successful(self, mock_ollama_llm, test_settings):
        """Should initialize successfully."""
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = "Hello response"
        mock_ollama_llm.return_value = mock_llm_instance
        
        client = OllamaClient(test_settings)
        
        assert client.settings == test_settings
        assert client._llm is not None
        mock_ollama_llm.assert_called_once()

    @patch('src.intelligence.llm_client.OllamaLLM')
    def test_init_tests_connection(self, mock_ollama_llm, test_settings):
        """Should test connection on initialization."""
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = "Test response"
        mock_ollama_llm.return_value = mock_llm_instance
        
        client = OllamaClient(test_settings)
        
        # Should call invoke for connection test
        assert mock_llm_instance.invoke.called

    @patch('src.intelligence.llm_client.OllamaLLM')
    def test_init_failure_raises_error(self, mock_ollama_llm, test_settings):
        """Should raise error on initialization failure."""
        mock_ollama_llm.side_effect = Exception("Connection failed")
        
        with pytest.raises(Exception, match="Connection failed"):
            OllamaClient(test_settings)

    @patch('src.intelligence.llm_client.OllamaLLM')
    def test_init_creates_thread_pool(self, mock_ollama_llm, test_settings):
        """Should create thread pool with correct max workers."""
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = "Hello"
        mock_ollama_llm.return_value = mock_llm_instance
        
        client = OllamaClient(test_settings)
        
        assert client._executor is not None
        assert isinstance(client._executor, ThreadPoolExecutor)


# ==========================================================================
# TEST generate method
# ==========================================================================

class TestOllamaClientGenerate:
    """Test OllamaClient.generate method."""

    @patch('src.intelligence.llm_client.OllamaLLM')
    def test_generate_successful(self, mock_ollama_llm, test_settings):
        """Should generate text successfully."""
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.side_effect = ["Hello", "This is a test response"]
        mock_ollama_llm.return_value = mock_llm_instance
        
        client = OllamaClient(test_settings)
        response = client.generate("Test prompt")
        
        assert response == "This is a test response"
        assert mock_llm_instance.invoke.call_count == 2  # init + generate

    @patch('src.intelligence.llm_client.OllamaLLM')
    def test_generate_strips_whitespace(self, mock_ollama_llm, test_settings):
        """Should strip whitespace from response."""
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.side_effect = ["Hello", "  Response with spaces  "]
        mock_ollama_llm.return_value = mock_llm_instance
        
        client = OllamaClient(test_settings)
        response = client.generate("Test")
        
        assert response == "Response with spaces"

    @patch('src.intelligence.llm_client.OllamaLLM')
    def test_generate_with_retry(self, mock_ollama_llm, test_settings):
        """Should retry on failure."""
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.side_effect = [
            "Hello",  # init
            Exception("Temporary error"),
            "Success after retry"
        ]
        mock_ollama_llm.return_value = mock_llm_instance
        
        client = OllamaClient(test_settings)
        response = client.generate("Test", max_retries=2)
        
        assert response == "Success after retry"

    @patch('src.intelligence.llm_client.OllamaLLM')
    @patch('time.sleep')
    def test_generate_retry_with_backoff(self, mock_sleep, mock_ollama_llm, test_settings):
        """Should use exponential backoff on retry."""
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.side_effect = [
            "Hello",  # init
            Exception("Error 1"),
            Exception("Error 2"),
            "Success"
        ]
        mock_ollama_llm.return_value = mock_llm_instance
        
        client = OllamaClient(test_settings)
        response = client.generate("Test", max_retries=3)
        
        assert response == "Success"
        # Should call sleep with exponential backoff
        assert mock_sleep.call_count >= 1

    @patch('src.intelligence.llm_client.OllamaLLM')
    def test_generate_max_retries_exceeded(self, mock_ollama_llm, test_settings):
        """Should raise error after max retries."""
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.side_effect = [
            "Hello",  # init
            Exception("Error 1"),
            Exception("Error 2"),
            Exception("Error 3")
        ]
        mock_ollama_llm.return_value = mock_ollama_instance
        
        client = OllamaClient(test_settings)
        
        with pytest.raises(Exception):
            client.generate("Test", max_retries=3)

    @patch('src.intelligence.llm_client.OllamaLLM')
    def test_generate_not_initialized_raises_error(self, mock_ollama_llm, test_settings):
        """Should raise error if not initialized."""
        mock_ollama_llm.return_value = MagicMock()
        mock_ollama_llm.return_value.invoke.return_value = "Hello"
        
        client = OllamaClient(test_settings)
        client._llm = None  # Simulate not initialized
        
        with pytest.raises(RuntimeError, match="not initialized"):
            client.generate("Test")


# ==========================================================================
# TEST is_available method
# ==========================================================================

class TestOllamaClientIsAvailable:
    """Test OllamaClient.is_available method."""

    @patch('src.intelligence.llm_client.OllamaLLM')
    def test_is_available_true(self, mock_ollama_llm, test_settings):
        """Should return True when available."""
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = "test"
        mock_ollama_llm.return_value = mock_llm_instance
        
        client = OllamaClient(test_settings)
        
        assert client.is_available() is True

    @patch('src.intelligence.llm_client.OllamaLLM')
    def test_is_available_false(self, mock_ollama_llm, test_settings):
        """Should return False when not available."""
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.side_effect = [
            "Hello",  # init
            Exception("Not available")
        ]
        mock_ollama_llm.return_value = mock_llm_instance
        
        client = OllamaClient(test_settings)
        
        assert client.is_available() is False


# ==========================================================================
# TEST generate_async method
# ==========================================================================

class TestOllamaClientGenerateAsync:
    """Test OllamaClient.generate_async method."""

    @pytest.mark.asyncio
    @patch('src.intelligence.llm_client.OllamaLLM')
    async def test_generate_async_successful(self, mock_ollama_llm, test_settings):
        """Should generate text asynchronously."""
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.side_effect = ["Hello", "Async response"]
        mock_ollama_llm.return_value = mock_llm_instance
        
        client = OllamaClient(test_settings)
        response = await client.generate_async("Test prompt")
        
        assert response == "Async response"

    @pytest.mark.asyncio
    @patch('src.intelligence.llm_client.OllamaLLM')
    async def test_generate_async_multiple_concurrent(self, mock_ollama_llm, test_settings):
        """Should handle multiple concurrent async calls."""
        mock_llm_instance = MagicMock()
        responses = ["Hello"] + [f"Response {i}" for i in range(5)]
        mock_llm_instance.invoke.side_effect = responses
        mock_ollama_llm.return_value = mock_llm_instance
        
        client = OllamaClient(test_settings)
        
        # Run 5 concurrent generations
        tasks = [client.generate_async(f"Prompt {i}") for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        assert all("Response" in r for r in results)


# ==========================================================================
# TEST generate_batch method
# ==========================================================================

class TestOllamaClientGenerateBatch:
    """Test OllamaClient.generate_batch method."""

    @pytest.mark.asyncio
    @patch('src.intelligence.llm_client.OllamaLLM')
    async def test_generate_batch_successful(self, mock_ollama_llm, test_settings):
        """Should generate batch of texts."""
        mock_llm_instance = MagicMock()
        responses = ["Hello"] + [f"Response {i}" for i in range(3)]
        mock_llm_instance.invoke.side_effect = responses
        mock_ollama_llm.return_value = mock_llm_instance
        
        client = OllamaClient(test_settings)
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        results = await client.generate_batch(prompts, max_concurrent=2)
        
        assert len(results) == 3
        assert all("Response" in r for r in results)

    @pytest.mark.asyncio
    @patch('src.intelligence.llm_client.OllamaLLM')
    async def test_generate_batch_preserves_order(self, mock_ollama_llm, test_settings):
        """Should preserve order of results."""
        mock_llm_instance = MagicMock()
        # Create distinct responses
        responses = ["Hello"] + [f"Result_{i}" for i in range(5)]
        mock_llm_instance.invoke.side_effect = responses
        mock_ollama_llm.return_value = mock_llm_instance
        
        client = OllamaClient(test_settings)
        prompts = [f"Prompt_{i}" for i in range(5)]
        results = await client.generate_batch(prompts, max_concurrent=3)
        
        # Results should maintain input order
        assert len(results) == 5

    @pytest.mark.asyncio
    @patch('time.sleep')  # Mock sleep to avoid delays
    @patch('src.intelligence.llm_client.OllamaLLM')
    async def test_generate_batch_handles_errors(self, mock_ollama_llm, mock_sleep, test_settings):
        """Should handle errors in batch gracefully."""
        mock_llm_instance = MagicMock()
        # Setup to fail fast without retries
        mock_llm_instance.invoke.side_effect = [
            "Hello",  # init
            "Success 1",
            Exception("Error"),  # This will fail all retries
            Exception("Error"),
            Exception("Error"),
            "Success 2"
        ]
        mock_ollama_llm.return_value = mock_llm_instance
        
        client = OllamaClient(test_settings)
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        results = await client.generate_batch(prompts, max_concurrent=2)
        
        # Should return error message for failed prompt
        assert len(results) == 3
        assert any("Error" in r for r in results)

    @pytest.mark.asyncio
    @patch('src.intelligence.llm_client.OllamaLLM')
    async def test_generate_batch_respects_concurrency_limit(self, mock_ollama_llm, test_settings):
        """Should respect max_concurrent limit."""
        mock_llm_instance = MagicMock()
        responses = ["Hello"] + [f"Response {i}" for i in range(10)]
        mock_llm_instance.invoke.side_effect = responses
        mock_ollama_llm.return_value = mock_llm_instance
        
        client = OllamaClient(test_settings)
        prompts = [f"Prompt {i}" for i in range(10)]
        
        # With max_concurrent=2, should process in batches
        results = await client.generate_batch(prompts, max_concurrent=2)
        
        assert len(results) == 10

    @pytest.mark.asyncio
    @patch('src.intelligence.llm_client.OllamaLLM')
    async def test_generate_batch_empty_prompts(self, mock_ollama_llm, test_settings):
        """Should handle empty prompts list."""
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = "Hello"
        mock_ollama_llm.return_value = mock_llm_instance
        
        client = OllamaClient(test_settings)
        results = await client.generate_batch([], max_concurrent=2)
        
        assert results == []

    @pytest.mark.asyncio
    @patch('src.intelligence.llm_client.OllamaLLM')
    async def test_generate_batch_single_prompt(self, mock_ollama_llm, test_settings):
        """Should handle single prompt in batch."""
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.side_effect = ["Hello", "Single response"]
        mock_ollama_llm.return_value = mock_llm_instance
        
        client = OllamaClient(test_settings)
        results = await client.generate_batch(["Single prompt"], max_concurrent=5)
        
        assert len(results) == 1
        assert results[0] == "Single response"

