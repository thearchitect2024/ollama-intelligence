"""
Quick test to verify Ollama connection with qwen2.5:7b-instruct-q4_0
"""
from src.config import get_settings
from src.intelligence.llm_client import OllamaClient

def test_ollama():
    print("=" * 80)
    print("Testing Ollama Connection")
    print("=" * 80)

    settings = get_settings()

    print(f"\n✓ Model: {settings.ollama_model}")
    print(f"✓ Base URL: {settings.ollama_base_url}")
    print(f"✓ Max Concurrent: {settings.max_concurrent_llm}")

    try:
        # Initialize client
        print("\nInitializing Ollama client...")
        client = OllamaClient(settings)

        # Test generation
        print("\nTesting generation with a simple prompt...")
        prompt = "Write a brief 50-word professional summary about a software engineer with 5 years of experience."

        response = client.generate(prompt)

        print("\n" + "=" * 80)
        print("RESPONSE:")
        print("=" * 80)
        print(response)
        print("=" * 80)

        word_count = len(response.split())
        print(f"\n✅ SUCCESS! Generated {word_count} words")
        print("\nOllama is working correctly with qwen2.5:7b-instruct-q4_0")
        print("You can now run the Streamlit app!")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nTroubleshooting:")
        print("1. Check if Ollama is running: curl http://localhost:11434/api/tags")
        print("2. Verify model exists: ollama list")
        print("3. Test model directly: ollama run qwen2.5:7b-instruct-q4_0 'Hello'")

if __name__ == "__main__":
    test_ollama()
