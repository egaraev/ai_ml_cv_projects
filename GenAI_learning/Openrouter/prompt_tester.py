#!/usr/bin/env python3
"""
Prompt tester script that sends the same task to Claude + Gemini + GPT-4o via OpenRouter
and prints results side by side.
"""

import asyncio
import aiohttp
import json
from typing import Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables")

# Test prompt
TEST_PROMPT = "Explain the concept of quantum computing in simple terms, then provide a practical example of how it might be used in cryptography."

# Model configurations
MODELS = {
    "claude-sonnet": {
        "name": "Claude Sonnet 4.6",
        "model": "anthropic/claude-sonnet-4.6",   # replaces claude-3.5-sonnet
        "max_tokens": 2000
    },
    "gemini-flash": {
        "name": "Gemini 2.5 Flash",
        "model": "google/gemini-2.5-flash",        # replaces gemini-1.5-pro
        "max_tokens": 2000
    },
    "gpt-4o-mini": {
        "name": "GPT-4o Mini",
        "model": "openai/gpt-4o-mini",
        "max_tokens": 2000
    }
}

# Headers for OpenRouter
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/openrouter",
    "X-Title": "Prompt Tester"
}

async def send_request(model_config: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    """Send a request to OpenRouter with a specific model."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    payload = {
        "model": model_config["model"],
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": model_config["max_tokens"],
        "temperature": 0.7
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=HEADERS, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "model": model_config["name"],
                        "success": True,
                        "response": data["choices"][0]["message"]["content"],
                        "usage": data.get("usage", {})
                    }
                else:
                    return {
                        "model": model_config["name"],
                        "success": False,
                        "error": f"HTTP {response.status}: {await response.text()}"
                    }
    except Exception as e:
        return {
            "model": model_config["name"],
            "success": False,
            "error": str(e)
        }

async def run_tests():
    """Run tests for all models concurrently."""
    print("Testing AI models with prompt:")
    print("=" * 80)
    print(TEST_PROMPT)
    print("=" * 80)
    print()
    
    # Run all requests concurrently
    tasks = [send_request(model_config, TEST_PROMPT) for model_config in MODELS.values()]
    results = await asyncio.gather(*tasks)
    
    # Print results side by side
    print("RESULTS SIDE BY SIDE:")
    print("=" * 80)
    
    # Calculate max width for model names
    max_model_width = max(len(result["model"]) for result in results)
    
    for result in results:
        print(f"\n{result['model']:<{max_model_width}}:")
        print("-" * (max_model_width + 2))
        
        if result["success"]:
            response_text = result["response"]
            # Truncate long responses for display
            if len(response_text) > 300:
                response_text = response_text[:300] + "..."
            print(response_text)
            
            # Show token usage if available
            if result.get("usage"):
                print(f"\nTokens used: {result['usage'].get('prompt_tokens', 'N/A')} prompt, "
                      f"{result['usage'].get('completion_tokens', 'N/A')} completion")
        else:
            print(f"ERROR: {result['error']}")
    
    print("\n" + "=" * 80)

async def main():
    """Main function to run the prompt tester."""
    try:
        await run_tests()
    except KeyboardInterrupt:
        print("\nTest cancelled by user.")
    except Exception as e:
        print(f"Error running tests: {e}")

if __name__ == "__main__":
    asyncio.run(main())