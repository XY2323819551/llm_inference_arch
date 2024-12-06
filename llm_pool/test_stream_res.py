import sys
import asyncio
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from llm_pool.llm import get_model_response_stream

# MODELS = ["deepseek-chat", "mixtral-8x7b-32768", "Qwen/Qwen2-72B-Instruct", "gpt-4o"]
MODELS = ["deepseek-chat", "mixtral-8x7b-32768"]



async def test_stream_for_model(model_name: str):
    print(f"\nTesting model: {model_name}")
    print("-" * 50)
    try:
        stream = await get_model_response_stream(
            model_name=model_name,
            messages=[{"role": "user", "content": "explain the following code step by step: 1+1=2"}]
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="", flush=True)
    except Exception as e:
        print(f"\nError with {model_name}: {str(e)}")
    print("\n" + "-" * 50)

async def test_all_models():
    for model in MODELS:
        await test_stream_for_model(model)

async def main():
    print("Starting stream output tests for all models:")
    await test_all_models()

if __name__ == "__main__":
    asyncio.run(main())
