"""Test all available API models to find which ones actually work."""

import os
import sys
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Model list from user's config — try all available
primary = os.getenv("PRIMARY_MODEL") or os.getenv("GRAPH_MODEL") or ""
secondary = os.getenv("SECONDARY_MODEL") or "qwen/qwen3-coder-480b-a35b:free"
tertiary = os.getenv("TERTIARY_MODEL") or "google/gemma-4-31b:free"
quaternary = os.getenv("QUATERNARY_MODEL") or "qwen/qwen3-32b"

if not primary:
    primary = "meta-llama/llama-3.3-70b-instruct:free"

model_fallbacks = [primary, secondary, tertiary, quaternary]
model_fallbacks = [m for m in model_fallbacks if m]

print("=" * 70)
print("API MODEL DIAGNOSTIC TEST")
print("=" * 70)

# Test Cerebras endpoint
cerebras_key = os.getenv("CEREBRAS_API_KEY")
print(f"\nCEREBRAS_API_KEY set? {'YES' if cerebras_key else 'NO'}")
if cerebras_key:
    print(f"  (first 8 chars: {cerebras_key[:8]}...)")

cerebras_client = OpenAI(
    api_key=cerebras_key,
    base_url="https://api.cerebras.ai/v1"
)

test_prompt = "Reply with exactly: OK. Do not add anything else."

for model in model_fallbacks:
    print(f"\n  Testing model: {model!r}")
    print(f"  Endpoint: https://api.cerebras.ai/v1")
    for attempt in range(1, 4):
        try:
            t0 = time.time()
            response = cerebras_client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=10,
                messages=[{"role": "user", "content": test_prompt}],
            )
            elapsed = time.time() - t0
            content = response.choices[0].message.content
            print(f"    Attempt {attempt}: ✅ SUCCESS ({elapsed:.1f}s)")
            print(f"    Response: {content!r}")
            break
        except Exception as e:
            elapsed = time.time() - t0 if 't0' in dir() else 0
            err_str = str(e)[:120]
            print(f"    Attempt {attempt}: ❌ FAILED ({elapsed:.1f}s)")
            print(f"    Error: {err_str}")
            if attempt < 3:
                time.sleep(2)
            else:
                print(f"    → Model {model!r} FAILED after 3 attempts")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)

