#!/usr/bin/env python3
"""
Diagnostic & Test Tool for OpenRouter Thinking Models
Tests:
1. openai/gpt-5.6-luna
2. qwen/qwen3.8-flash
3. google/gemini-3.8-flash
4. deepseek/deepseek-v4-pro
"""

import os
import sys
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

PAPER1_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=PAPER1_DIR / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

MODELS = [
    "openai/gpt-5.6-luna",
    "qwen/qwen3.8-flash",
    "google/gemini-3.8-flash",
    "deepseek/deepseek-v4-pro",
]


def test_model(model_name: str):
    print("=" * 70)
    print(f"[*] Testing Model: {model_name}")
    print("=" * 70)

    if not OPENROUTER_API_KEY:
        print("[!] Error: OPENROUTER_API_KEY is not set in paper1/.env")
        return False

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/indic-ir/invariance-study",
        "X-Title": "Evaluation Invariance Study",
    }

    # Test query translation prompt
    prompt = (
        "Translate the following search query into Bengali. "
        "Produce ONLY the translated query without any explanation.\n\n"
        "QUERY: what is the capital of assam"
    )

    # 1. Payload with high max_tokens to accommodate thinking tokens
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 4096,  # Generous budget for reasoning + output
    }

    endpoint = f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions"

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=90)
        print(f"[*] HTTP Status Code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"[!] Raw Response Error: {response.text}")
            return False

        data = response.json()
        if "error" in data:
            print(f"[!] Error in JSON body: {data['error']}")
            return False

        choices = data.get("choices", [])
        if not choices:
            print(f"[!] Empty choices in response: {data}")
            return False

        msg = choices[0].get("message", {})
        content = msg.get("content")
        reasoning = msg.get("reasoning")
        usage = data.get("usage", {})

        print(f"[✓] Token Usage: {usage}")
        if reasoning:
            print(f"[*] Reasoning tokens detected ({len(reasoning)} chars)")
        print(f"[✓] Output Content: {content}")
        return True

    except Exception as e:
        print(f"[!] Exception occurred: {e}")
        return False


def main():
    print("=" * 70)
    print(" OpenRouter Thinking Models Connectivity & Payload Test")
    print("=" * 70)
    print(f"[*] API Base: {OPENROUTER_BASE_URL}")
    print(f"[*] API Key : {OPENROUTER_API_KEY[:8]}..." if OPENROUTER_API_KEY else "[!] NO KEY")
    print("-" * 70)

    results = {}
    for m in MODELS:
        success = test_model(m)
        results[m] = "PASS" if success else "FAIL"
        print()

    print("=" * 70)
    print("                     SUMMARY")
    print("=" * 70)
    for m, status in results.items():
        print(f" {status:6s} : {m}")
    print("=" * 70)


if __name__ == "__main__":
    main()
