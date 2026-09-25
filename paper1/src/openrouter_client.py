"""
Async High-Throughput OpenRouter API Client with Thinking Model Support
Strictly configured for:
- Primary Transformation: google/gemini-3.7-flash
- Primary Judge: google/gemini-3.8-flash
- Independent Secondary/Judge B: qwen/qwen3.8-flash
- Arbiter: deepseek/deepseek-chat

Handles reasoning tokens (<think>...</think>, reasoning field), async pooling,
and persistent response caching.
"""

import os
import re
import json
import time
import random
import hashlib
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
import httpx

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    CACHE_DIR,
)


class OpenRouterClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        max_retries: int = 3,
        timeout: float = 30.0,
        concurrency_limit: int = 16,
    ):
        self.api_key = api_key or OPENROUTER_API_KEY
        self.base_url = base_url or OPENROUTER_BASE_URL
        self.cache_dir = cache_dir or (CACHE_DIR / "llm_responses")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.timeout = timeout
        self.concurrency_limit = concurrency_limit
        self._semaphore = None

        # Fallback chain strictly avoiding gemini-2.5-flash
        self.model_fallbacks = {
            "google/gemini-3.7-flash": ["qwen/qwen3.8-flash", "deepseek/deepseek-chat"],
            "google/gemini-3.8-flash": ["qwen/qwen3.8-flash", "deepseek/deepseek-chat"],
            "qwen/qwen3.8-flash": ["deepseek/deepseek-chat", "google/gemini-3.7-flash"],
            "openai/gpt-5.6-luna": ["google/gemini-3.7-flash", "qwen/qwen3.8-flash"],
            "deepseek/deepseek-chat": ["qwen/qwen3.8-flash", "google/gemini-3.7-flash"],
        }

        if not self.api_key:
            print("[!] Warning: OPENROUTER_API_KEY is not set in paper1/.env", flush=True)

    @property
    def semaphore(self):
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.concurrency_limit)
        return self._semaphore

    def _get_cache_key(self, model: str, prompt: str) -> str:
        content = f"{model}::{prompt}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _read_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    val = data.get("cleaned_response")
                    if val is not None and len(str(val).strip()) > 0:
                        return data
            except Exception:
                return None
        return None

    def _write_cache(self, cache_key: str, model: str, prompt: str, raw_response: str, cleaned_response: str):
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({
                    "model": model,
                    "prompt": prompt,
                    "raw_response": raw_response,
                    "cleaned_response": cleaned_response,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def clean_output(self, text: str) -> str:
        """Strips <think>...</think> blocks, markdown fences, and preamble."""
        if not text:
            return ""
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
        return cleaned.strip()

    async def generate_async(
        self,
        prompt: str,
        model: str = "google/gemini-3.7-flash",
        temperature: Optional[float] = None,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
        use_cache: bool = True,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Dict[str, Any]:
        cache_key = self._get_cache_key(model, prompt)
        if use_cache:
            cached = self._read_cache(cache_key)
            if cached is not None:
                return cached

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/indic-ir/invariance-study",
            "X-Title": "Evaluation Invariance Study",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature

        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
        candidate_models = [model] + self.model_fallbacks.get(model, ["qwen/qwen3.8-flash"])

        close_client = False
        if client is None:
            client = httpx.AsyncClient(timeout=self.timeout)
            close_client = True

        try:
            for current_model in candidate_models:
                payload["model"] = current_model
                for attempt in range(1, self.max_retries + 1):
                    async with self.semaphore:
                        try:
                            resp = await client.post(endpoint, json=payload, headers=headers)
                            if resp.status_code == 200:
                                data = resp.json()
                                choices = data.get("choices", [])
                                if choices:
                                    msg = choices[0].get("message", {})
                                    raw_text = msg.get("content") or msg.get("reasoning") or ""
                                    cleaned_text = self.clean_output(str(raw_text))
                                    if cleaned_text:
                                        result = {
                                            "model": current_model,
                                            "raw_response": str(raw_text),
                                            "cleaned_response": cleaned_text,
                                        }
                                        if use_cache:
                                            self._write_cache(cache_key, current_model, prompt, str(raw_text), cleaned_text)
                                        return result
                            elif resp.status_code in [401, 402]:
                                err_msg = resp.text
                                print(f"\n[!] OpenRouter API Fatal Error ({resp.status_code}): {err_msg}", flush=True)
                                return {
                                    "model": current_model,
                                    "raw_response": "",
                                    "cleaned_response": "",
                                    "error": f"HTTP {resp.status_code}: {err_msg}",
                                }
                            elif resp.status_code == 429:
                                await asyncio.sleep(1.0 + random.uniform(0.5, 1.0))
                            else:
                                if attempt == self.max_retries:
                                    print(f"[!] OpenRouter error on {current_model} (attempt {attempt}): {resp.status_code} - {resp.text[:120]}", flush=True)
                                await asyncio.sleep(0.5)
                        except Exception as e:
                            if attempt == self.max_retries:
                                print(f"[!] Exception calling OpenRouter on {current_model}: {e}", flush=True)
                            await asyncio.sleep(0.5)

            return {
                "model": model,
                "raw_response": "",
                "cleaned_response": "",
                "error": "All model attempts failed",
            }
        finally:
            if close_client:
                await client.aclose()
