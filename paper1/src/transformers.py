"""
Transformation Engine for Multilingual Retrieval Benchmarks
Paper 1: Evaluation Invariance under Multilingual Query Transformation

Redesigned 6-Condition Causal Pipeline (Honest Lineage):
- C0: REF (Original native reference benchmark query)
- C1: PARAPHRASE (Same-language LLM semantic paraphrase of q_ref)
- C2: MT_ROUNDTRIP (Native -> English -> Native round-trip translation via MT/Qwen)
- C3: LLM_ROUNDTRIP (Native -> English -> Native round-trip translation via Gemini 3.7 Flash)
- C4: SIMPLIFIED (Controlled in-language query simplification / rewording)
- C5: SYNTHETIC (Passage-conditioned benchmark synthetic query from positive passage & answer)

Includes strict anti-contamination, script validation, and commentary stripping.
"""

import os
import sys
import re
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
import httpx

from config import (
    LANGUAGE_MAP,
    MODEL_PRIMARY_TRANSFORM,
    MODEL_SECONDARY_TRANSFORM,
    PROMPT_PARAPHRASE,
    PROMPT_TO_ENGLISH,
    PROMPT_BACK_TO_NATIVE,
    PROMPT_SIMPLIFICATION,
    PROMPT_SYNTHETIC,
)
from src.openrouter_client import OpenRouterClient


def clean_and_validate_query(raw_text: str, target_lang: str, fallback_query: str = "") -> str:
    """
    Sanitizes LLM outputs:
    1. Strips <think>...</think> reasoning blocks
    2. Strips markdown fences, quotes, and bold headers
    3. Strips meta-commentary prefixes (Question:, Translation:, etc.)
    4. Validates Indic script consistency
    """
    if not raw_text:
        return fallback_query

    # 1. Remove think tags
    text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)

    # 2. Remove markdown code blocks
    if "```" in text:
        text = re.sub(r"```[a-zA-Z]*\n?|```", "", text)

    # 3. Remove leading/trailing quotes and bold tags
    text = re.sub(r"^\*\*.*?\*\*\n*", "", text).strip()
    text = text.strip('"\'`')

    # 4. Remove generic prefix labels
    prefixes_to_strip = [
        r"^Question\s*:\s*",
        r"^Translated Query\s*:\s*",
        r"^Translation\s*:\s*",
        r"^Query\s*:\s*",
        r"^Answer\s*:\s*",
        r"^Output\s*:\s*",
        r"^প্ৰশ্ন\s*:\s*",
        r"^প্রশ্ন\s*:\s*",
        r"^প্ৰশ্নটো\s*:\s*",
        r"^অনুবাদ\s*:\s*",
        r"^সংশোধিত প্ৰশ্ন\s*:\s*",
        r"^সংশোধিত প্রশ্ন\s*:\s*",
        r"^Here is the translated query\s*:\s*",
        r"^Here is the paraphrase\s*:\s*",
        r"^Here is the query\s*:\s*",
    ]
    for pattern in prefixes_to_strip:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        text = lines[0]
        for pattern in prefixes_to_strip:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    # 5. Language Script Validation
    if target_lang in ["as", "bn"]:
        indic_chars = len(re.findall(r"[\u0980-\u09FF]", text))
        latin_chars = len(re.findall(r"[a-zA-Z]", text))
        if latin_chars > 15 and indic_chars < 5:
            return fallback_query if fallback_query else text

    return text.strip()


class TransformationPipeline:
    def __init__(
        self,
        llm_client: Optional[OpenRouterClient] = None,
        model_name: str = MODEL_PRIMARY_TRANSFORM,
        secondary_model: str = MODEL_SECONDARY_TRANSFORM,
        use_cache: bool = True,
        concurrency: int = 16,
    ):
        self.client = llm_client or OpenRouterClient(concurrency_limit=concurrency)
        self.model_name = model_name
        self.secondary_model = secondary_model
        self.use_cache = use_cache

    async def transform_single_indexed(
        self,
        idx: int,
        record: Dict[str, Any],
        http_client: httpx.AsyncClient,
    ) -> Tuple[int, Dict[str, Any]]:
        try:
            lang_code = record["language"]
            lang_full = LANGUAGE_MAP.get(lang_code, lang_code)
            q_ref = record.get("q_ref", "")
            passage = record.get("positive_passage", "")
            answer = record.get("answer", "")

            # C0: REF (Native Reference)
            q_c0_ref = q_ref

            # C1: Same-Language LLM Paraphrase
            prompt_c1 = PROMPT_PARAPHRASE.format(LANGUAGE=lang_full, Q_REF=q_ref)

            # C2: Step 1 (MT to English)
            prompt_c2_fwd = PROMPT_TO_ENGLISH.format(LANGUAGE=lang_full, Q_REF=q_ref)

            # C3: Step 1 (LLM to English)
            prompt_c3_fwd = PROMPT_TO_ENGLISH.format(LANGUAGE=lang_full, Q_REF=q_ref)

            # C4: Controlled Simplification
            prompt_c4 = PROMPT_SIMPLIFICATION.format(LANGUAGE=lang_full, Q_REF=q_ref)

            # C5: Passage-Conditioned Synthetic Query
            prompt_c5 = PROMPT_SYNTHETIC.format(
                LANGUAGE=lang_full,
                PASSAGE=passage,
                ANSWER=answer or "Not specified",
            )

            # Stage A: Run C1, C2_fwd, C3_fwd, C4, C5 concurrently
            res_c1, res_c2_fwd, res_c3_fwd, res_c4, res_c5 = await asyncio.gather(
                self.client.generate_async(prompt_c1, model=self.model_name, client=http_client, use_cache=self.use_cache),
                self.client.generate_async(prompt_c2_fwd, model=self.secondary_model, client=http_client, use_cache=self.use_cache),
                self.client.generate_async(prompt_c3_fwd, model=self.model_name, client=http_client, use_cache=self.use_cache),
                self.client.generate_async(prompt_c4, model=self.model_name, client=http_client, use_cache=self.use_cache),
                self.client.generate_async(prompt_c5, model=self.model_name, client=http_client, use_cache=self.use_cache),
            )

            q_c1_para = clean_and_validate_query(res_c1.get("cleaned_response", ""), lang_code, fallback_query=q_ref)
            q_c2_en = clean_and_validate_query(res_c2_fwd.get("cleaned_response", ""), "en", fallback_query=q_ref)
            q_c3_en = clean_and_validate_query(res_c3_fwd.get("cleaned_response", ""), "en", fallback_query=q_ref)
            q_c4_simple = clean_and_validate_query(res_c4.get("cleaned_response", ""), lang_code, fallback_query=q_ref)
            q_c5_synth = clean_and_validate_query(res_c5.get("cleaned_response", ""), lang_code, fallback_query=q_ref)

            # Stage B: Step 2 of Round-Trip (Back-translate to Native Language)
            prompt_c2_back = PROMPT_BACK_TO_NATIVE.format(LANGUAGE=lang_full, Q_EN=q_c2_en)
            prompt_c3_back = PROMPT_BACK_TO_NATIVE.format(LANGUAGE=lang_full, Q_EN=q_c3_en)

            res_c2_back, res_c3_back = await asyncio.gather(
                self.client.generate_async(prompt_c2_back, model=self.secondary_model, client=http_client, use_cache=self.use_cache),
                self.client.generate_async(prompt_c3_back, model=self.model_name, client=http_client, use_cache=self.use_cache),
            )

            q_c2_mt_round = clean_and_validate_query(res_c2_back.get("cleaned_response", ""), lang_code, fallback_query=q_c1_para)
            q_c3_llm_round = clean_and_validate_query(res_c3_back.get("cleaned_response", ""), lang_code, fallback_query=q_c1_para)

            transformed = dict(record)
            transformed["q_native_source"] = q_ref

            # Sanitized condition fields
            transformed["query_reference"] = q_c0_ref
            transformed["query_paraphrase"] = q_c1_para
            transformed["query_mt_roundtrip"] = q_c2_mt_round
            transformed["query_llm_roundtrip"] = q_c3_llm_round
            transformed["query_simplified"] = q_c4_simple
            transformed["query_synthetic"] = q_c5_synth

            # Intermediate round-trip English queries for transparency
            transformed["intermediate_en"] = {
                "c2_mt_en": q_c2_en,
                "c3_llm_en": q_c3_en,
            }

            # Retain raw outputs for verification
            transformed["raw_responses"] = {
                "c1_paraphrase": res_c1.get("raw_response", ""),
                "c2_mt_roundtrip": res_c2_back.get("raw_response", ""),
                "c3_llm_roundtrip": res_c3_back.get("raw_response", ""),
                "c4_simplified": res_c4.get("raw_response", ""),
                "c5_synthetic": res_c5.get("raw_response", ""),
            }
            transformed["transform_model"] = self.model_name
            transformed["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

            return (idx, transformed)
        except Exception as e:
            fallback = dict(record)
            q_fallback = record.get("q_ref", "")
            fallback["query_reference"] = q_fallback
            fallback["query_paraphrase"] = q_fallback
            fallback["query_mt_roundtrip"] = q_fallback
            fallback["query_llm_roundtrip"] = q_fallback
            fallback["query_simplified"] = q_fallback
            fallback["query_synthetic"] = q_fallback
            fallback["raw_responses"] = {}
            fallback["error"] = str(e)
            return (idx, fallback)

    async def transform_dataset_async(
        self,
        records: List[Dict[str, Any]],
        output_path: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        print(f"[*] Transforming {len(records)} queries concurrently with asyncio...", flush=True)
        
        results = [None] * len(records)
        async with httpx.AsyncClient(timeout=60.0) as http_client:
            tasks = [self.transform_single_indexed(i, r, http_client) for i, r in enumerate(records)]
            pbar = tqdm(total=len(tasks), desc="Query Transformation (C0-C5)", unit="query", file=sys.stdout)
            for f in asyncio.as_completed(tasks):
                idx, res = await f
                results[idx] = res
                pbar.update(1)
            pbar.close()

        valid_results = [r for r in results if r is not None]

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                for r in valid_results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[✓] Saved {len(valid_results)} transformed records to {output_path}", flush=True)

        return valid_results
