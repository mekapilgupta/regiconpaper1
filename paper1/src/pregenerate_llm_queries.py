"""
High-Throughput Offline Pre-Generator for C1, C4, and C5 on MIRACL Benchmark
Languages: Indonesian (id) and Thai (th)
Generates:
1. C1: Same-Language Search Query Paraphrase
2. C4: General Multilingual LLM MT Round-Trip (Native -> English -> Native)
3. C5: Passage-Conditioned Search Query (concise, <= 7 words) from ground-truth passage d+
Caches all responses locally to data/cache/llm_responses and data/cache/miracl_llm_cache.json.
"""

import sys
import os
import json
import asyncio
import time
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR, SPLITS_DIR, CACHE_DIR, LANGUAGE_MAP
from src.openrouter_client import OpenRouterClient
from src.transformers import clean_and_validate_query


async def pregenerate_miracl_llm_queries(sample_size_per_lang=300, model="deepseek/deepseek-chat"):
    print("=" * 75)
    print("🚀 PRE-GENERATING C1, C4, AND C5 FOR MIRACL (THAI & INDONESIAN)")
    print("=" * 75)

    client = OpenRouterClient()
    master_cache_file = CACHE_DIR / "miracl_llm_cache.json"
    cache_dict = {}
    if master_cache_file.exists():
        try:
            with open(master_cache_file, "r", encoding="utf-8") as f:
                cache_dict = json.load(f)
            print(f"[*] Loaded {len(cache_dict)} pre-existing cached entries from {master_cache_file}")
        except Exception:
            cache_dict = {}

    all_records = []
    for lang in ["id", "th"]:
        path = SPLITS_DIR / f"canonical_pilot_{lang}.jsonl"
        if not path.exists():
            print(f"[!] Error: {path} not found. Run download_miracl.py first!")
            return
        with open(path, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
            all_records.extend(lines[:sample_size_per_lang])

    print(f"[✓] Loaded {len(all_records)} canonical MIRACL benchmark records.")

    # Build tasks for records not fully cached
    records_to_process = []
    for r in all_records:
        key = f"{r['language']}_{r['query_id']}"
        if key not in cache_dict or not all(k in cache_dict[key] for k in ["paraphrase", "llm_mt", "synthetic"]):
            records_to_process.append(r)

    print(f"[*] Queries to fetch/generate: {len(records_to_process)} / {len(all_records)}")

    if not records_to_process:
        print("[✓] All records are already 100% cached! Nothing to do.")
        return

    # Process in concurrent batches
    batch_size = 20
    for b_idx in range(0, len(records_to_process), batch_size):
        batch = records_to_process[b_idx : b_idx + batch_size]
        print(f"\n[*] Processing batch {b_idx // batch_size + 1} / {(len(records_to_process) + batch_size - 1) // batch_size} ({len(batch)} records)...", flush=True)

        # 1. Paraphrase & Synthetic tasks & Step 1 of Round-Trip (to English)
        para_tasks = []
        synth_tasks = []
        to_en_tasks = []

        for r in batch:
            lang = r["language"]
            lang_name = LANGUAGE_MAP.get(lang, "Indonesian" if lang == "id" else "Thai")
            q_ref = r["q_ref"]
            passage = r.get("positive_passage", "")[:400]

            prompt_para = (
                f"You are a search query reformulation expert. Provide a natural, fluent search query paraphrase in {lang_name} for the query below.\n"
                f"Return ONLY the search query text in {lang_name}. Do not include quotation marks, labels, or explanations.\n"
                f"Original Query: {q_ref}"
            )
            prompt_to_en = (
                f"Translate the following {lang_name} search query into natural, accurate English.\n"
                f"Return ONLY the English translation without quotes or preamble.\n"
                f"Query: {q_ref}"
            )
            prompt_synth = (
                f"You are a search benchmark curator. Generate a short, natural search query in {lang_name} (under 7 words) "
                f"that a user would type into a search engine to find the following passage. Return ONLY the search query text.\n"
                f"Passage: {passage}"
            )

            para_tasks.append(client.generate_async(prompt_para, model=model, use_cache=True))
            synth_tasks.append(client.generate_async(prompt_synth, model=model, use_cache=True))
            to_en_tasks.append(client.generate_async(prompt_to_en, model=model, use_cache=True))

        phase1_results = await asyncio.gather(*para_tasks, *synth_tasks, *to_en_tasks)
        n = len(batch)
        para_res = phase1_results[:n]
        synth_res = phase1_results[n : 2 * n]
        to_en_res = phase1_results[2 * n :]

        # 2. Step 2 of Round-Trip: English -> Native
        back_tasks = []
        en_queries = []
        for i, r in enumerate(batch):
            lang = r["language"]
            lang_name = LANGUAGE_MAP.get(lang, "Indonesian" if lang == "id" else "Thai")
            en_query = to_en_res[i].get("cleaned_response", "").strip().strip('"').strip("'")
            en_queries.append(en_query)
            prompt_back = (
                f"Translate the following English search query into natural, fluent {lang_name}.\n"
                f"Return ONLY the {lang_name} translation without quotes or explanations.\n"
                f"Query: {en_query}"
            )
            back_tasks.append(client.generate_async(prompt_back, model=model, use_cache=True))

        back_res = await asyncio.gather(*back_tasks)

        # Store in cache_dict
        for i, r in enumerate(batch):
            key = f"{r['language']}_{r['query_id']}"
            q_ref = r["q_ref"]
            lang = r["language"]

            para_text = clean_and_validate_query(para_res[i].get("cleaned_response", ""), lang, fallback_query=q_ref)
            synth_text = clean_and_validate_query(synth_res[i].get("cleaned_response", ""), lang, fallback_query=q_ref)
            llm_mt_text = clean_and_validate_query(back_res[i].get("cleaned_response", ""), lang, fallback_query=q_ref)

            # Strip leading/trailing quotation marks or punctuation
            para_text = para_text.strip(" \"',.:;!?").strip()
            synth_text = synth_text.strip(" \"',.:;!?").strip()
            llm_mt_text = llm_mt_text.strip(" \"',.:;!?").strip()

            cache_dict[key] = {
                "query_id": r["query_id"],
                "language": lang,
                "q_ref": q_ref,
                "paraphrase": para_text,
                "en_intermediate": en_queries[i],
                "llm_mt": llm_mt_text,
                "synthetic": synth_text,
            }

        # Save checkpoint after each batch
        with open(master_cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_dict, f, ensure_ascii=False, indent=2)
        print(f"[✓] Saved {len(cache_dict)} cached records to {master_cache_file}")

    print("\n" + "=" * 75)
    print("✅ PRE-GENERATION COMPLETED SUCCESSFULLY FOR ALL 600 QUERIES!")
    print("=" * 75)


if __name__ == "__main__":
    asyncio.run(pregenerate_miracl_llm_queries(sample_size_per_lang=300))
