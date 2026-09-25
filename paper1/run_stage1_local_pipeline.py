"""
High-Precision Stage 1 Transformation & Cascaded Audit Pipeline for Paper 1
Benchmark: MIRACL (Southeast Asian Languages: Indonesian [id] and Thai [th])

Combines:
1. Offline Neural MT: NLLB-200 (C2) and OPUS-MT / M2M100 (C3)
2. High-Quality Multilingual LLM Generations: C1 (Paraphrase), C4 (LLM MT Round-Trip), C5 (Synthetic from d+)
3. Cascaded Quality Audit: Stage A (Rules) + Stage B (Local BGE-M3 Similarity)
4. Saves both Master set and Validated set.
"""

import sys
import os
import time
import argparse
import asyncio
from pathlib import Path
import json
import torch
from tqdm import tqdm

from config import (
    DATA_DIR,
    SPLITS_DIR,
    CACHE_DIR,
    DEFAULT_LANGUAGES,
    LANGUAGE_MAP,
)
from src.local_mt import LocalMTEngine
from src.judge import CascadedJudge
from src.sanity_check import run_pre_gpu_audit
from src.openrouter_client import OpenRouterClient
from src.transformers import clean_and_validate_query

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


def load_cached_llm_queries():
    """Loads pre-cached C1, C4, C5 queries from disk."""
    cache_file = CACHE_DIR / "miracl_llm_cache.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def run_local_pipeline(
    sample_size: int = 150,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    languages: list = None,
    allow_api: bool = False,
):
    languages = languages or ["id", "th"]
    start_time = time.time()
    print("=" * 75)
    print("🚀 RUNNING STAGE 1 MIRACL TRANSFORMATION & AUDIT PIPELINE (SOUTHEAST ASIA)")
    print("=" * 75)
    print(f"[*] Languages        : {languages}")
    print(f"[*] Sample size/lang : {sample_size}")
    print(f"[*] Compute device   : {device}")

    # 1. Load canonical MIRACL benchmark queries
    all_records = []
    for lang in languages:
        path = SPLITS_DIR / f"canonical_pilot_{lang}.jsonl"
        if not path.exists():
            print(f"[!] Error: {path} not found. Running MIRACL ingestion first...", flush=True)
            from src.download_miracl import download_and_prepare_miracl
            download_and_prepare_miracl(languages=[lang], sample_size=sample_size)

        with open(path, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
            all_records.extend(lines[:sample_size])

    print(f"[✓] Loaded {len(all_records)} canonical MIRACL queries ({sample_size} per lang).")

    # 2. Load C1, C4, C5 LLM Generations from cache
    llm_cache = load_cached_llm_queries()
    print(f"[*] Loaded {len(llm_cache)} pre-cached LLM queries from cache.")

    # 3. Local Neural MT Engine transformations (NLLB-200 & OPUS-MT / M2M100)
    mt_engine = LocalMTEngine(device=device)

    q_refs = [r["q_ref"] for r in all_records]
    lang_codes = [r["language"] for r in all_records]

    id_indices = [i for i, l in enumerate(lang_codes) if l == "id"]
    th_indices = [i for i, l in enumerate(lang_codes) if l == "th"]

    id_refs = [q_refs[i] for i in id_indices]
    th_refs = [q_refs[i] for i in th_indices]

    # Step A: NLLB-200 Round-Trip (C2)
    print(f"\n--- [MT Step 1/2] Running NLLB-200 Round-Trip [{len(q_refs)} queries] ---", flush=True)
    id_nllb_en = mt_engine.translate_nllb_batch(id_refs, src_lang="id", tgt_lang="en", desc="NLLB: ID -> EN") if id_refs else []
    th_nllb_en = mt_engine.translate_nllb_batch(th_refs, src_lang="th", tgt_lang="en", desc="NLLB: TH -> EN") if th_refs else []

    id_nllb_round = mt_engine.translate_nllb_batch(id_nllb_en, src_lang="en", tgt_lang="id", desc="NLLB: EN -> ID") if id_nllb_en else []
    th_nllb_round = mt_engine.translate_nllb_batch(th_nllb_en, src_lang="en", tgt_lang="th", desc="NLLB: EN -> TH") if th_nllb_en else []

    # Step B: OPUS-MT / M2M100 Round-Trip (C3)
    print(f"\n--- [MT Step 2/2] Running OPUS-MT / M2M100 Round-Trip [{len(q_refs)} queries] ---", flush=True)
    id_opus_en = mt_engine.translate_opus_batch(id_refs, src_lang="id", tgt_lang="en", desc="OPUS: ID -> EN") if id_refs else []
    th_opus_en = mt_engine.translate_opus_batch(th_refs, src_lang="th", tgt_lang="en", desc="OPUS: TH -> EN") if th_refs else []

    id_opus_round = mt_engine.translate_opus_batch(id_opus_en, src_lang="en", tgt_lang="id", desc="OPUS: EN -> ID") if id_opus_en else []
    th_opus_round = mt_engine.translate_opus_batch(th_opus_en, src_lang="en", tgt_lang="th", desc="OPUS: EN -> TH") if th_opus_en else []

    # Assemble transformed records
    transformed_records = [dict(r) for r in all_records]

    for idx, orig_idx in enumerate(id_indices):
        r = transformed_records[orig_idx]
        key = f"id_{r['query_id']}"
        cache_entry = llm_cache.get(key, {})

        r["query_reference"] = r["q_ref"]
        r["query_paraphrase"] = cache_entry.get("paraphrase", r["q_ref"])
        r["query_nllb"] = id_nllb_round[idx] if idx < len(id_nllb_round) else r["q_ref"]
        r["query_opus"] = id_opus_round[idx] if idx < len(id_opus_round) else r["q_ref"]
        r["query_llm_mt"] = cache_entry.get("llm_mt", r["query_nllb"])
        r["query_synthetic"] = cache_entry.get("synthetic", r["q_ref"])

    for idx, orig_idx in enumerate(th_indices):
        r = transformed_records[orig_idx]
        key = f"th_{r['query_id']}"
        cache_entry = llm_cache.get(key, {})

        r["query_reference"] = r["q_ref"]
        r["query_paraphrase"] = cache_entry.get("paraphrase", r["q_ref"])
        r["query_nllb"] = th_nllb_round[idx] if idx < len(th_nllb_round) else r["q_ref"]
        r["query_opus"] = th_opus_round[idx] if idx < len(th_opus_round) else r["q_ref"]
        r["query_llm_mt"] = cache_entry.get("llm_mt", r["query_nllb"])
        r["query_synthetic"] = cache_entry.get("synthetic", r["q_ref"])

    # 4. Cascaded Validation
    bge_model = None
    if SentenceTransformer is not None:
        try:
            print("\n[*] Loading BAAI/bge-m3 for free local semantic similarity validation...", flush=True)
            bge_model = SentenceTransformer("BAAI/bge-m3", device=device)
        except Exception as e:
            print(f"[!] Note: BGE-M3 validation loading: {e}", flush=True)

    judge = CascadedJudge(validator_model=bge_model, allow_api=allow_api)
    conditions_to_audit = [
        "query_paraphrase",
        "query_nllb",
        "query_opus",
        "query_llm_mt",
        "query_synthetic",
    ]
    audited_records = asyncio.run(judge.audit_dataset(transformed_records, conditions=conditions_to_audit))

    # 5. Save splits
    master_path = SPLITS_DIR / "stage1_transformed_master.jsonl"
    validated_path = SPLITS_DIR / "stage1_transformed_validated.jsonl"

    with open(master_path, "w", encoding="utf-8") as f:
        for r in audited_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Define validated records: reference, paraphrase, NLLB, and synthetic must be valid
    core_valid_records = []
    for r in audited_records:
        evals = r.get("condition_eval", {})
        is_core_valid = (
            evals.get("query_paraphrase", {}).get("valid", False)
            and evals.get("query_nllb", {}).get("valid", False)
            and evals.get("query_synthetic", {}).get("valid", False)
        )
        if is_core_valid:
            r["is_fully_validated"] = True
            core_valid_records.append(r)

    with open(validated_path, "w", encoding="utf-8") as f:
        for r in core_valid_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    elapsed = time.time() - start_time
    print(f"\n[✓] Master set saved: {master_path} ({len(audited_records)} records)")
    print(f"[✓] Validated set saved: {validated_path} ({len(core_valid_records)} / {len(audited_records)} records)")
    print(f"[✓] Pipeline completed in {elapsed:.1f}s ({elapsed/60:.1f}m)")

    # 6. Pre-GPU Sanity Audit
    run_pre_gpu_audit(master_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-Precision Stage 1 Pipeline (MIRACL)")
    parser.add_argument("--sample_size", type=int, default=150)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow_api", action="store_true", help="Allow strictly budgeted API calls")
    args = parser.parse_args()

    run_local_pipeline(
        sample_size=args.sample_size,
        device=args.device,
        allow_api=args.allow_api,
    )
