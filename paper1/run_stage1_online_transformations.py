"""
Stage 1: Online Query Transformations & Dual-Judge Quality Audit (Redesigned Experiment)
Paper 1: Evaluation Invariance under Multilingual Query Transformation

Executes on CPU/Login Node:
1. Loads 150 AS and 150 BN pilot queries with deduplicated MD5 passage corpus
2. Generates 6 honest causal conditions:
   - C0: Native Reference (q_ref)
   - C1: Same-Language LLM Paraphrase
   - C2: Native -> English -> Native MT Round-Trip
   - C3: Native -> English -> Native LLM Round-Trip
   - C4: Controlled Simplification / Rewording
   - C5: Passage-Conditioned Synthetic Query
3. Runs Dual Independent Judge consensus (Gemini 3.8 Flash + Qwen 3.8 Flash + DeepSeek Arbiter)
4. Produces ALL_SET (300 queries) and VALIDATED_SET (Certified valid across all conditions)
5. Runs Pre-GPU Sanity Audit and prints 10 random sample comparisons
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path

# Ensure unbuffered terminal output
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import (
    DATA_DIR,
    SPLITS_DIR,
    DEFAULT_LANGUAGES,
    PILOT_SAMPLE_SIZE,
    MODEL_PRIMARY_TRANSFORM,
    MODEL_SECONDARY_TRANSFORM,
    MODEL_PRIMARY_JUDGE,
    MODEL_SECONDARY_JUDGE,
    MODEL_ARBITER,
)
from src.dataset_loader import IndicMSMARCOLoader
from src.transformers import TransformationPipeline
from src.judge import DualTransformationJudge
from src.sanity_check import run_pre_gpu_sanity_check


async def main_async(args):
    print("=" * 80, flush=True)
    print(" 🚀 STAGE 1: ONLINE QUERY TRANSFORMATIONS & DUAL-JUDGE AUDIT (REDESIGNED)", flush=True)
    print("=" * 80, flush=True)
    print(f"[*] Languages          : {args.languages}", flush=True)
    print(f"[*] Queries per Lang   : {args.sample_size}", flush=True)
    print(f"[*] Transformation LLM : {args.model} (Primary) & {args.secondary_model} (MT)", flush=True)
    print(f"[*] Dual Judges        : {args.judge_a} & {args.judge_b} (Arbiter: {args.arbiter})", flush=True)
    print(f"[*] Async Concurrency  : {args.concurrency}", flush=True)
    print("-" * 80, flush=True)

    # 1. Dataset Loading & Deduplicated Corpus Preparation
    print("\n--- [Step 1/3] Preparing Canonical Dataset & Deduplicated Corpus ---", flush=True)
    loader = IndicMSMARCOLoader()
    pilot_data, corpus, qrels = loader.build_pilot_splits(
        sample_size=args.sample_size,
        languages=args.languages,
    )

    all_queries = []
    for lang in args.languages:
        all_queries.extend(pilot_data[lang])

    print(f"[✓] Total active queries to transform: {len(all_queries)} ({len(all_queries)//len(args.languages)} per lang)", flush=True)

    # 2. Asynchronous Multi-Condition Transformations (C0 - C5)
    print("\n--- [Step 2/3] Generating Transformations (C0 to C5) via Async OpenRouter ---", flush=True)
    transformer = TransformationPipeline(
        model_name=args.model,
        secondary_model=args.secondary_model,
        concurrency=args.concurrency,
    )
    transformed_records = await transformer.transform_dataset_async(all_queries)

    # 3. Dual-Judge Consensus & Arbiter Quality Audit
    print("\n--- [Step 3/3] Dual Independent Judge Consensus Audit ---", flush=True)
    judge = DualTransformationJudge(
        judge_a_model=args.judge_a,
        judge_b_model=args.judge_b,
        arbiter_model=args.arbiter,
        concurrency=args.concurrency,
    )
    audited_records = await judge.evaluate_dataset_async(transformed_records)

    # Save Master ALL_SET File
    master_file = SPLITS_DIR / "stage1_transformed_master.jsonl"
    with open(master_file, "w", encoding="utf-8") as f:
        for r in audited_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[✓] ALL_SET saved to: {master_file}", flush=True)

    # Build and Save VALIDATED_SET (Filtered by strict judge validity across all transformed conditions)
    validated_records = []
    for r in audited_records:
        judgements = r.get("judgements", {})
        all_valid = True
        for cond in ["query_paraphrase", "query_mt_roundtrip", "query_llm_roundtrip", "query_simplified", "query_synthetic"]:
            j = judgements.get(f"judge_{cond}", {})
            if j.get("overall_validity", "").upper() != "YES":
                all_valid = False
                break
        if all_valid:
            validated_records.append(r)

    validated_file = SPLITS_DIR / "stage1_transformed_validated.jsonl"
    with open(validated_file, "w", encoding="utf-8") as f:
        for r in validated_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[✓] VALIDATED_SET saved to: {validated_file} ({len(validated_records)}/{len(audited_records)} queries)", flush=True)

    # 4. Pre-GPU Sanity Audit & Inspection
    print("\n--- Running Pre-GPU Scientific Integrity & Sanity Audit ---", flush=True)
    passed = run_pre_gpu_sanity_check(master_file)

    if passed:
        print("\n" + "=" * 80, flush=True)
        print(" 🎉 STAGE 1 COMPLETED SUCCESSFULLY WITH 100% SCIENTIFIC COMPLIANCE!", flush=True)
        print("    You can now proceed to Stage 2 GPU Retrieval on your GPU node.", flush=True)
        print("=" * 80, flush=True)
    else:
        print("\n[!] Please inspect the sanity report before running Stage 2.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Online Query Transformations & Dual-Judge Validation")
    parser.add_argument("--languages", nargs="+", default=DEFAULT_LANGUAGES, help="Languages (e.g. as bn)")
    parser.add_argument("--sample_size", type=int, default=PILOT_SAMPLE_SIZE, help="Queries per language (default: 150)")
    parser.add_argument("--model", type=str, default="google/gemini-3.7-flash", help="Primary transform model (Strictly Gemini 3.7 Flash)")
    parser.add_argument("--secondary_model", type=str, default="qwen/qwen3.8-flash", help="Secondary/MT model")
    parser.add_argument("--judge_a", type=str, default="google/gemini-3.8-flash", help="Judge A model (Strictly Gemini 3.8 Flash)")
    parser.add_argument("--judge_b", type=str, default="qwen/qwen3.8-flash", help="Judge B model")
    parser.add_argument("--arbiter", type=str, default="deepseek/deepseek-chat", help="Arbiter model")
    parser.add_argument("--concurrency", type=int, default=16, help="Async concurrent workers (default: 16)")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
