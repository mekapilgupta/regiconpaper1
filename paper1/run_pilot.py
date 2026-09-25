#!/usr/bin/env python3
"""
Pilot Experiment Runner: Evaluation Invariance under Multilingual Benchmark Transformation
Runs ~100–150 queries × 2 languages (Assamese, Bengali) × 6 transformations × 6 retrieval systems.
Designed for execution on the GPU compute node.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import pandas as pd
from tabulate import tabulate

# Add workspace path
PAPER1_DIR = Path(__file__).resolve().parent
if str(PAPER1_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER1_DIR))

from config import (
    DEFAULT_LANGUAGES,
    PILOT_SAMPLE_SIZE,
    EXPERIMENT_SEED,
    MODEL_PRIMARY_TRANSFORM,
    MODEL_PRIMARY_JUDGE,
    RESULTS_DIR,
    SPLITS_DIR,
    OUTPUTS_DIR,
)
from src.dataset_loader import IndicMSMARCOLoader
from src.openrouter_client import OpenRouterClient
from src.transformers import TransformationPipeline
from src.judge import TransformationJudge
from src.retrievers import RetrievalSuite
from src.evaluation import InvarianceEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Run Evaluation Invariance Pilot Study")
    parser.add_argument(
        "--languages",
        type=str,
        default="as,bn",
        help="Comma-separated language codes (default: as,bn)",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=PILOT_SAMPLE_SIZE,
        help=f"Number of queries per language for pilot (default: {PILOT_SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run offline retrievers on (cuda or cpu)",
    )
    parser.add_argument(
        "--skip_transform",
        action="store_true",
        default=False,
        help="Skip LLM transformations if cached/already generated",
    )
    parser.add_argument(
        "--skip_judge",
        action="store_true",
        default=False,
        help="Skip LLM judging step",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="pilot_experiment",
        help="Prefix name for results output",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    languages = args.languages.split(",")

    print("=" * 80)
    print(" 🧪 PILOT EXPERIMENT: Evaluation Invariance under Multilingual Benchmark Transformation")
    print("=" * 80)
    print(f"[*] Languages         : {languages}")
    print(f"[*] Sample Size / Lang: {args.sample_size}")
    print(f"[*] Transform Model   : {MODEL_PRIMARY_TRANSFORM}")
    print(f"[*] Judge Model       : {MODEL_PRIMARY_JUDGE}")
    print(f"[*] Retrieval Device  : {args.device}")
    print(f"[*] Results Directory : {RESULTS_DIR}")
    print("-" * 80)

    # 1. Dataset Loading & Deterministic Splitting
    print("\n--- [Phase 1/5] Dataset Loading & Deterministic Split ---")
    loader = IndicMSMARCOLoader(seed=EXPERIMENT_SEED)
    pilot_data, _ = loader.create_deterministic_splits(
        languages=languages,
        pilot_n=args.sample_size,
    )

    # 2. Benchmark Query Transformations (C0 - C5)
    print("\n--- [Phase 2/5] Applying Query Transformations (C0 - C5) ---")
    client = OpenRouterClient()
    pipeline = TransformationPipeline(llm_client=client)

    all_transformed_records = []
    for lang in languages:
        records = pilot_data[lang]
        cache_path = SPLITS_DIR / f"transformed_pilot_{lang}.jsonl"
        
        if args.skip_transform and cache_path.exists():
            print(f"[*] Loading existing transformed records from {cache_path}")
            with open(cache_path, "r", encoding="utf-8") as f:
                lang_transformed = [json.loads(line) for line in f if line.strip()]
        else:
            lang_transformed = pipeline.transform_dataset(records, output_path=cache_path)
        all_transformed_records.extend(lang_transformed)

    # 3. Multi-Model LLM Quality Judging
    if not args.skip_judge:
        print("\n--- [Phase 3/5] Multi-Model LLM Quality Validation ---")
        judge = TransformationJudge(llm_client=client)
        judge_path = SPLITS_DIR / "judged_pilot_records.jsonl"
        all_transformed_records = judge.evaluate_transformed_dataset(
            all_transformed_records,
            output_path=judge_path,
        )

    # 4. Offline Retrieval Evaluation (BM25, BGE-M3, mE5-base, mE5-large, LaBSE, Hybrid-RRF)
    print("\n--- [Phase 4/5] Running Offline Retrieval across all Systems & Conditions ---")
    
    # Construct fixed corpus & qrels
    corpus = []
    seen_docs = set()
    qrels = {}

    for r in all_transformed_records:
        qid = r["query_id"]
        doc_id = r["positive_passage_id"]
        doc_text = r["positive_passage"]
        
        if doc_id not in seen_docs:
            corpus.append({"doc_id": doc_id, "text": doc_text})
            seen_docs.add(doc_id)
            
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][doc_id] = 1

    print(f"[*] Fixed Retrieval Corpus size: {len(corpus)} documents.")
    print(f"[*] Total Queries evaluated: {len(all_transformed_records)}")

    # Prepare queries dictionary for retrieval
    # {query_id: {condition: query_text}}
    conditions = [
        "query_reference",
        "query_mt",
        "query_llm_translation",
        "query_postedited",
        "query_paraphrase",
        "query_synthetic",
    ]

    queries_dict = {}
    for r in all_transformed_records:
        qid = r["query_id"]
        queries_dict[qid] = {cond: r.get(cond, "") for cond in conditions}

    # Initialize & Run Retrievers
    retrieval_suite = RetrievalSuite(device=args.device)
    retrieval_suite.setup_systems(corpus)
    retrieval_results = retrieval_suite.run_all(queries_dict, top_k=10)

    # 5. Invariance Analysis & Metrics
    print("\n--- [Phase 5/5] Invariance Analysis & Statistical Reporting ---")
    evaluator = InvarianceEvaluator(qrels=qrels)
    perf_df = evaluator.evaluate_full_suite(retrieval_results, k=10)
    invariance_df = evaluator.compute_invariance_matrix(perf_df, metric="nDCG@10", ref_condition="query_reference")

    # Save Results
    results_perf_csv = RESULTS_DIR / f"{args.output_name}_performance.csv"
    results_invar_csv = RESULTS_DIR / f"{args.output_name}_invariance.csv"
    perf_df.to_csv(results_perf_csv, index=False)
    invariance_df.to_csv(results_invar_csv, index=False)

    # Print Summary Tables
    print("\n" + "=" * 80)
    print("                    RETRIEVAL PERFORMANCE (nDCG@10)")
    print("=" * 80)
    perf_pivot = perf_df.pivot(index="System", columns="Condition", values="nDCG@10")
    print(tabulate(perf_pivot, headers="keys", tablefmt="github", floatfmt=".4f"))

    print("\n" + "=" * 80)
    print("          EVALUATION INVARIANCE & SYSTEM RANKING STABILITY (vs REF)")
    print("=" * 80)
    print(tabulate(invariance_df, headers="keys", tablefmt="github", floatfmt=".4f"))

    print("\n" + "=" * 80)
    print(f"[✓] Pilot Experiment Completed Successfully!")
    print(f"[*] Performance Table saved to: {results_perf_csv}")
    print(f"[*] Invariance Table saved to  : {results_invar_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
