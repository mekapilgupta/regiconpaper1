#!/usr/bin/env python3
"""
STAGE 2: Offline GPU Retrieval & Invariance Evaluation (Runs on GPU Node gpu04)
Paper 1: Evaluation Invariance under Multilingual Query Transformation

Evaluates retrieval systems (BM25, BGE-M3, mE5-base, mE5-large, Qwen3-0.6B, LaBSE, Hybrid-RRF)
across all 6 conditions (C0 to C5) on both:
1. ALL_SET (All 600 queries)
2. VALIDATED_SET (Queries certified by Dual-Judge consensus)
3. Language-wise splits (Indonesian vs Thai)

Computes:
- nDCG@10, MRR@10, Recall@10, Recall@100
- 10,000 paired bootstrap CIs for system pairs
- Invariance metrics (Kendall's tau-b, Spearman's rho, mean shift Δ, rank flips)
- Retriever-Family Interaction Contrasts (Lexical vs Dense, Hybrid vs Dense)
- Behavioral Shifts (Top-10 Jaccard Overlap, RBO p=0.9, Positive Document ΔRR)
- Translation-Quality to Retrieval-Change Correlation (chrF vs Δ nDCG)
"""

import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
from tabulate import tabulate
import torch

PAPER1_DIR = Path(__file__).resolve().parent
if str(PAPER1_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER1_DIR))

from config import (
    SPLITS_DIR,
    RESULTS_DIR,
    OUTPUTS_DIR,
)
from src.retrievers import RetrievalSuite
from src.evaluation import InvarianceEvaluator
from run_analysis import generate_latex_tables


CONDITIONS = [
    "query_reference",
    "query_paraphrase",
    "query_nllb",
    "query_opus",
    "query_llm_mt",
    "query_synthetic",
]


def evaluate_dataset_slice(
    retrieval_suite: RetrievalSuite,
    records: list,
    corpus: list,
    eval_name: str,
    top_k: int = 10,
    n_bootstraps: int = 10000,
):
    print("\n" + "=" * 80)
    print(f" 📊 EVALUATING DATASET: {eval_name.upper()} ({len(records)} queries)")
    print("=" * 80)

    # 1. Ground-truth qrels & query language partition
    qrels = {}
    id_qids = set()
    th_qids = set()

    for r in records:
        qid = str(r["query_id"])
        lang = r.get("language", r.get("lang", ""))
        if lang == "id":
            id_qids.add(qid)
        elif lang == "th":
            th_qids.add(qid)

        if qid not in qrels:
            qrels[qid] = {}
        if "all_positive_docids" in r and r["all_positive_docids"]:
            for did in r["all_positive_docids"]:
                qrels[qid][str(did)] = 1
        elif "positive_passage_id" in r:
            qrels[qid][str(r["positive_passage_id"])] = 1

    # 2. Prepare queries dictionary
    queries_dict = {}
    for r in records:
        qid = str(r["query_id"])
        queries_dict[qid] = {cond: r.get(cond, "") for cond in CONDITIONS}

    # 3. Execute retrieval inference across all systems & conditions
    retrieval_results = retrieval_suite.run_all(queries_dict, top_k=100)

    # 4. Evaluation & Invariance metrics (Overall)
    evaluator = InvarianceEvaluator(qrels=qrels)
    perf_df = evaluator.evaluate_full_suite(retrieval_results, k=top_k)
    invariance_df = evaluator.compute_invariance_matrix(
        perf_df,
        metric=f"nDCG@{top_k}",
        ref_condition="query_reference",
    )

    # 5. Language-Wise Breakdown (ID vs TH)
    print(f"[*] Computing Indonesian (ID: {len(id_qids)} queries) & Thai (TH: {len(th_qids)} queries) breakdowns...")
    perf_id_df = evaluator.evaluate_full_suite(retrieval_results, k=top_k, filter_qids=id_qids)
    invar_id_df = evaluator.compute_invariance_matrix(perf_id_df, metric=f"nDCG@{top_k}")
    pair_sig_id_df = evaluator.compute_system_pair_significance(retrieval_results, metric=f"nDCG@{top_k}", filter_qids=id_qids, n_bootstraps=n_bootstraps) if id_qids else pd.DataFrame()
    interaction_id_df = evaluator.compute_retriever_family_interaction(retrieval_results, metric=f"nDCG@{top_k}", filter_qids=id_qids, n_bootstraps=n_bootstraps) if id_qids else pd.DataFrame()
    corr_id_df = evaluator.compute_translation_quality_correlation(records, retrieval_results, metric=f"nDCG@{top_k}", filter_qids=id_qids) if id_qids else pd.DataFrame()

    perf_th_df = evaluator.evaluate_full_suite(retrieval_results, k=top_k, filter_qids=th_qids)
    invar_th_df = evaluator.compute_invariance_matrix(perf_th_df, metric=f"nDCG@{top_k}")
    pair_sig_th_df = evaluator.compute_system_pair_significance(retrieval_results, metric=f"nDCG@{top_k}", filter_qids=th_qids, n_bootstraps=n_bootstraps) if th_qids else pd.DataFrame()
    interaction_th_df = evaluator.compute_retriever_family_interaction(retrieval_results, metric=f"nDCG@{top_k}", filter_qids=th_qids, n_bootstraps=n_bootstraps) if th_qids else pd.DataFrame()
    corr_th_df = evaluator.compute_translation_quality_correlation(records, retrieval_results, metric=f"nDCG@{top_k}", filter_qids=th_qids) if th_qids else pd.DataFrame()

    # 6. 10,000-Bootstrap Significance-Aware System-Pair Decision Change Analysis (Overall)
    print(f"[*] Running 10,000 paired bootstrap resamples for system-pair significance...")
    pair_significance_df = evaluator.compute_system_pair_significance(
        retrieval_results,
        metric=f"nDCG@{top_k}",
        ref_condition="query_reference",
        n_bootstraps=n_bootstraps,
    )

    # 7. Behavioral Overlap (Jaccard, RBO) & Positive Doc Rank Shift (ΔRR)
    behavior_df = evaluator.compute_behavioral_overlap_and_rank_shift(
        retrieval_results,
        ref_condition="query_reference",
        k=top_k,
        rbo_p=0.9,
    )

    # 8. Retriever-Family Interaction Contrasts (Overall)
    interaction_df = evaluator.compute_retriever_family_interaction(
        retrieval_results,
        metric=f"nDCG@{top_k}",
        ref_condition="query_reference",
        n_bootstraps=n_bootstraps,
    )

    # 9. Surface-Form Fidelity Correlation with Δ nDCG (Overall)
    corr_df = evaluator.compute_translation_quality_correlation(
        records,
        retrieval_results,
        metric=f"nDCG@{top_k}",
        ref_condition="query_reference",
    )

    # Save CSVs
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    perf_csv = RESULTS_DIR / f"{eval_name}_performance.csv"
    invar_csv = RESULTS_DIR / f"{eval_name}_invariance.csv"
    sig_csv = RESULTS_DIR / f"{eval_name}_system_pair_significance.csv"
    behavior_csv = RESULTS_DIR / f"{eval_name}_behavioral_shifts.csv"
    interaction_csv = RESULTS_DIR / f"{eval_name}_family_interactions.csv"
    corr_csv = RESULTS_DIR / f"{eval_name}_translation_correlations.csv"
    
    perf_id_csv = RESULTS_DIR / f"{eval_name}_performance_id.csv"
    invar_id_csv = RESULTS_DIR / f"{eval_name}_invariance_id.csv"
    sig_id_csv = RESULTS_DIR / f"{eval_name}_system_pair_significance_id.csv"
    interaction_id_csv = RESULTS_DIR / f"{eval_name}_family_interactions_id.csv"
    corr_id_csv = RESULTS_DIR / f"{eval_name}_translation_correlations_id.csv"

    perf_th_csv = RESULTS_DIR / f"{eval_name}_performance_th.csv"
    invar_th_csv = RESULTS_DIR / f"{eval_name}_invariance_th.csv"
    sig_th_csv = RESULTS_DIR / f"{eval_name}_system_pair_significance_th.csv"
    interaction_th_csv = RESULTS_DIR / f"{eval_name}_family_interactions_th.csv"
    corr_th_csv = RESULTS_DIR / f"{eval_name}_translation_correlations_th.csv"

    perf_df.to_csv(perf_csv, index=False)
    invariance_df.to_csv(invar_csv, index=False)
    pair_significance_df.to_csv(sig_csv, index=False)
    behavior_df.to_csv(behavior_csv, index=False)
    interaction_df.to_csv(interaction_csv, index=False)
    corr_df.to_csv(corr_csv, index=False)

    perf_id_df.to_csv(perf_id_csv, index=False)
    invar_id_df.to_csv(invar_id_csv, index=False)
    pair_sig_id_df.to_csv(sig_id_csv, index=False)
    interaction_id_df.to_csv(interaction_id_csv, index=False)
    corr_id_df.to_csv(corr_id_csv, index=False)

    perf_th_df.to_csv(perf_th_csv, index=False)
    invar_th_df.to_csv(invar_th_csv, index=False)
    pair_sig_th_df.to_csv(sig_th_csv, index=False)
    interaction_th_df.to_csv(interaction_th_csv, index=False)
    corr_th_df.to_csv(corr_th_csv, index=False)

    # Print Summaries
    print("\n" + "-" * 80)
    print(f" Retrieval Performance (nDCG@{top_k}) - {eval_name.upper()} (Overall)")
    print("-" * 80)
    perf_pivot = perf_df.pivot(index="System", columns="Condition", values=f"nDCG@{top_k}")
    print(tabulate(perf_pivot, headers="keys", tablefmt="github", floatfmt=".4f"))

    print("\n" + "-" * 80)
    print(f" Evaluation Invariance & Ranking Stability (vs REF) - {eval_name.upper()}")
    print("-" * 80)
    print(tabulate(invariance_df, headers="keys", tablefmt="github", floatfmt=".4f"))

    if not pair_significance_df.empty:
        total_pair_tests = len(pair_significance_df)
        changed_tests = pair_significance_df["Decision_Changed"].sum()
        change_pct = (changed_tests / total_pair_tests * 100) if total_pair_tests > 0 else 0.0
        print(f"\n[*] Significance-Aware System-Pair Decision Instability: {changed_tests}/{total_pair_tests} ({change_pct:.1f}% changed)")

    print("\n" + "-" * 80)
    print(f" Retriever-Family Interaction Contrasts (10,000 Bootstrap CIs)")
    print("-" * 80)
    print(tabulate(interaction_df, headers="keys", tablefmt="github", floatfmt=".4f"))

    print("\n" + "-" * 80)
    print(f" Translation Quality (chrF) vs Retrieval Delta Correlation")
    print("-" * 80)
    print(tabulate(corr_df, headers="keys", tablefmt="github", floatfmt=".4f"))

    return perf_df, invariance_df, perf_csv, invar_csv


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Offline GPU Retrieval & Evaluation")
    parser.add_argument("--device", type=str, default="cuda", help="Device (default: cuda)")
    parser.add_argument("--top_k", type=int, default=10, help="Top-K cutoff (default: 10)")
    parser.add_argument("--include_4b", action="store_true", default=False, help="Include Qwen3-Embedding-4B")
    parser.add_argument("--bootstraps", type=int, default=10000, help="Number of bootstrap resamples (default: 10000)")
    parser.add_argument("--eval_validated", action="store_true", default=True, help="Evaluate both ALL_SET and VALIDATED_SET")
    args = parser.parse_args()

    print("=" * 80)
    print(" 🚀 STAGE 2: OFFLINE GPU RETRIEVAL & INVARIANCE EVALUATION (ENHANCED)")
    print("=" * 80)

    # 1. CUDA verification
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[!] ERROR: CUDA is not available! Run this script on your GPU node (gpu04).")
        sys.exit(1)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[*] GPU Node Active   : {gpu_name} (Total VRAM: {gpu_mem:.2f} GB)")
    print(f"[*] Evaluation Cutoff  : Top-{args.top_k}")
    print(f"[*] Bootstrap Resamples: {args.bootstraps}")
    print("-" * 80)

    # 2. Load Master ALL_SET
    master_file = SPLITS_DIR / "stage1_transformed_master.jsonl"
    if not master_file.exists():
        print(f"[!] Error: Stage 1 data file not found at: {master_file}")
        sys.exit(1)

    all_records = []
    with open(master_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_records.append(json.loads(line))

    # 3. Load Unified Corpus
    corpus_file = SPLITS_DIR / "pilot_corpus.json"
    with open(corpus_file, "r", encoding="utf-8") as f:
        corpus_dict = json.load(f)
        corpus = [{"doc_id": str(k), "text": v} for k, v in corpus_dict.items()]

    print(f"[*] Fixed Retrieval Corpus Size : {len(corpus)} documents.")
    print(f"[*] Total Queries in ALL_SET    : {len(all_records)} queries.")

    # 4. Setup Retrieval Suite on A100 GPU
    print("\n--- Initializing & Indexing Retrieval Systems on GPU ---")
    retrieval_suite = RetrievalSuite(device=args.device)
    retrieval_suite.setup_systems(corpus)

    # 5. Evaluate ALL_SET
    perf_all, invar_all, perf_all_csv, invar_all_csv = evaluate_dataset_slice(
        retrieval_suite, all_records, corpus, eval_name="all_set", top_k=args.top_k, n_bootstraps=args.bootstraps
    )

    # 6. Evaluate VALIDATED_SET
    validated_file = SPLITS_DIR / "stage1_transformed_validated.jsonl"
    if validated_file.exists() and args.eval_validated:
        val_records = []
        with open(validated_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    val_records.append(json.loads(line))
        if val_records:
            perf_val, invar_val, perf_val_csv, invar_val_csv = evaluate_dataset_slice(
                retrieval_suite, val_records, corpus, eval_name="validated_set", top_k=args.top_k, n_bootstraps=args.bootstraps
            )

    # 7. Generate LaTeX Tables
    generate_latex_tables(perf_all_csv, invar_all_csv, OUTPUTS_DIR)

    print("\n" + "=" * 80)
    print(" 🎉 STAGE 2 EVALUATION COMPLETED SUCCESSFULLY!")
    print(f" [*] LaTeX Tables Generated : {OUTPUTS_DIR / 'paper1_tables.tex'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
