"""
Diagnostics Suite for Paper 1
1. q_en_source vs q_ref semantic-validity rate (AS vs BN).
2. Per-condition semantic-validity counts and rates.
3. Separate AS and BN retrieval performance tables.
4. Bootstrap significance-aware pairwise comparison matrix for C0 vs every condition.
5. 20 Random diagnostic cases where C1/C2/C3 show major rank drops with judge annotations.
"""

import sys
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tabulate import tabulate
from scipy import stats
from typing import List, Tuple, Dict, Any

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import SPLITS_DIR, RESULTS_DIR, OUTPUTS_DIR
from src.evaluation import compute_ndcg_at_k, compute_mrr_at_k, compute_recall_at_k
from src.retrievers import BM25Retriever


def bootstrap_paired_test(
    scores_a: List[float],
    scores_b: List[float],
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, Tuple[float, float], float]:
    """
    Computes paired difference, p-value (paired t-test), and 95% bootstrap CI.
    """
    arr_a = np.array(scores_a)
    arr_b = np.array(scores_b)
    diffs = arr_b - arr_a
    mean_diff = float(np.mean(diffs))

    # Paired t-test
    _, p_val = stats.ttest_rel(arr_a, arr_b)

    # Bootstrap CI
    rng = np.random.RandomState(seed)
    boot_means = []
    n = len(diffs)
    for _ in range(n_bootstraps):
        sample_diffs = rng.choice(diffs, size=n, replace=True)
        boot_means.append(np.mean(sample_diffs))

    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    return mean_diff, float(p_val), (ci_lower, ci_upper), float(np.std(diffs))


def run_all_diagnostics():
    master_file = SPLITS_DIR / "stage1_transformed_master.jsonl"
    corpus_file = SPLITS_DIR / "pilot_corpus.json"
    qrels_file = SPLITS_DIR / "pilot_qrels.json"

    if not master_file.exists():
        print(f"[!] Master file not found at {master_file}")
        return

    records = []
    with open(master_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    with open(corpus_file, "r", encoding="utf-8") as f:
        corpus_dict = json.load(f)
        corpus = [{"doc_id": k, "text": v} for k, v in corpus_dict.items()]

    with open(qrels_file, "r", encoding="utf-8") as f:
        qrels = json.load(f)

    as_records = [r for r in records if r.get("language") == "as"]
    bn_records = [r for r in records if r.get("language") == "bn"]

    print("=" * 80)
    print(" 🔬 PAPER 1: FOUR CORE DIAGNOSTICS & ALIGNMENT AUDIT")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Diagnostic 1 & 2: Semantic Validity Rates (Overall and AS vs BN)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print(" [Diagnostic 1 & 2] Per-Condition Semantic Validity & Cross-Language Rates")
    print("-" * 80)

    conditions = [
        "query_mt",
        "query_llm_translation",
        "query_postedited",
        "query_paraphrase",
        "query_synthetic",
    ]

    validity_rows = []
    for cond in conditions:
        for lang_name, lang_recs in [("All", records), ("Assamese (AS)", as_records), ("Bengali (BN)", bn_records)]:
            valid_count = 0
            agree_count = 0
            intent_scores = []
            entity_scores = []

            for r in lang_recs:
                j = r.get("judgements", {}).get(f"judge_{cond}", {})
                if j:
                    if j.get("overall_validity", "").upper() == "YES":
                        valid_count += 1
                    if j.get("agreement", False):
                        agree_count += 1
                    intent_scores.append(j.get("intent_preservation_score", 4))
                    entity_scores.append(j.get("entity_fidelity_score", 4))

            total = len(lang_recs)
            valid_pct = (valid_count / total * 100) if total > 0 else 0
            agree_pct = (agree_count / total * 100) if total > 0 else 0
            mean_intent = np.mean(intent_scores) if intent_scores else 0
            mean_entity = np.mean(entity_scores) if entity_scores else 0

            validity_rows.append({
                "Condition": cond,
                "Language": lang_name,
                "Valid Count": f"{valid_count}/{total}",
                "Valid (%)": f"{valid_pct:.1f}%",
                "Judge Agreement (%)": f"{agree_pct:.1f}%",
                "Mean Intent": f"{mean_intent:.2f}/5",
                "Mean Entity": f"{mean_entity:.2f}/5",
            })

    validity_df = pd.DataFrame(validity_rows)
    print(tabulate(validity_df, headers="keys", tablefmt="github", showindex=False))

    # -------------------------------------------------------------------------
    # Diagnostic 3: Separate AS and BN Retrieval Performance Tables (BM25 + Dense)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print(" [Diagnostic 3] Separate Retrieval Performance for Assamese (AS) vs Bengali (BN)")
    print("-" * 80)

    # We evaluate BM25 across AS and BN
    bm25 = BM25Retriever()
    bm25.index(corpus)

    all_conditions = ["query_reference", "query_mt", "query_llm_translation", "query_postedited", "query_paraphrase", "query_synthetic"]

    for lang_code, lang_name, recs in [("as", "ASSAMESE", as_records), ("bn", "BENGALI", bn_records)]:
        lang_perf_rows = []
        for cond in all_conditions:
            ndcgs, mrrs, recalls = [], [], []
            for r in recs:
                qid = r["query_id"]
                q_text = r.get(cond, "")
                retrieved = bm25.retrieve(q_text, top_k=10)
                ret_ids = [d for d, _ in retrieved]
                qrels_for_q = qrels.get(qid, {r["positive_passage_id"]: 1})

                ndcgs.append(compute_ndcg_at_k(ret_ids, qrels_for_q, k=10))
                mrrs.append(compute_mrr_at_k(ret_ids, qrels_for_q, k=10))
                recalls.append(compute_recall_at_k(ret_ids, qrels_for_q, k=10))

            lang_perf_rows.append({
                "Condition": cond,
                "nDCG@10": np.mean(ndcgs),
                "MRR@10": np.mean(mrrs),
                "Recall@10": np.mean(recalls),
            })
        print(f"\n--- {lang_name} BM25 Retrieval Performance ---")
        df_lang = pd.DataFrame(lang_perf_rows)
        print(tabulate(df_lang, headers="keys", tablefmt="github", floatfmt=".4f", showindex=False))

    # -------------------------------------------------------------------------
    # Diagnostic 4: Bootstrap Significance-Aware Pairwise Comparison Matrix (vs C0)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print(" [Diagnostic 4] Bootstrap Significance-Aware Pairwise Comparison (vs C0 Reference)")
    print("-" * 80)

    ref_ndcgs = []
    condition_ndcgs = {c: [] for c in all_conditions if c != "query_reference"}

    for r in records:
        qid = r["query_id"]
        qrels_for_q = qrels.get(qid, {r["positive_passage_id"]: 1})

        # C0 reference
        ret_c0 = [d for d, _ in bm25.retrieve(r.get("query_reference", ""), top_k=10)]
        ref_ndcgs.append(compute_ndcg_at_k(ret_c0, qrels_for_q, k=10))

        for c in condition_ndcgs.keys():
            ret_c = [d for d, _ in bm25.retrieve(r.get(c, ""), top_k=10)]
            condition_ndcgs[c].append(compute_ndcg_at_k(ret_c, qrels_for_q, k=10))

    sig_rows = []
    for c, scores in condition_ndcgs.items():
        diff, p_val, (ci_low, ci_high), std_diff = bootstrap_paired_test(ref_ndcgs, scores, n_bootstraps=1000)
        sig_rows.append({
            "Transformed Condition": c,
            "C0 Mean": float(np.mean(ref_ndcgs)),
            "Cond Mean": float(np.mean(scores)),
            "Mean Shift (Δ)": diff,
            "95% Bootstrap CI": f"[{ci_low:.4f}, {ci_high:.4f}]",
            "p-value (paired-t)": f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.4f}",
            "Statistically Significant": "YES (p < 0.01)" if p_val < 0.01 else "NO",
        })

    sig_df = pd.DataFrame(sig_rows)
    print(tabulate(sig_df, headers="keys", tablefmt="github", floatfmt=".4f", showindex=False))

    # -------------------------------------------------------------------------
    # Diagnostic 5: 20 Random Cases with Large Rank Drops (C1/C2/C3) & Judge Annotations
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" 🔍 20 RANDOM DIAGNOSTIC CASES (C1/C2/C3 Rank Drops vs C0 with Judge Analysis)")
    print("=" * 80)

    case_candidates = []
    for r in records:
        qid = r["query_id"]
        pos_doc_id = r["positive_passage_id"]

        ret_c0 = [d for d, _ in bm25.retrieve(r.get("query_reference", ""), top_k=50)]
        rank_c0 = (ret_c0.index(pos_doc_id) + 1) if pos_doc_id in ret_c0 else 999

        for cond_key in ["query_mt", "query_llm_translation", "query_postedited"]:
            ret_c = [d for d, _ in bm25.retrieve(r.get(cond_key, ""), top_k=50)]
            rank_c = (ret_c.index(pos_doc_id) + 1) if pos_doc_id in ret_c else 999

            judge_info = r.get("judgements", {}).get(f"judge_{cond_key}", {})
            judge_valid = judge_info.get("overall_validity", "UNKNOWN")

            # Condition: C0 retrieves doc well (rank <= 5), but transformed query misses or drops (rank >= 15)
            if rank_c0 <= 5 and rank_c >= 15:
                case_candidates.append({
                    "record": r,
                    "condition": cond_key,
                    "rank_c0": rank_c0,
                    "rank_c": rank_c,
                    "judge_info": judge_info,
                })

    rng = random.Random(20260921)
    selected_cases = rng.sample(case_candidates, min(20, len(case_candidates)))

    for idx, c in enumerate(selected_cases, 1):
        rec = c["record"]
        cond = c["condition"]
        ji = c["judge_info"]

        ja = ji.get("judge_a", {})
        jb = ji.get("judge_b", {})

        print(f"\n==================== Case {idx:02d} [ID: {rec.get('query_id')} | Lang: {rec.get('language')}] ====================")
        print(f" [Original English q_en] : {rec.get('q_en_source')}")
        print(f" [Reference Query q_ref]  : {rec.get('query_reference')}")
        print(f" [Transformed {cond}]   : {rec.get(cond)}")
        print(f" [Positive Passage (Snippet)]: {rec.get('positive_passage')[:180]}...")
        print(f" [Rank Comparison]        : C0 Ref Rank = {c['rank_c0']}  --->  {cond} Rank = {c['rank_c']}")
        print(f" [Judge A (Gemini 3.8)]   : Validity = {ja.get('overall_validity', 'N/A')} | Intent = {ja.get('intent_preservation_score', 'N/A')}/5 | Reason: {ja.get('reason', '')[:120]}...")
        print(f" [Judge B (Qwen 3.8)]     : Validity = {jb.get('overall_validity', 'N/A')} | Intent = {jb.get('intent_preservation_score', 'N/A')}/5 | Reason: {jb.get('reason', '')[:120]}...")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    run_all_diagnostics()
