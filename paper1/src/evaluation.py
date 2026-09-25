"""
Invariance Evaluation, Ranking Stability & Statistical Analysis Suite
Paper 1: Evaluation Invariance under Multilingual Query Transformation

Computes:
1. Information Retrieval Metrics:
   - nDCG@10, MRR@10, Recall@10, Recall@100
2. Invariance & Stability Metrics:
   - Kendall's Tau-b (τ_b) and Spearman's Rho (ρ)
   - Pairwise system ranking inversion rate
   - Mean absolute score shift (Δ)
   - Top-10 retrieved document overlap (Jaccard index)
   - Rank-Biased Overlap (RBO, p=0.9)
   - Positive Document Reciprocal Rank Shift (ΔRR)
3. Significance-Aware System-Pair Hypothesis Stability:
   - Paired bootstrap CI (10,000 resamples) for every system pair (A, B)
   - Classifies decisions: A > B, A ≈ B, B > A
   - Quantifies Conclusion Instability Rate (percentage of statistically supported relations that flip)
4. Retriever-Family Interaction Contrasts:
   - D_{m1, m2, T} = (Δ_{m1, T} - Δ_{m2, T}) with 10,000 bootstrap CI
   - Lexical vs Dense contrast and Hybrid vs Dense contrast
5. Translation-Quality to Retrieval-Change Correlation:
   - chrF character n-gram similarity & token Jaccard vs Δ nDCG@10
6. Language-Wise Disaggregated Analysis (Thai vs Indonesian vs Overall)
"""

import math
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional, Set


def compute_ndcg_at_k(retrieved_doc_ids: List[str], qrels_dict: Dict[str, int], k: int = 10) -> float:
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_doc_ids[:k]):
        rel = qrels_dict.get(doc_id, 0)
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(i + 2)

    ideal_rels = sorted(list(qrels_dict.values()), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_rels):
        if rel > 0:
            idcg += (2**rel - 1) / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def compute_mrr_at_k(retrieved_doc_ids: List[str], qrels_dict: Dict[str, int], k: int = 10) -> float:
    for i, doc_id in enumerate(retrieved_doc_ids[:k]):
        if qrels_dict.get(doc_id, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def compute_recall_at_k(retrieved_doc_ids: List[str], qrels_dict: Dict[str, int], k: int = 10) -> float:
    relevant_total = sum(1 for rel in qrels_dict.values() if rel > 0)
    if relevant_total == 0:
        return 0.0
    hits = sum(1 for doc_id in retrieved_doc_ids[:k] if qrels_dict.get(doc_id, 0) > 0)
    return hits / relevant_total


def compute_jaccard_overlap(list_a: List[str], list_b: List[str], k: int = 10) -> float:
    set_a = set(list_a[:k])
    set_b = set(list_b[:k])
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b) if (set_a | set_b) else 1.0


def compute_rbo(list_a: List[str], list_b: List[str], p: float = 0.9, k: int = 10) -> float:
    """Computes Rank-Biased Overlap (RBO) between two ranked lists up to depth k."""
    if not list_a and not list_b:
        return 1.0
    list_a = list_a[:k]
    list_b = list_b[:k]
    max_depth = max(len(list_a), len(list_b))
    if max_depth == 0:
        return 1.0

    seen_a = set()
    seen_b = set()
    overlap = 0
    rbo_sum = 0.0

    for d in range(1, max_depth + 1):
        if d <= len(list_a):
            seen_a.add(list_a[d - 1])
        if d <= len(list_b):
            seen_b.add(list_b[d - 1])
        overlap = len(seen_a & seen_b)
        agreement = overlap / d
        rbo_sum += (p ** (d - 1)) * agreement

    return (1.0 - p) * rbo_sum


def compute_chrf_similarity(s1: str, s2: str, max_n: int = 6, beta: float = 2.0) -> float:
    """Computes character n-gram F-score (chrF-like) between two strings."""
    s1_clean = "".join(s1.split())
    s2_clean = "".join(s2.split())
    if not s1_clean or not s2_clean:
        return 0.0
    if s1_clean == s2_clean:
        return 1.0

    f_scores = []
    for n in range(1, max_n + 1):
        ngrams1 = Counter([s1_clean[i:i+n] for i in range(len(s1_clean) - n + 1)])
        ngrams2 = Counter([s2_clean[i:i+n] for i in range(len(s2_clean) - n + 1)])
        overlap = sum((ngrams1 & ngrams2).values())
        total1 = sum(ngrams1.values())
        total2 = sum(ngrams2.values())

        prec = overlap / total2 if total2 > 0 else 0.0
        rec = overlap / total1 if total1 > 0 else 0.0
        if prec + rec > 0:
            f = (1 + beta**2) * prec * rec / ((beta**2 * prec) + rec)
        else:
            f = 0.0
        f_scores.append(f)

    return float(np.mean(f_scores))


class InvarianceEvaluator:
    def __init__(self, qrels: Dict[str, Dict[str, int]], seed: int = 42):
        self.qrels = qrels
        self.seed = seed

    def evaluate_run_per_query(
        self,
        retrieval_results: Dict[str, List[Tuple[str, float]]],
        k: int = 10,
    ) -> Dict[str, Dict[str, Any]]:
        """Computes IR metrics per query for statistical tests."""
        per_query = {}
        for qid, docs_scores in retrieval_results.items():
            if qid not in self.qrels:
                continue
            retrieved_ids = [doc_id for doc_id, _ in docs_scores]
            qrels_for_q = self.qrels[qid]

            per_query[qid] = {
                f"nDCG@{k}": compute_ndcg_at_k(retrieved_ids, qrels_for_q, k=k),
                f"MRR@{k}": compute_mrr_at_k(retrieved_ids, qrels_for_q, k=k),
                f"Recall@{k}": compute_recall_at_k(retrieved_ids, qrels_for_q, k=k),
                "Recall@100": compute_recall_at_k(retrieved_ids, qrels_for_q, k=100),
                "retrieved_ids": retrieved_ids,
            }
        return per_query

    def evaluate_full_suite(
        self,
        full_results: Dict[str, Dict[str, Dict[str, List[Tuple[str, float]]]]],
        k: int = 10,
        filter_qids: Optional[Set[str]] = None,
    ) -> pd.DataFrame:
        rows = []
        for sys_name, conditions in full_results.items():
            for cond, runs in conditions.items():
                per_q = self.evaluate_run_per_query(runs, k=k)
                if filter_qids is not None:
                    per_q = {qid: v for qid, v in per_q.items() if qid in filter_qids}
                ndcgs = [v[f"nDCG@{k}"] for v in per_q.values()]
                mrrs = [v[f"MRR@{k}"] for v in per_q.values()]
                recalls_10 = [v[f"Recall@{k}"] for v in per_q.values()]
                recalls_100 = [v["Recall@100"] for v in per_q.values()]

                rows.append({
                    "System": sys_name,
                    "Condition": cond,
                    f"nDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
                    f"MRR@{k}": float(np.mean(mrrs)) if mrrs else 0.0,
                    f"Recall@{k}": float(np.mean(recalls_10)) if recalls_10 else 0.0,
                    "Recall@100": float(np.mean(recalls_100)) if recalls_100 else 0.0,
                })
        return pd.DataFrame(rows)

    def compute_invariance_matrix(
        self,
        perf_df: pd.DataFrame,
        metric: str = "nDCG@10",
        ref_condition: str = "query_reference",
    ) -> pd.DataFrame:
        pivot = perf_df.pivot(index="System", columns="Condition", values=metric)
        if ref_condition not in pivot.columns:
            return pd.DataFrame()

        ref_series = pivot[ref_condition]
        ref_rank = ref_series.rank(ascending=False)

        invariance_rows = []
        for cond in pivot.columns:
            cond_series = pivot[cond]
            cond_rank = cond_series.rank(ascending=False)

            tau, _ = stats.kendalltau(ref_rank, cond_rank)
            rho, _ = stats.spearmanr(ref_rank, cond_rank)
            mean_shift = float(cond_series.mean() - ref_series.mean())

            # Count pairwise rank inversions
            systems = list(pivot.index)
            flips = 0
            total_pairs = 0
            for i in range(len(systems)):
                for j in range(i + 1, len(systems)):
                    s1, s2 = systems[i], systems[j]
                    ref_diff = ref_series[s1] - ref_series[s2]
                    cond_diff = cond_series[s1] - cond_series[s2]
                    if (ref_diff * cond_diff) < 0:
                        flips += 1
                    total_pairs += 1

            flip_rate = (flips / total_pairs * 100.0) if total_pairs > 0 else 0.0

            invariance_rows.append({
                "Condition": cond,
                f"Mean {metric}": float(cond_series.mean()),
                "Mean Shift (Δ)": mean_shift,
                "Kendall Tau (τ_b)": float(tau),
                "Spearman Rho (ρ)": float(rho),
                "Pairwise Flips": flips,
                "Flip Rate (%)": flip_rate,
            })

        return pd.DataFrame(invariance_rows)

    def compute_system_pair_significance(
        self,
        full_results: Dict[str, Dict[str, Dict[str, List[Tuple[str, float]]]]],
        metric: str = "nDCG@10",
        ref_condition: str = "query_reference",
        n_bootstraps: int = 10000,
        filter_qids: Optional[Set[str]] = None,
    ) -> pd.DataFrame:
        """Computes 10,000 paired bootstrap confidence intervals for every system pair."""
        systems = sorted(list(full_results.keys()))
        conditions = sorted(list(full_results[systems[0]].keys()))

        query_scores = {}
        for s in systems:
            query_scores[s] = {}
            for c in conditions:
                per_q = self.evaluate_run_per_query(full_results[s][c])
                if filter_qids is not None:
                    per_q = {qid: v for qid, v in per_q.items() if qid in filter_qids}
                query_scores[s][c] = {qid: vals[metric] for qid, vals in per_q.items()}

        common_qids = sorted(list(set.intersection(*[
            set(query_scores[s][c].keys()) for s in systems for c in conditions
        ])))

        pair_rows = []
        rng = np.random.RandomState(self.seed)

        for sys_a, sys_b in itertools.combinations(systems, 2):
            scores_a_ref = np.array([query_scores[sys_a][ref_condition][q] for q in common_qids])
            scores_b_ref = np.array([query_scores[sys_b][ref_condition][q] for q in common_qids])
            diffs_ref = scores_a_ref - scores_b_ref

            boot_ref = [np.mean(rng.choice(diffs_ref, size=len(diffs_ref), replace=True)) for _ in range(n_bootstraps)]
            ci_low_ref, ci_high_ref = np.percentile(boot_ref, 2.5), np.percentile(boot_ref, 97.5)

            if ci_low_ref > 0:
                dec_ref = f"{sys_a} > {sys_b}"
            elif ci_high_ref < 0:
                dec_ref = f"{sys_b} > {sys_a}"
            else:
                dec_ref = f"{sys_a} ≈ {sys_b}"

            for cond in conditions:
                if cond == ref_condition:
                    continue
                scores_a_cond = np.array([query_scores[sys_a][cond][q] for q in common_qids])
                scores_b_cond = np.array([query_scores[sys_b][cond][q] for q in common_qids])
                diffs_cond = scores_a_cond - scores_b_cond

                boot_cond = [np.mean(rng.choice(diffs_cond, size=len(diffs_cond), replace=True)) for _ in range(n_bootstraps)]
                ci_low_c, ci_high_c = np.percentile(boot_cond, 2.5), np.percentile(boot_cond, 97.5)

                if ci_low_c > 0:
                    dec_cond = f"{sys_a} > {sys_b}"
                elif ci_high_c < 0:
                    dec_cond = f"{sys_b} > {sys_a}"
                else:
                    dec_cond = f"{sys_a} ≈ {sys_b}"

                pair_rows.append({
                    "System_A": sys_a,
                    "System_B": sys_b,
                    "Condition": cond,
                    "Decision_C0": dec_ref,
                    "CI_C0": f"[{ci_low_ref:.4f}, {ci_high_ref:.4f}]",
                    "Decision_Cond": dec_cond,
                    "CI_Cond": f"[{ci_low_c:.4f}, {ci_high_c:.4f}]",
                    "Decision_Changed": (dec_ref != dec_cond),
                })

        return pd.DataFrame(pair_rows)

    def compute_behavioral_overlap_and_rank_shift(
        self,
        full_results: Dict[str, Dict[str, Dict[str, List[Tuple[str, float]]]]],
        ref_condition: str = "query_reference",
        k: int = 10,
        rbo_p: float = 0.9,
    ) -> pd.DataFrame:
        """
        Computes Top-10 Jaccard Overlap, Rank-Biased Overlap (RBO), and
        Target Positive Document Reciprocal Rank Shift (ΔRR) per system & condition.
        """
        systems = sorted(list(full_results.keys()))
        conditions = [c for c in full_results[systems[0]].keys() if c != ref_condition]

        rows = []
        for sys_name in systems:
            ref_runs = full_results[sys_name][ref_condition]
            for cond in conditions:
                cond_runs = full_results[sys_name][cond]
                common_qids = sorted(list(set(ref_runs.keys()) & set(cond_runs.keys()) & set(self.qrels.keys())))

                jaccards = []
                rbos = []
                rr_shifts = []

                for qid in common_qids:
                    ref_retrieved = [doc_id for doc_id, _ in ref_runs[qid][:k]]
                    cond_retrieved = [doc_id for doc_id, _ in cond_runs[qid][:k]]

                    # 1. Jaccard @ k
                    jaccards.append(compute_jaccard_overlap(ref_retrieved, cond_retrieved, k=k))

                    # 2. RBO @ k
                    rbos.append(compute_rbo(ref_retrieved, cond_retrieved, p=rbo_p, k=k))

                    # 3. Positive doc reciprocal rank shift
                    qrels_for_q = self.qrels[qid]
                    mrr_ref = compute_mrr_at_k(ref_retrieved, qrels_for_q, k=k)
                    mrr_cond = compute_mrr_at_k(cond_retrieved, qrels_for_q, k=k)
                    rr_shifts.append(mrr_cond - mrr_ref)

                rows.append({
                    "System": sys_name,
                    "Condition": cond,
                    f"Jaccard@{k}": float(np.mean(jaccards)),
                    f"RBO@{k} (p={rbo_p})": float(np.mean(rbos)),
                    "Mean ΔRR": float(np.mean(rr_shifts)),
                })

        return pd.DataFrame(rows)

    def compute_retriever_family_interaction(
        self,
        full_results: Dict[str, Dict[str, Dict[str, List[Tuple[str, float]]]]],
        metric: str = "nDCG@10",
        ref_condition: str = "query_reference",
        family_map: Optional[Dict[str, str]] = None,
        n_bootstraps: int = 10000,
        filter_qids: Optional[Set[str]] = None,
    ) -> pd.DataFrame:
        """
        Computes formal 2x2 Interaction Contrasts between Retrieval Families:
        D = Δ_family1 - Δ_family2 where Δ_family = Mean_{sys in family} [ Score(T) - Score(C0) ].
        Negative D (D < 0) indicates family 1 (e.g. Lexical) suffers GREATER performance degradation than family 2 (Dense).
        """
        if family_map is None:
            family_map = {
                "BM25": "Lexical",
                "Qwen3-0.6B": "Dense",
                "BGE-M3": "Dense",
                "mE5-base": "Dense",
                "mE5-large": "Dense",
                "LaBSE": "Dense",
                "Hybrid-RRF": "Hybrid",
            }

        systems = sorted(list(full_results.keys()))
        conditions = [c for c in full_results[systems[0]].keys() if c != ref_condition]

        query_scores = {}
        for s in systems:
            query_scores[s] = {}
            for c in [ref_condition] + conditions:
                per_q = self.evaluate_run_per_query(full_results[s][c])
                if filter_qids is not None:
                    per_q = {qid: vals for qid, vals in per_q.items() if qid in filter_qids}
                query_scores[s][c] = {qid: vals[metric] for qid, vals in per_q.items()}

        common_qids = sorted(list(set.intersection(*[
            set(query_scores[s][c].keys()) for s in systems for c in [ref_condition] + conditions
        ])))

        if not common_qids:
            return pd.DataFrame()

        # Precompute per-query delta for each system: Δ_{i, m, T} = nDCG(T) - nDCG(C0)
        sys_deltas = {s: {} for s in systems}
        for s in systems:
            for cond in conditions:
                sys_deltas[s][cond] = np.array([
                    query_scores[s][cond][q] - query_scores[s][ref_condition][q] for q in common_qids
                ])

        # Group by family
        families = ["Lexical", "Dense", "Hybrid"]
        family_members = {f: [s for s in systems if family_map.get(s) == f] for f in families}

        family_deltas = {f: {} for f in families if family_members[f]}
        for f, mems in family_members.items():
            if not mems:
                continue
            for cond in conditions:
                family_deltas[f][cond] = np.mean([sys_deltas[s][cond] for s in mems], axis=0)

        rng = np.random.RandomState(self.seed)
        contrast_rows = []

        # Family contrasts: Lexical vs Dense, Hybrid vs Dense
        family_pairs = [("Lexical", "Dense"), ("Hybrid", "Dense")]
        for f1, f2 in family_pairs:
            if f1 not in family_deltas or f2 not in family_deltas:
                continue
            for cond in conditions:
                diffs = family_deltas[f1][cond] - family_deltas[f2][cond]
                mean_diff = float(np.mean(diffs))

                boot = [np.mean(rng.choice(diffs, size=len(diffs), replace=True)) for _ in range(n_bootstraps)]
                ci_low, ci_high = np.percentile(boot, 2.5), np.percentile(boot, 97.5)

                contrast_rows.append({
                    "Comparison": f"{f1} vs {f2}",
                    "Condition": cond,
                    "Mean Contrast D": mean_diff,
                    "95% CI": f"[{ci_low:.4f}, {ci_high:.4f}]",
                    "Significant (excludes 0)": (ci_low > 0 or ci_high < 0),
                })

        return pd.DataFrame(contrast_rows)

    def compute_translation_quality_correlation(
        self,
        records: List[Dict[str, Any]],
        full_results: Dict[str, Dict[str, Dict[str, List[Tuple[str, float]]]]],
        metric: str = "nDCG@10",
        ref_condition: str = "query_reference",
        filter_qids: Optional[Set[str]] = None,
    ) -> pd.DataFrame:
        """
        Computes correlation between surface-form / round-trip textual fidelity (chrF character F1)
        and retrieval score shift (Δ nDCG@10) across transformed queries.
        """
        systems = sorted(list(full_results.keys()))
        conditions = [c for c in full_results[systems[0]].keys() if c != ref_condition]

        # Record queries by qid
        record_map = {str(r["query_id"]): r for r in records}
        if filter_qids is not None:
            record_map = {qid: r for qid, r in record_map.items() if qid in filter_qids}

        # Query scores
        query_scores = {}
        for s in systems:
            query_scores[s] = {}
            for c in [ref_condition] + conditions:
                per_q = self.evaluate_run_per_query(full_results[s][c])
                if filter_qids is not None:
                    per_q = {qid: vals for qid, vals in per_q.items() if qid in filter_qids}
                query_scores[s][c] = {qid: vals[metric] for qid, vals in per_q.items()}

        corr_rows = []
        for cond in conditions:
            chrf_scores = []
            mean_deltas = []

            for qid, r in record_map.items():
                if qid not in query_scores[systems[0]][ref_condition]:
                    continue
                q_ref = r.get(ref_condition, "")
                q_cond = r.get(cond, "")
                if not q_ref or not q_cond:
                    continue

                chrf = compute_chrf_similarity(q_ref, q_cond)
                # Average delta across all retrieval systems for query qid
                avg_delta = np.mean([
                    query_scores[s][cond][qid] - query_scores[s][ref_condition][qid]
                    for s in systems
                ])

                chrf_scores.append(chrf)
                mean_deltas.append(avg_delta)

            if len(chrf_scores) > 2:
                p_r, p_val = stats.pearsonr(chrf_scores, mean_deltas)
                s_rho, s_val = stats.spearmanr(chrf_scores, mean_deltas)
                corr_rows.append({
                    "Condition": cond,
                    "Mean chrF Fidelity": float(np.mean(chrf_scores)),
                    "Pearson r (chrF vs Δ)": float(p_r),
                    "p-value (r)": float(p_val),
                    "Spearman ρ (chrF vs Δ)": float(s_rho),
                    "p-value (ρ)": float(s_val),
                })

        return pd.DataFrame(corr_rows)
