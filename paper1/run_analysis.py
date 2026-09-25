#!/usr/bin/env python3
"""
Publication LaTeX Table Generator & Statistical Reporting
Paper 1: Evaluation Invariance under Multilingual Query Transformation

Generates Comprehensive Publication Tables with Language Breakdowns:
1. Table 1a, 1b, 1c: Retrieval Performance (nDCG@10) - Overall, Indonesian (id), Thai (th)
2. Table 2: Evaluation Invariance & Global Model Ordering Stability (Kendall's tau-b, Spearman's rho, rank flips)
3. Table 3: Significance-Aware System-Pair Decision Instability (10,000 Paired Bootstrap CIs) - Overall, ID, TH
4. Table 4: Retriever-Family Interaction Contrasts D = Delta_Lexical - Delta_Dense with 95% CIs
5. Table 5: Behavioral Ranking Shifts (Top-10 Jaccard Overlap, RBO p=0.9, Target Document Delta RR)
6. Table 6: Surface-Form / Round-Trip Textual Fidelity (chrF) vs Delta nDCG Correlations - Overall, ID, TH
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tabulate import tabulate

PAPER1_DIR = Path(__file__).resolve().parent
if str(PAPER1_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER1_DIR))

from config import RESULTS_DIR, OUTPUTS_DIR


def generate_latex_tables(perf_csv: Path, invar_csv: Path, output_dir: Path):
    if not perf_csv.exists() or not invar_csv.exists():
        print(f"[!] Results CSV files not found in {RESULTS_DIR}")
        return

    base_dir = perf_csv.parent
    prefix = perf_csv.name.replace("_performance.csv", "")

    # Load all pooled and per-language CSVs
    perf_df = pd.read_csv(perf_csv)
    invar_df = pd.read_csv(invar_csv)

    perf_id_csv = base_dir / f"{prefix}_performance_id.csv"
    perf_th_csv = base_dir / f"{prefix}_performance_th.csv"
    perf_id_df = pd.read_csv(perf_id_csv) if perf_id_csv.exists() else pd.DataFrame()
    perf_th_df = pd.read_csv(perf_th_csv) if perf_th_csv.exists() else pd.DataFrame()

    invar_id_csv = base_dir / f"{prefix}_invariance_id.csv"
    invar_th_csv = base_dir / f"{prefix}_invariance_th.csv"
    invar_id_df = pd.read_csv(invar_id_csv) if invar_id_csv.exists() else pd.DataFrame()
    invar_th_df = pd.read_csv(invar_th_csv) if invar_th_csv.exists() else pd.DataFrame()

    sig_csv = base_dir / f"{prefix}_system_pair_significance.csv"
    sig_id_csv = base_dir / f"{prefix}_system_pair_significance_id.csv"
    sig_th_csv = base_dir / f"{prefix}_system_pair_significance_th.csv"
    sig_df = pd.read_csv(sig_csv) if sig_csv.exists() else pd.DataFrame()
    sig_id_df = pd.read_csv(sig_id_csv) if sig_id_csv.exists() else pd.DataFrame()
    sig_th_df = pd.read_csv(sig_th_csv) if sig_th_csv.exists() else pd.DataFrame()

    interaction_csv = base_dir / f"{prefix}_family_interactions.csv"
    interaction_df = pd.read_csv(interaction_csv) if interaction_csv.exists() else pd.DataFrame()

    behavior_csv = base_dir / f"{prefix}_behavioral_shifts.csv"
    behavior_df = pd.read_csv(behavior_csv) if behavior_csv.exists() else pd.DataFrame()

    corr_csv = base_dir / f"{prefix}_translation_correlations.csv"
    corr_id_csv = base_dir / f"{prefix}_translation_correlations_id.csv"
    corr_th_csv = base_dir / f"{prefix}_translation_correlations_th.csv"
    corr_df = pd.read_csv(corr_csv) if corr_csv.exists() else pd.DataFrame()
    corr_id_df = pd.read_csv(corr_id_csv) if corr_id_csv.exists() else pd.DataFrame()
    corr_th_df = pd.read_csv(corr_th_csv) if corr_th_csv.exists() else pd.DataFrame()

    print("=" * 80)
    print("           PUBLICATION-READY LATEX TABLES & MULTILINGUAL REPORT")
    print("=" * 80)

    latex_sections = []

    # 1. Performance Table (LaTeX) - Overall
    perf_pivot = perf_df.pivot(index="System", columns="Condition", values="nDCG@10")
    latex_perf = perf_pivot.to_latex(
        float_format="%.4f",
        caption="Overall Retrieval performance (nDCG@10) across multilingual query transformation conditions.",
        label="tab:retrieval_performance_overall",
    )
    latex_sections.append(("% --- Table 1a: Overall Retrieval Performance ---", latex_perf))

    # 1b. Language-Wise Performance Tables
    if not perf_id_df.empty:
        perf_id_pivot = perf_id_df.pivot(index="System", columns="Condition", values="nDCG@10")
        latex_perf_id = perf_id_pivot.to_latex(
            float_format="%.4f",
            caption="Indonesian (id) Retrieval performance (nDCG@10) across transformation conditions.",
            label="tab:retrieval_performance_id",
        )
        latex_sections.append(("% --- Table 1b: Indonesian Retrieval Performance ---", latex_perf_id))

    if not perf_th_df.empty:
        perf_th_pivot = perf_th_df.pivot(index="System", columns="Condition", values="nDCG@10")
        latex_perf_th = perf_th_pivot.to_latex(
            float_format="%.4f",
            caption="Thai (th) Retrieval performance (nDCG@10) across transformation conditions.",
            label="tab:retrieval_performance_th",
        )
        latex_sections.append(("% --- Table 1c: Thai Retrieval Performance ---", latex_perf_th))

    # 2. Invariance Summary Table (LaTeX)
    latex_invar = invar_df.to_latex(
        index=False,
        float_format="%.4f",
        caption="Evaluation invariance and ranking stability metrics relative to Native Reference (C0).",
        label="tab:evaluation_invariance_overall",
    )
    latex_sections.append(("% --- Table 2: Evaluation Invariance Metrics (Overall) ---", latex_invar))

    # 3. Significance-Aware System Pair Decision Table (LaTeX)
    def summarize_sig_df(df, lang_name="Overall"):
        if df.empty:
            return pd.DataFrame()
        cond_changes = df.groupby("Condition").agg(
            Total_Pairs=("Decision_Changed", "count"),
            Changed_Decisions=("Decision_Changed", "sum"),
        )
        cond_changes["Instability_Rate (%)"] = (cond_changes["Changed_Decisions"] / cond_changes["Total_Pairs"]) * 100.0
        return cond_changes

    if not sig_df.empty:
        cond_changes_all = summarize_sig_df(sig_df, "Overall")
        latex_sig = cond_changes_all.to_latex(
            float_format="%.2f",
            caption="Significance-aware system-pair decision instability (10,000 Paired Bootstrap CIs, Overall).",
            label="tab:pairwise_significance_instability_overall",
        )
        latex_sections.append(("% --- Table 3a: Significance-Aware System-Pair Instability (Overall) ---", latex_sig))

    if not sig_id_df.empty:
        cond_changes_id = summarize_sig_df(sig_id_df, "Indonesian")
        latex_sig_id = cond_changes_id.to_latex(
            float_format="%.2f",
            caption="Significance-aware system-pair decision instability (10,000 Paired Bootstrap CIs, Indonesian [id]).",
            label="tab:pairwise_significance_instability_id",
        )
        latex_sections.append(("% --- Table 3b: Significance-Aware System-Pair Instability (Indonesian) ---", latex_sig_id))

    if not sig_th_df.empty:
        cond_changes_th = summarize_sig_df(sig_th_df, "Thai")
        latex_sig_th = cond_changes_th.to_latex(
            float_format="%.2f",
            caption="Significance-aware system-pair decision instability (10,000 Paired Bootstrap CIs, Thai [th]).",
            label="tab:pairwise_significance_instability_th",
        )
        latex_sections.append(("% --- Table 3c: Significance-Aware System-Pair Instability (Thai) ---", latex_sig_th))

    # 4. Retriever-Family Interaction Contrasts Table (LaTeX)
    if not interaction_df.empty:
        latex_interaction = interaction_df.to_latex(
            index=False,
            float_format="%.4f",
            caption="Retriever-family interaction contrasts D = Delta_family1 - Delta_family2 with 10,000-bootstrap 95% CIs. Negative contrast (D < 0) denotes greater performance degradation for Lexical/Hybrid systems than Dense systems.",
            label="tab:retriever_family_interactions",
        )
        latex_sections.append(("% --- Table 4: Retriever-Family Interaction Contrasts ---", latex_interaction))

    # 5. Behavioral Shifts Table (LaTeX)
    if not behavior_df.empty:
        latex_behavior = behavior_df.to_latex(
            index=False,
            float_format="%.4f",
            caption="Behavioral ranking shifts: Top-10 Jaccard Overlap, Rank-Biased Overlap (RBO, p=0.9), and Target Document Reciprocal Rank Shift (Delta RR) relative to C0.",
            label="tab:behavioral_shifts",
        )
        latex_sections.append(("% --- Table 5: Behavioral Ranking Shifts ---", latex_behavior))

    # 6. Surface-Form Textual Fidelity Correlation Table (LaTeX)
    if not corr_df.empty:
        latex_corr = corr_df.to_latex(
            index=False,
            float_format="%.4f",
            caption="Pearson and Spearman correlations between query surface-form / round-trip textual fidelity (chrF) and retrieval score shift (Delta nDCG@10, Overall).",
            label="tab:translation_correlations_overall",
        )
        latex_sections.append(("% --- Table 6a: Surface-Form Fidelity Correlation (Overall) ---", latex_corr))

    if not corr_id_df.empty:
        latex_corr_id = corr_id_df.to_latex(
            index=False,
            float_format="%.4f",
            caption="Pearson and Spearman correlations between query surface-form / round-trip textual fidelity (chrF) and retrieval score shift (Delta nDCG@10, Indonesian [id]).",
            label="tab:translation_correlations_id",
        )
        latex_sections.append(("% --- Table 6b: Surface-Form Fidelity Correlation (Indonesian) ---", latex_corr_id))

    if not corr_th_df.empty:
        latex_corr_th = corr_th_df.to_latex(
            index=False,
            float_format="%.4f",
            caption="Pearson and Spearman correlations between query surface-form / round-trip textual fidelity (chrF) and retrieval score shift (Delta nDCG@10, Thai [th]).",
            label="tab:translation_correlations_th",
        )
        latex_sections.append(("% --- Table 6c: Surface-Form Fidelity Correlation (Thai) ---", latex_corr_th))

    # Write output to LaTeX file
    output_dir.mkdir(parents=True, exist_ok=True)
    latex_file = output_dir / "paper1_tables.tex"
    with open(latex_file, "w", encoding="utf-8") as f:
        for header, content in latex_sections:
            f.write(f"{header}\n{content}\n\n")

    print(f"[✓] Complete multilingual LaTeX tables written to: {latex_file}")
    print("\n--- Summary Performance Table (nDCG@10) ---")
    print(tabulate(perf_pivot, headers="keys", tablefmt="github", floatfmt=".4f"))
    if not perf_id_df.empty:
        print("\n--- Indonesian Performance Table (nDCG@10) ---")
        print(tabulate(perf_id_pivot, headers="keys", tablefmt="github", floatfmt=".4f"))
    if not perf_th_df.empty:
        print("\n--- Thai Performance Table (nDCG@10) ---")
        print(tabulate(perf_th_pivot, headers="keys", tablefmt="github", floatfmt=".4f"))


def main():
    perf_csv = RESULTS_DIR / "all_set_performance.csv"
    invar_csv = RESULTS_DIR / "all_set_invariance.csv"
    generate_latex_tables(perf_csv, invar_csv, OUTPUTS_DIR)


if __name__ == "__main__":
    main()
