#!/usr/bin/env python3
"""
Generate publication-grade, high-readability figures designed specifically
for A4/LNCS pages and senior reviewers (large, crisp typography, 2-tier flow).
Outputs to: /home/kshitij22256/fineline/paper/figures/
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

FIGURES_DIR = Path("/home/kshitij22256/fineline/paper/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# High-contrast, clean sans-serif typography
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 11,
    "figure.dpi": 300,
})


def plot_drawio_pipeline():
    """
    Generates a 2-tier high-readability architectural pipeline diagram.
    Uses large 11-14pt fonts, bold color coding, and spacious cards
    so it renders crystal clear on screen and in print without squinting.
    """
    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 5.8)

    # -------------------------------------------------------------
    # Palette (Accessible, High-Contrast draw.io Theme)
    # -------------------------------------------------------------
    c_blue_hdr = "#0284C7"
    c_blue_bg = "#F0F9FF"
    c_blue_stroke = "#0284C7"

    c_indigo_hdr = "#4F46E5"
    c_indigo_bg = "#EEF2FF"
    c_indigo_stroke = "#4F46E5"

    c_purple_hdr = "#7E22CE"
    c_purple_bg = "#FAF5FF"
    c_purple_stroke = "#7E22CE"

    c_emerald_hdr = "#059669"
    c_emerald_bg = "#ECFDF5"
    c_emerald_stroke = "#059669"

    def draw_card(x, y, w, h, title, fill, stroke, header_bg, step_num):
        # Card Body
        bg = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.18",
            facecolor=fill, edgecolor=stroke, linewidth=1.8, zorder=1
        )
        ax.add_patch(bg)
        
        # Header Box
        header_h = 0.58
        hdr = patches.FancyBboxPatch(
            (x, y + h - header_h), w, header_h, boxstyle="round,pad=0.0,rounding_size=0.18",
            facecolor=header_bg, edgecolor=header_bg, linewidth=1, zorder=2
        )
        ax.add_patch(hdr)
        
        # Header Text (Large 11.5pt bold)
        ax.text(x + w / 2, y + h - header_h / 2, f"STAGE {step_num}: {title.upper()}",
                ha="center", va="center", fontsize=11, fontweight="bold", color="white", zorder=3)

    # =============================================================
    # TIER 1 (Top Half: Input & Transformations) y: 3.1 -> 5.5
    # =============================================================
    
    # 1. Block 1: Canonical Dataset (x: 0.5, y: 3.1, w: 4.4, h: 2.4)
    draw_card(0.5, 3.1, 4.4, 2.4, "Canonical Benchmark", c_blue_bg, c_blue_stroke, c_blue_hdr, "1")
    
    # Inner box 1: Languages
    b_lang = patches.FancyBboxPatch((0.7, 3.95), 4.0, 0.85, boxstyle="round,pad=0.0,rounding_size=0.1",
                                   facecolor="#FFFFFF", edgecolor="#BAE6FD", linewidth=1.2, zorder=2)
    ax.add_patch(b_lang)
    ax.text(0.85, 4.58, "Target Benchmark Languages:", fontsize=10, fontweight="bold", color="#0369A1", zorder=3)
    ax.text(0.85, 4.30, "• Indonesian (id):", fontsize=9.5, fontweight="bold", color="#0F172A", zorder=3)
    ax.text(2.35, 4.30, "Latin Alphabet (Whitespace Spaced)", fontsize=9, color="#475569", zorder=3)
    ax.text(0.85, 4.08, "• Thai (th):", fontsize=9.5, fontweight="bold", color="#0F172A", zorder=3)
    ax.text(1.95, 4.08, "Non-Latin Continuous Unsegmented Script", fontsize=9, color="#475569", zorder=3)

    # Inner box 2: Ground truth
    b_gt = patches.FancyBboxPatch((0.7, 3.25), 4.0, 0.58, boxstyle="round,pad=0.0,rounding_size=0.1",
                                 facecolor="#E0F2FE", edgecolor="#7DD3FC", linewidth=1.0, zorder=2)
    ax.add_patch(b_gt)
    ax.text(2.7, 3.54, "Ground-Truth: Native Query (q_ref) + Passages (d+)", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="#0284C7", zorder=3)

    # 2. Block 2: 6 Query Conditions (x: 5.6, y: 3.1, w: 4.4, h: 2.4)
    draw_card(5.6, 3.1, 4.4, 2.4, "6 Controlled Query Conditions", c_indigo_bg, c_indigo_stroke, c_indigo_hdr, "2")
    
    cond_rows = [
        ("C0", "Native Reference (q_ref)", "Human-authored canonical baseline", "#2563EB"),
        ("C1", "LLM Paraphrase", "Same-language semantic rewrite", "#4F46E5"),
        ("C2", "NLLB-200 Round-Trip", "Neural pivot translation via English", "#7C3AED"),
        ("C3", "OPUS-MT Round-Trip", "Open multilingual neural MT", "#DB2777"),
        ("C4", "LLM-MT Round-Trip", "Zero-shot instruction translation", "#D97706"),
        ("C5", "Synthetic Query (d+)", "Passage-conditioned query generation", "#059669"),
    ]
    cur_y = 4.78
    for tag, name, desc, badge_col in cond_rows:
        # Tag badge
        t_box = patches.FancyBboxPatch((5.78, cur_y - 0.17), 0.44, 0.22, boxstyle="round,pad=0.0,rounding_size=0.06",
                                      facecolor=badge_col, edgecolor="none", zorder=3)
        ax.add_patch(t_box)
        ax.text(6.0, cur_y - 0.06, tag, ha="center", va="center", fontsize=8.5, fontweight="bold", color="white", zorder=4)
        # Name and desc
        ax.text(6.32, cur_y - 0.06, name, ha="left", va="center", fontsize=9, fontweight="bold", color="#1E293B", zorder=3)
        ax.text(8.35, cur_y - 0.06, desc.split()[0] + " " + desc.split()[1], ha="left", va="center", fontsize=8, color="#64748B", zorder=3)
        cur_y -= 0.28

    # Arrow Tier 1: Block 1 -> Block 2
    ax.annotate(
        "", xy=(5.6, 4.3), xytext=(4.9, 4.3),
        arrowprops=dict(arrowstyle="-|>", lw=2.5, color="#1E293B", mutation_scale=16),
        zorder=5
    )
    ax.text(5.25, 4.58, "Generate\n5 Variants", ha="center", va="center", fontsize=8, fontweight="bold", color="#1E293B",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#94A3B8", lw=0.9), zorder=6)

    # =============================================================
    # TRANSITION ARROW (Tier 1 -> Tier 2, Downwards Flow)
    # =============================================================
    ax.annotate(
        "", xy=(7.8, 2.7), xytext=(7.8, 3.1),
        arrowprops=dict(arrowstyle="-|>", lw=2.5, color="#1E293B", mutation_scale=16),
        zorder=5
    )
    ax.text(7.8, 2.9, "Execute All Queries on 7 Retrievers", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#1E293B",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FEF3C7", edgecolor="#F59E0B", lw=1.0), zorder=6)

    # =============================================================
    # TIER 2 (Bottom Half: Retrieval & Evaluation) y: 0.2 -> 2.6
    # =============================================================

    # 3. Block 3: Retrieval Models (x: 0.5, y: 0.2, w: 4.4, h: 2.4)
    draw_card(0.5, 0.2, 4.4, 2.4, "7 Evaluated Retrieval Systems", c_purple_bg, c_purple_stroke, c_purple_hdr, "3")
    
    # Lexical
    b_lex = patches.FancyBboxPatch((0.7, 1.45), 4.0, 0.50, boxstyle="round,pad=0.0,rounding_size=0.08",
                                  facecolor="#FFFFFF", edgecolor="#E9D5FF", linewidth=1.0, zorder=2)
    ax.add_patch(b_lex)
    ax.text(0.85, 1.70, "• Lexical Baseline:", fontsize=9.5, fontweight="bold", color="#7E22CE", zorder=3)
    ax.text(2.35, 1.70, "BM25 (Language Tokenized)", fontsize=9, color="#1E293B", zorder=3)

    # Dense
    b_dense = patches.FancyBboxPatch((0.7, 0.85), 4.0, 0.52, boxstyle="round,pad=0.0,rounding_size=0.08",
                                    facecolor="#FFFFFF", edgecolor="#E9D5FF", linewidth=1.0, zorder=2)
    ax.add_patch(b_dense)
    ax.text(0.85, 1.10, "• Dense Multilingual:", fontsize=9.5, fontweight="bold", color="#7E22CE", zorder=3)
    ax.text(2.55, 1.10, "Qwen3-0.6B, BGE-M3, mE5, LaBSE", fontsize=8.8, color="#1E293B", zorder=3)

    # Hybrid
    b_hyb = patches.FancyBboxPatch((0.7, 0.32), 4.0, 0.45, boxstyle="round,pad=0.0,rounding_size=0.08",
                                  facecolor="#FFFFFF", edgecolor="#E9D5FF", linewidth=1.0, zorder=2)
    ax.add_patch(b_hyb)
    ax.text(0.85, 0.54, "• Hybrid RRF:", fontsize=9.5, fontweight="bold", color="#7E22CE", zorder=3)
    ax.text(1.95, 0.54, "BM25 + Qwen3-0.6B Fusion (k=60)", fontsize=9, color="#1E293B", zorder=3)

    # 4. Block 4: Invariance & Statistical Audit (x: 5.6, y: 0.2, w: 4.4, h: 2.4)
    draw_card(5.6, 0.2, 4.4, 2.4, "Evaluation Invariance Audit", c_emerald_bg, c_emerald_stroke, c_emerald_hdr, "4")

    # Invariance items
    audit_items = [
        ("Macro Ranking Invariance:", "Kendall's tau_b & Spearman rho (tau_b = 1.000 for Synth)", "#065F46"),
        ("10,000 Bootstrap CIs:", "21 System Pairs -> Up to 14.3% Decision Instability", "#065F46"),
        ("Retriever Contrast (D < 0):", "D = Delta_Lexical - Delta_Dense (BM25 Disproportionately Harmed)", "#B91C1C"),
        ("Textual Fidelity (chrF):", "chrF vs. Delta nDCG (Strong correlation for MT, p < 0.001)", "#065F46")
    ]
    cur_y = 1.78
    for a_title, a_desc, col_title in audit_items:
        b_aud = patches.FancyBboxPatch((5.78, cur_y - 0.16), 4.04, 0.36, boxstyle="round,pad=0.0,rounding_size=0.06",
                                      facecolor="#FFFFFF", edgecolor="#A7F3D0", linewidth=0.8, zorder=2)
        ax.add_patch(b_aud)
        ax.text(5.90, cur_y + 0.02, a_title, fontsize=8.8, fontweight="bold", color=col_title, zorder=3)
        ax.text(5.90, cur_y - 0.10, a_desc, fontsize=7.8, color="#047857" if col_title != "#B91C1C" else "#991B1B", zorder=3)
        cur_y -= 0.40

    # Horizontal Arrow Tier 2: Block 3 -> Block 4
    ax.annotate(
        "", xy=(5.6, 1.4), xytext=(4.9, 1.4),
        arrowprops=dict(arrowstyle="-|>", lw=2.5, color="#1E293B", mutation_scale=16),
        zorder=5
    )
    ax.text(5.25, 1.68, "Rankings &\nQrels", ha="center", va="center", fontsize=8, fontweight="bold", color="#1E293B",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#94A3B8", lw=0.9), zorder=6)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_pipeline.png", bbox_inches="tight", dpi=300)
    plt.savefig(FIGURES_DIR / "fig1_pipeline.pdf", bbox_inches="tight")
    plt.close()
    print("[✓] Generated High-Readability 2-Tier Figure 1: Pipeline Diagram")


def plot_family_interaction_contrasts():
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=300)

    conditions = ["Paraphrase", "NLLB MT", "OPUS MT", "LLM MT", "Synthetic"]
    x = np.arange(len(conditions))
    width = 0.26

    delta_bm25 = [-0.0708, -0.2066, -0.2245, -0.1296, +0.0056]
    delta_dense = [-0.0111, -0.0814, -0.1417, -0.0164, -0.0101]
    delta_hybrid = [-0.0643, -0.1675, -0.2200, -0.0963, -0.0299]

    ax.bar(x - width, delta_bm25, width, label="Lexical (BM25)", color="#E11D48", alpha=0.9, edgecolor="black", linewidth=0.5)
    ax.bar(x, delta_hybrid, width, label="Hybrid-RRF", color="#D97706", alpha=0.9, edgecolor="black", linewidth=0.5)
    ax.bar(x + width, delta_dense, width, label="Dense Encoders (Qwen3/mE5/BGE)", color="#2563EB", alpha=0.9, edgecolor="black", linewidth=0.5)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel(r"Effectiveness Shift ($\Delta$ nDCG@10)", fontsize=11, fontweight="bold")
    ax.set_title(r"Retriever-Family Performance Shifts ($\Delta$) Under Query Transformation", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontweight="bold", fontsize=10)
    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.95, fontsize=9.5)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.set_ylim(-0.26, 0.04)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_family_degradation.png", bbox_inches="tight", dpi=300)
    plt.savefig(FIGURES_DIR / "fig2_family_degradation.pdf", bbox_inches="tight")
    plt.close()
    print("[✓] Generated Figure 2: Family Interaction Plot")


def plot_language_script_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 4.0), dpi=300, sharey=True)

    conditions = ["Ref (C0)", "Para", "NLLB", "OPUS", "LLM", "Synth"]
    x = np.arange(len(conditions))

    # Indonesian
    id_bm25 = [0.4245, 0.4311, 0.2991, 0.2797, 0.4046, 0.4777]
    id_qwen = [0.5415, 0.5294, 0.5098, 0.4874, 0.5155, 0.5485]
    id_me5 = [0.5413, 0.5315, 0.5117, 0.4927, 0.5270, 0.5492]

    # Thai
    th_bm25 = [0.3278, 0.1771, 0.0389, 0.0188, 0.0870, 0.2728]
    th_qwen = [0.7202, 0.7134, 0.5863, 0.4769, 0.7022, 0.6809]
    th_me5 = [0.7248, 0.7126, 0.6057, 0.5075, 0.7103, 0.6858]

    # Plot Indonesian
    ax1.plot(x, id_qwen, marker="o", color="#2563EB", linewidth=2.2, markersize=6, label="Qwen3-0.6B")
    ax1.plot(x, id_me5, marker="s", color="#4F46E5", linewidth=2.2, linestyle="--", markersize=6, label="mE5-large")
    ax1.plot(x, id_bm25, marker="^", color="#E11D48", linewidth=2.2, markersize=7, label="BM25 (Lexical)")
    ax1.set_title("Indonesian (id) [Latin Script]", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Retrieval Score (nDCG@10)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower left", fontsize=9, framealpha=0.95)

    # Plot Thai
    ax2.plot(x, th_qwen, marker="o", color="#2563EB", linewidth=2.2, markersize=6, label="Qwen3-0.6B")
    ax2.plot(x, th_me5, marker="s", color="#4F46E5", linewidth=2.2, linestyle="--", markersize=6, label="mE5-large")
    ax2.plot(x, th_bm25, marker="^", color="#E11D48", linewidth=2.2, markersize=7, label="BM25 (Lexical)")
    ax2.set_title("Thai (th) [Continuous Script]", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions, fontsize=9.5, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_script_comparison.png", bbox_inches="tight", dpi=300)
    plt.savefig(FIGURES_DIR / "fig3_script_comparison.pdf", bbox_inches="tight")
    plt.close()
    print("[✓] Generated Figure 3: Language Script Comparison")


if __name__ == "__main__":
    plot_drawio_pipeline()
    plot_family_interaction_contrasts()
    plot_language_script_comparison()
