from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def find_column(df, possible_names):
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for name in possible_names:
        key = name.lower().strip()
        if key in cols_lower:
            return cols_lower[key]
    raise ValueError(f"No matching column found for: {possible_names}\nAvailable columns:\n{list(df.columns)}")


def plot_grouped_score_comparison(file_path):
    file_path = Path(file_path)

    df = pd.read_excel(file_path)

    baseline_cols = {
        "Theoretical grounding": find_column(df, [
            "baseline_theoretical_grounding",
            "baseline_grounding",
            "theoretical_grounding_baseline",
            "grounding_baseline"
        ]),
        "Political positioning": find_column(df, [
            "baseline_political_positioning",
            "baseline_positioning",
            "political_positioning_baseline",
            "positioning_baseline"
        ]),
        "Specificity": find_column(df, [
            "baseline_specificity",
            "specificity_baseline",
            "baseline_specificity_abstraction",
            "baseline_specificity_vs_abstraction"
        ]),
        "Coherence": find_column(df, [
            "baseline_epistemic_coherence",
            "baseline_coherence",
            "epistemic_coherence_baseline",
            "coherence_baseline"
        ]),
    }

    rag_cols = {
        "Theoretical grounding": find_column(df, [
            "rag_theoretical_grounding",
            "rag_grounding",
            "theoretical_grounding_rag",
            "grounding_rag"
        ]),
        "Political positioning": find_column(df, [
            "rag_political_positioning",
            "rag_positioning",
            "political_positioning_rag",
            "positioning_rag"
        ]),
        "Specificity": find_column(df, [
            "rag_specificity",
            "specificity_rag",
            "rag_specificity_abstraction",
            "rag_specificity_vs_abstraction"
        ]),
        "Coherence": find_column(df, [
            "rag_epistemic_coherence",
            "rag_coherence",
            "epistemic_coherence_rag",
            "coherence_rag"
        ]),
    }

    dimensions = list(baseline_cols.keys())

    baseline_means = [pd.to_numeric(df[baseline_cols[d]], errors="coerce").mean() for d in dimensions]
    rag_means = [pd.to_numeric(df[rag_cols[d]], errors="coerce").mean() for d in dimensions]

    baseline_std = [pd.to_numeric(df[baseline_cols[d]], errors="coerce").std() for d in dimensions]
    rag_std = [pd.to_numeric(df[rag_cols[d]], errors="coerce").std() for d in dimensions]

    neutral = "#B0B0B0"
    rag_color = "#2E86C1"
    edge = "#333333"
    text = "#111111"

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
    })

    fig, ax = plt.subplots(figsize=(9.5, 5.8))

    x = np.arange(len(dimensions))
    width = 0.36

    bars_baseline = ax.bar(
        x - width / 2,
        baseline_means,
        width,
        label="Baseline",
        color=neutral,
        edgecolor=edge,
        linewidth=1.0
    )

    bars_rag = ax.bar(
        x + width / 2,
        rag_means,
        width,
        label="RAG",
        color=rag_color,
        edgecolor=edge,
        linewidth=1.0
    )

    ax.errorbar(
        x - width / 2,
        baseline_means,
        yerr=baseline_std,
        fmt="none",
        ecolor=edge,
        elinewidth=0.8,
        capsize=3,
        alpha=0.65
    )

    ax.errorbar(
        x + width / 2,
        rag_means,
        yerr=rag_std,
        fmt="none",
        ecolor=edge,
        elinewidth=0.8,
        capsize=3,
        alpha=0.65
    )

    ax.set_title(
        "Structured Coding: Baseline vs RAG",
        fontsize=13,
        fontweight="bold",
        pad=15,
        color=text
    )

    ax.set_ylabel("Mean coding score", fontsize=10, color=text)
    ax.set_xticks(x)
    ax.set_xticklabels(dimensions, fontsize=9)
    ax.set_ylim(0, 2.15)

    ax.tick_params(axis="y", labelsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(edge)
    ax.spines["bottom"].set_color(edge)

    ax.grid(axis="y", alpha=0.15, linewidth=0.8)
    ax.set_axisbelow(True)

    ax.legend(frameon=False, fontsize=9, loc="upper left")

    for bars in [bars_baseline, bars_rag]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.04,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.3,
                color=text
            )

    output_dir = file_path.parent / "figures"
    output_dir.mkdir(exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_dir / "fig_structured_coding_comparison.png", dpi=300)
    plt.savefig(output_dir / "fig_structured_coding_comparison.pdf")
    plt.savefig(output_dir / "fig_structured_coding_comparison.svg")
    plt.close()

    print("Plot saved to:", output_dir)
    print("\nColumns used:")
    print("Baseline:", baseline_cols)
    print("RAG:", rag_cols)
    print("\nMeans:")
    for d, b, r in zip(dimensions, baseline_means, rag_means):
        print(f"{d}: Baseline={b:.2f}, RAG={r:.2f}")


if __name__ == "__main__":
    plot_grouped_score_comparison(
        r"C:\Users\SinaElBasiouni\Documents\3_Semester\Thesis\data\class_conscious_organization_files\results\evaluation\full_30_coding_filled.xlsx"
    )