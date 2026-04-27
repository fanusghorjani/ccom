from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_grouped_score_comparison(file_path):
    file_path = Path(file_path)
    df = pd.read_excel(file_path)

    required_columns = [
        "Condition",
        "Theoretical Grounding",
        "Political Positioning",
        "Specificity",
        "Coherence",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    df["Condition"] = df["Condition"].astype(str).str.strip().str.lower()

    baseline_df = df[df["Condition"] == "baseline"]
    rag_df = df[df["Condition"] == "rag"]

    if baseline_df.empty:
        raise ValueError("No rows found for Condition = 'baseline'")
    if rag_df.empty:
        raise ValueError("No rows found for Condition = 'rag'")

    source_cols = [
        "Theoretical Grounding",
        "Political Positioning",
        "Specificity",
        "Coherence",
    ]

    x_labels = [
        "Theoretical\nGrounding",
        "Political\nPositioning",
        "Specificity",
        "Coherence",
    ]

    baseline_means = [
        pd.to_numeric(baseline_df[col], errors="coerce").mean()
        for col in source_cols
    ]
    rag_means = [
        pd.to_numeric(rag_df[col], errors="coerce").mean()
        for col in source_cols
    ]

    baseline_std = [
        pd.to_numeric(baseline_df[col], errors="coerce").std()
        for col in source_cols
    ]
    rag_std = [
        pd.to_numeric(rag_df[col], errors="coerce").std()
        for col in source_cols
    ]

    neutral = "#B0B0B0"
    rag_color = "#2E86C1"
    edge = "#333333"
    text = "#111111"
    muted = "#666666"

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
    })

    fig, ax = plt.subplots(figsize=(9.5, 5.8))

    x = np.arange(len(source_cols))
    width = 0.34

    bars_baseline = ax.bar(
        x - width / 2,
        baseline_means,
        width,
        label="Baseline",
        color=neutral,
        edgecolor=edge,
        linewidth=0.9,
    )

    bars_rag = ax.bar(
        x + width / 2,
        rag_means,
        width,
        label="RAG",
        color=rag_color,
        edgecolor=edge,
        linewidth=0.9,
    )

    ax.errorbar(
        x - width / 2,
        baseline_means,
        yerr=baseline_std,
        fmt="none",
        ecolor=edge,
        elinewidth=0.8,
        capsize=3,
        alpha=0.55,
    )

    ax.errorbar(
        x + width / 2,
        rag_means,
        yerr=rag_std,
        fmt="none",
        ecolor=edge,
        elinewidth=0.8,
        capsize=3,
        alpha=0.55,
    )

    ax.set_title(
        "Epistemic Effects of RAG: Structured Coding Comparison",
        fontsize=13,
        fontweight="bold",
        pad=14,
        color=text,
    )

    ax.set_ylabel("Mean coding score", fontsize=10, color=text)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylim(0, 2.25)

    ax.tick_params(axis="y", labelsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(edge)
    ax.spines["bottom"].set_color(edge)

    ax.grid(axis="y", alpha=0.15, linewidth=0.8)
    ax.set_axisbelow(True)

    ax.legend(
        frameon=False,
        fontsize=9,
        loc="upper right",
    )



    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()

            if pd.isna(height) or height == 0:
                continue

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.055,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=text,
            )

    add_value_labels(bars_baseline)
    add_value_labels(bars_rag)

    output_dir = file_path.parent / "figures"
    output_dir.mkdir(exist_ok=True)

    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "fig_05_structured_coding_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "fig_05_structured_coding_comparison.pdf", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "fig_05_structured_coding_comparison.svg", bbox_inches="tight")

    plt.close()

    print("Plot saved to:", output_dir)
    print("\nMeans:")
    for d, b, r in zip(source_cols, baseline_means, rag_means):
        print(f"{d}: Baseline={b:.2f}, RAG={r:.2f}")


if __name__ == "__main__":
    plot_grouped_score_comparison(
        r"C:\Users\SinaElBasiouni\Documents\3_Semester\Thesis\data\class_conscious_organization_files\results\evaluation\full_30_coding_filled.xlsx"
    )