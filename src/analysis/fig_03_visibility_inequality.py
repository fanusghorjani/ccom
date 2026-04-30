from pathlib import Path
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_doc_ids(entry):
    """Parse document IDs separated by |"""
    if pd.isna(entry):
        return []

    entry = str(entry).strip()

    if "|" in entry:
        return [x.strip() for x in entry.split("|") if x.strip()]

    return [entry]


def plot_visibility_inequality(file_path):
    file_path = Path(file_path)

    # Load data
    df = pd.read_excel(file_path)

    column = "rag_retrieved_doc_ids"

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")

    # Flatten retrieved document IDs
    all_docs = []
    for entry in df[column].dropna():
        all_docs.extend(parse_doc_ids(entry))

    counts = Counter(all_docs)

    if not counts:
        raise ValueError("No retrieved document IDs found.")

    # Lorenz curve calculation
    values = np.array(sorted(counts.values()))

    cumulative = np.cumsum(values)
    cumulative = np.insert(cumulative, 0, 0)

    cumulative_share = cumulative / cumulative[-1]
    document_share = np.linspace(0, 1, len(cumulative_share))

    # Gini coefficient
    gini = 1 - 2 * np.trapezoid(cumulative_share, document_share)

    # Style
    foucault_color = "#2E86C1"
    neutral = "#B0B0B0"
    edge = "#333333"
    text = "#111111"
    muted = "#666666"

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
    })

    # Plot
    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    ax.plot(
        document_share,
        cumulative_share,
        color=foucault_color,
        linewidth=2.4,
        label="Observed retrieval visibility"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        color=neutral,
        linewidth=1.4,
        linestyle="--",
        label="Equal visibility"
    )

    # Labels & title
    ax.set_title(
        "Unequal Visibility in Retrieval",
        fontsize=13,
        fontweight="bold",
        pad=15,
        color=text
    )

    ax.set_xlabel("Cumulative share of retrieved documents", fontsize=10, color=text)
    ax.set_ylabel("Cumulative share of retrieval appearances", fontsize=10, color=text)

    ax.text(
        0.05,
        0.88,
        f"Gini = {gini:.2f}",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        color=text
    )

    # Clean style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(edge)
    ax.spines["bottom"].set_color(edge)

    ax.grid(alpha=0.15, linewidth=0.8)
    ax.set_axisbelow(True)

    ax.legend(
        frameon=False,
        fontsize=9,
        loc="lower right"
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

   
    # Save

    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "fig_03_visibility_inequality.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "fig_03_visibility_inequality.pdf", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "fig_03_visibility_inequality.svg", bbox_inches="tight")

    plt.close()

    print("Plot saved to:", OUTPUT_DIR)
    print(f"Gini coefficient: {gini:.3f}")


if __name__ == "__main__":
   plot_visibility_inequality(
    "results/analysis/full_analysis.xlsx"
    )