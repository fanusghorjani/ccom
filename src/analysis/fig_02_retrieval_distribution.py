from pathlib import Path
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

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


def plot_retrieval_concentration(file_path):
    file_path = Path(file_path)

    # -----------------------
    # Load data
    # -----------------------
    df = pd.read_excel(file_path)

    column = "rag_retrieved_doc_ids"

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")

    # -----------------------
    # Flatten document IDs
    # -----------------------
    all_docs = []
    for entry in df[column].dropna():
        all_docs.extend(parse_doc_ids(entry))

    # -----------------------
    # Count frequency
    # -----------------------
    counts = Counter(all_docs)
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    top_n = 15
    top_docs = sorted_counts[:top_n]

    labels = [doc for doc, _ in top_docs][::-1]
    values = [count for _, count in top_docs][::-1]

    # -----------------------
    # Style (consistent)
    # -----------------------
    pasq_color = "#C0392B"
    neutral = "#B0B0B0"
    edge = "#333333"
    text = "#111111"

    colors = [pasq_color if i < 2 else neutral for i in range(len(values))]
    colors = colors[::-1]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
    })

    # -----------------------
    # Plot
    # -----------------------
    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    ax.barh(
        labels,
        values,
        color=colors,
        edgecolor=edge,
        linewidth=1.0
    )

    # -----------------------
    # Labels & Title
    # -----------------------
    ax.set_title(
        "Retrieval Distribution: Visibility Concentrates in a Small Document Core",
        fontsize=13,
        fontweight="bold",
        pad=15,
        color=text
    )

    ax.set_xlabel("Retrieval frequency", fontsize=10, color=text)
    ax.set_ylabel("Retrieved document ID", fontsize=10, color=text)

    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)

    # -----------------------
    # Clean style
    # -----------------------
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(edge)
    ax.spines["bottom"].set_color(edge)

    ax.grid(axis="x", alpha=0.15, linewidth=0.8)
    ax.set_axisbelow(True)

    # -----------------------
    # Value labels
    # -----------------------
    for i, value in enumerate(values):
        ax.text(
            value + max(values) * 0.02,
            i,
            str(value),
            va="center",
            fontsize=8.5,
            color=text,
        )

    # -----------------------
    # Save
    # -----------------------
    output_dir = file_path.parent / "figures"
    output_dir.mkdir(exist_ok=True)

    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "fig_02_retrieval_distribution.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "fig_02_retrieval_distribution.pdf", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "fig_02_retrieval_distribution.svg", bbox_inches="tight")  

    plt.close()

    print("Plot saved to:", output_dir)


if __name__ == "__main__":
    plot_retrieval_concentration(
    "results/analysis/full_analysis.xlsx"
    )