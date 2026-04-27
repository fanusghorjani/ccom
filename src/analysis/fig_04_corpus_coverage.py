from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_doc_ids(entry):
    if pd.isna(entry):
        return []

    entry = str(entry).strip()

    if "|" in entry:
        return [x.strip() for x in entry.split("|") if x.strip()]

    return [entry]


def plot_coverage(file_path):
    file_path = Path(file_path)

    # -----------------------
    # Load data
    # -----------------------
    df = pd.read_excel(file_path)

    column = "rag_retrieved_doc_ids"

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")

    # -----------------------
    # Extract retrieved docs
    # -----------------------
    retrieved_docs = set()

    for entry in df[column].dropna():
        retrieved_docs.update(parse_doc_ids(entry))

    # -----------------------
    # Total corpus size
    # -----------------------
    # 🔴 WICHTIG: anpassen falls nötig
    total_docs = 50   # <-- HIER ggf. ändern!

    retrieved_count = len(retrieved_docs)
    not_retrieved = total_docs - retrieved_count

    # -----------------------
    # Plot
    # -----------------------
    labels = ["Retrieved", "Not Retrieved"]
    values = [retrieved_count, not_retrieved]

    colors = ["#2E86C1", "#B0B0B0"]  # blau = sichtbar, grau = unsichtbar
    edge = "#333333"
    text = "#111111"

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
    })

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    bars = ax.bar(
        labels,
        values,
        color=colors,
        edgecolor=edge,
        linewidth=1.2
    )

    # -----------------------
    # Labels
    # -----------------------
    ax.set_title(
        "Limited Corpus Coverage in Retrieval",
        fontsize=12.5,
        fontweight="bold",
        pad=15,
        color=text
    )

    ax.set_ylabel("Number of documents", fontsize=10, color=text)

    # -----------------------
    # Value labels
    # -----------------------
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            str(int(height)),
            ha="center",
            va="bottom",
            fontsize=9,
            color=text
        )

    # -----------------------
    # Clean style
    # -----------------------
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(edge)
    ax.spines["bottom"].set_color(edge)

    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)

    # -----------------------
    # Save
    # -----------------------

    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "fig_04_corpus_coverage.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "fig_04_corpus_coverage.pdf", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "fig_04_corpus_coverage.svg", bbox_inches="tight")

    plt.close()

    print("Coverage plot saved to:", output_dir)
    print(f"Retrieved: {retrieved_count}")
    print(f"Not retrieved: {not_retrieved}")


if __name__ == "__main__":
    plot_coverage(
    "results/analysis/full_analysis.xlsx"
    )