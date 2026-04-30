from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_tradeoff_grounding_coherence(file_path):
    file_path = Path(file_path)

    # Load data
    df = pd.read_excel(file_path)

    required_cols = [
        "Query",
        "Condition",
        "Theoretical Grounding",
        "Coherence",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Missing column: {col}\nAvailable columns: {list(df.columns)}"
            )

    df["Theoretical Grounding"] = pd.to_numeric(
        df["Theoretical Grounding"], errors="coerce"
    )
    df["Coherence"] = pd.to_numeric(df["Coherence"], errors="coerce")

    df = df.dropna(subset=["Theoretical Grounding", "Coherence", "Condition"])

    # Style
    baseline_color = "#B0B0B0"
    rag_color = "#2E86C1"
    edge = "#333333"
    text = "#111111"

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
    })

    fig, ax = plt.subplots(figsize=(7.6, 6.5))

    rng = np.random.default_rng(42)
    jitter_strength = 0.08

    condition_specs = {
        "Baseline": {
            "color": baseline_color,
            "label": "Baseline responses",
        },
        "RAG": {
            "color": rag_color,
            "label": "RAG responses",
        },
    }

    means = {}

    # Scatter points
    for condition, spec in condition_specs.items():
        subset = df[
            df["Condition"].astype(str).str.lower() == condition.lower()
        ].copy()

        if subset.empty:
            continue

        x = subset["Coherence"].to_numpy()
        y = subset["Theoretical Grounding"].to_numpy()

        x_jitter = x + rng.normal(0, jitter_strength, size=len(x))
        y_jitter = y + rng.normal(0, jitter_strength, size=len(y))

        ax.scatter(
            x_jitter,
            y_jitter,
            s=45,
            color=spec["color"],
            edgecolor=edge,
            linewidth=0.55,
            alpha=0.55,
            label=spec["label"],
            zorder=3,
        )

        mean_x = x.mean()
        mean_y = y.mean()
        means[condition] = (mean_x, mean_y)

        ax.scatter(
            mean_x,
            mean_y,
            s=190,
            color=spec["color"],
            edgecolor=edge,
            linewidth=1.5,
            marker="D",
            alpha=1.0,
            zorder=6,
            label=f"{condition} mean",
        )

        ax.text(
            mean_x + 0.04,
            mean_y + 0.04,
            f"{condition}\nmean",
            fontsize=8.3,
            color=text,
            ha="left",
            va="bottom",
            zorder=7,
        )

    # Arrow between means
    if "Baseline" in means and "RAG" in means:
        bx, by = means["Baseline"]
        rx, ry = means["RAG"]

        ax.annotate(
            "",
            xy=(rx, ry),
            xytext=(bx, by),
            arrowprops=dict(
                arrowstyle="->",
                color=edge,
                linewidth=1.2,
                shrinkA=14,
                shrinkB=14,
            ),
            zorder=5,
        )

        ax.text(
            (bx + rx) / 2,
            (by + ry) / 2 + 0.12,
            "mean shift",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=edge,
            zorder=7,
        )

    # Labels
    ax.set_title(
        "Epistemic Trade-off: Grounding Increases, Coherence Varies",
        fontsize=13,
        fontweight="bold",
        pad=15,
        color=text,
    )

    ax.set_xlabel("Coherence", fontsize=10, color=text)
    ax.set_ylabel("Theoretical grounding", fontsize=10, color=text)

    ax.set_xlim(-0.20, 2.25)
    ax.set_ylim(-0.20, 2.25)

    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])

    ax.set_xticklabels(["0\nlow", "1\npartial", "2\nhigh"], fontsize=9)
    ax.set_yticklabels(["0\nlow", "1\npartial", "2\nhigh"], fontsize=9)

    # Clean style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(edge)
    ax.spines["bottom"].set_color(edge)

    ax.grid(alpha=0.16, linewidth=0.8)
    ax.set_axisbelow(True)

    # Clean legend
    handles, labels = ax.get_legend_handles_labels()

    # keep only response clouds, not mean markers
    clean_handles = []
    clean_labels = []

    for handle, label in zip(handles, labels):
        if label in ["Baseline responses", "RAG responses"]:
            clean_handles.append(handle)
            clean_labels.append(label)

    ax.legend(
        clean_handles,
        clean_labels,
        frameon=False,
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        borderaxespad=0.0,
    )

    # Save
    output_dir = file_path.parent / "figures"
    output_dir.mkdir(exist_ok=True)

    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "fig_06_tradeoff_grounding_coherence.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "fig_06_tradeoff_grounding_coherence.pdf", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "fig_06_tradeoff_grounding_coherence.svg", bbox_inches="tight")

    plt.close()

    print("Trade-off plot saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    plot_tradeoff_grounding_coherence(
        "results/evaluation/full_30_coding_filled.xlsx"
    )