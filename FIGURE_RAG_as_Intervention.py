import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle


def create_theory_tech_diagram():
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # --------------------
    # Colors
    # --------------------
    neutral_fill = "#F7F7F7"
    neutral_edge = "#333333"

    pasq_color = "#C0392B"      # red: Pasquinelli
    foucault_color = "#2E86C1"  # blue: Foucault

    muted = "#666666"
    text = "#111111"

    # --------------------
    # Helpers
    # --------------------
    def add_box(x, y, w, h, title, subtitle, edge_color=neutral_edge):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=neutral_fill,
            edgecolor=edge_color,
            linewidth=2.0 if edge_color != neutral_edge else 1.3,
        )
        ax.add_patch(box)

        ax.text(
            x + w / 2,
            y + h * 0.62,
            title,
            ha="center",
            va="center",
            fontsize=11.5,
            fontweight="bold",
            color=edge_color if edge_color != neutral_edge else text,
        )

        ax.text(
            x + w / 2,
            y + h * 0.32,
            subtitle,
            ha="center",
            va="center",
            fontsize=9.2,
            color=muted,
            style="italic",
        )

    def add_arrow(x1, y1, x2, y2, label, color=neutral_edge):
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.2,
            color=color,
        )
        ax.add_patch(arrow)

        ax.text(
            x1 + 0.08,
            (y1 + y2) / 2,
            label,
            ha="left",
            va="center",
            fontsize=9,
            color=color,
        )

    # --------------------
    # Title
    # --------------------
    ax.text(
        0.5,
        0.94,
        "RAG as Epistemic Intervention: Linking Theory and Technique",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=text,
    )

    ax.text(
        0.5,
        0.905,
        "How classification, approximation and selection produce epistemic visibility",
        ha="center",
        va="center",
        fontsize=10,
        color=muted,
    )

    # --------------------
    # Pipeline
    # --------------------
    x = 0.30
    w = 0.36
    h = 0.065

    ys = [0.79, 0.68, 0.57, 0.46, 0.35, 0.24]

    steps = [
        ("Corpus", "text archive", neutral_edge),
        ("Classification", "chunking + metadata", pasq_color),
        ("Embedding Space", "relational approximation", pasq_color),
        ("Retrieval", "top-k selection", foucault_color),
        ("Visible Context", "epistemic visibility", foucault_color),
        ("Generation", "articulation under constraint", neutral_edge),
    ]

    labels = [
        ("segmentation", neutral_edge),
        ("comparability", pasq_color),
        ("similarity ranking", pasq_color),
        ("selective visibility", foucault_color),
        ("conditions what can be articulated", foucault_color),
    ]

    for i, (title, subtitle, edge) in enumerate(steps):
        add_box(x, ys[i], w, h, title, subtitle, edge)

        if i < len(steps) - 1:
            add_arrow(
                x + w / 2,
                ys[i],
                x + w / 2,
                ys[i + 1] + h,
                labels[i][0],
                labels[i][1],
            )

    # --------------------
    # Theory anchors
    # --------------------
    ax.text(
        0.08,
        0.135,
        "Pasquinelli",
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=pasq_color,
    )
    ax.text(
        0.08,
        0.105,
        "classification + approximation",
        ha="left",
        va="center",
        fontsize=9.5,
        color=text,
    )
    ax.text(
        0.08,
        0.075,
        "Elements become comparable through\nsegmentation and similarity relations.",
        ha="left",
        va="top",
        fontsize=8.8,
        color=muted,
        linespacing=1.25,
    )

    ax.text(
        0.58,
        0.135,
        "Foucault",
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=foucault_color,
    )
    ax.text(
        0.58,
        0.105,
        "visibility through ordering",
        ha="left",
        va="center",
        fontsize=9.5,
        color=text,
    )
    ax.text(
        0.58,
        0.075,
        "What becomes visible conditions\nwhat can be articulated as knowledge.",
        ha="left",
        va="top",
        fontsize=8.8,
        color=muted,
        linespacing=1.25,
    )

    # --------------------
    # Compact legend
    # --------------------
    legend_x = 0.735
    legend_y = 0.79

    ax.text(
        legend_x,
        legend_y + 0.06,
        "Theoretical mapping",
        ha="left",
        va="center",
        fontsize=9.8,
        fontweight="bold",
        color=text,
    )

    legend_items = [
        (pasq_color, "Pasquinelli: classification / approximation"),
        (foucault_color, "Foucault: visibility / ordering"),
        (neutral_edge, "technical transition"),
    ]

    for i, (color, label) in enumerate(legend_items):
        yy = legend_y + 0.025 - i * 0.035
        ax.add_patch(
            Rectangle(
                (legend_x, yy - 0.01),
                0.018,
                0.018,
                facecolor="white",
                edgecolor=color,
                linewidth=1.8,
            )
        )
        ax.text(
            legend_x + 0.028,
            yy,
            label,
            ha="left",
            va="center",
            fontsize=8.3,
            color=muted,
        )

    # --------------------

    plt.tight_layout()
    plt.savefig("fig_theory_tech_diagram.png", dpi=300, bbox_inches="tight")
    plt.savefig("fig_theory_tech_diagram.pdf", bbox_inches="tight")
    plt.savefig("fig_theory_tech_diagram.svg", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    create_theory_tech_diagram()