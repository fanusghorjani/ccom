"""Generate publication-quality thesis figures for:
1) Retrieval distribution
2) Field of visibility (organizational structure case)
3) Classification -> visibility mechanism
4) Transformation diagram
"""

from __future__ import annotations

import math
import re
import textwrap
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle


# ----------------------------
# Shared design system
# ----------------------------
COLORS = {
    "blue": "#2F5D9B",
    "red": "#B04652",
    "pink": "#D88FA3",
    "orange": "#D08B45",
    "gray": "#666666",
    "light_gray": "#D9D9D9",
    "ink": "#1A1A1A",
}

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.titlesize": 20,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "axes.edgecolor": COLORS["ink"],
        "axes.linewidth": 0.8,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "grid.color": "#EAEAEA",
        "grid.linewidth": 0.8,
        "lines.linewidth": 2.1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# ----------------------------
# Utility helpers
# ----------------------------
def wrap(s: str, width: int = 32) -> str:
    return "\n".join(textwrap.wrap(str(s), width=width, break_long_words=False))


def ensure_out(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def find_column(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    """Find a column by exact or normalized name."""
    cols = list(df.columns)
    norm_map = {
        re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_"): c
        for c in cols
    }

    for cand in candidates:
        if cand in cols:
            return cand

    for cand in candidates:
        norm = re.sub(r"[^a-z0-9]+", "_", cand.strip().lower()).strip("_")
        if norm in norm_map:
            return norm_map[norm]

    if required:
        raise KeyError(
            f"Could not find any of these columns: {candidates}\n"
            f"Available columns are: {cols}"
        )
    return None


# ----------------------------
# Minimal XLSX reader (no openpyxl dependency)
# ----------------------------
NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "p": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def _norm_target(target: str) -> str:
    target = target.lstrip("/")
    return target if target.startswith("xl/") else f"xl/{target}"


def _col_idx(col: str) -> int:
    n = 0
    for ch in col:
        if "A" <= ch <= "Z":
            n = n * 26 + ord(ch) - 64
    return n - 1


def read_xlsx(path: Path, sheet_name: str | None = None) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        relmap = {
            r.attrib["Id"]: _norm_target(r.attrib["Target"])
            for r in rels.findall("p:Relationship", NS)
        }

        sheets = []
        for s in wb.findall("a:sheets/a:sheet", NS):
            name = s.attrib["name"]
            rid = s.attrib[f"{{{NS['r']}}}id"]
            sheets.append((name, relmap[rid]))

        if not sheets:
            raise ValueError(f"No sheets found in {path}")

        if sheet_name is None:
            sheet_name = sheets[0][0]

        sheet_lookup = dict(sheets)
        if sheet_name not in sheet_lookup:
            raise KeyError(f"Sheet '{sheet_name}' not found in {path}. Available: {list(sheet_lookup)}")

        ws_target = sheet_lookup[sheet_name]

        sst = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", NS):
                sst.append("".join(t.text or "" for t in si.findall(".//a:t", NS)))

        ws = ET.fromstring(z.read(ws_target))
        rows = []

        for row in ws.findall("a:sheetData/a:row", NS):
            vals = {}
            for c in row.findall("a:c", NS):
                ref = c.attrib.get("r", "A1")
                match = re.match(r"([A-Z]+)", ref)
                if not match:
                    continue

                col = match.group(1)
                idx = _col_idx(col)
                t = c.attrib.get("t")

                if t == "inlineStr":
                    val = "".join(x.text or "" for x in c.findall(".//a:t", NS))
                else:
                    v = c.find("a:v", NS)
                    val = (v.text if v is not None else "") or ""
                    if t == "s" and val:
                        val = sst[int(val)]
                vals[idx] = val

            if vals:
                maxc = max(vals)
                arr = [""] * (maxc + 1)
                for i, v in vals.items():
                    arr[i] = v
                rows.append(arr)

    if not rows:
        raise ValueError(f"No tabular rows found in {path}")

    header = rows[0]
    if not header:
        raise ValueError(f"Empty header row in {path}")

    data = [r + [""] * (len(header) - len(r)) for r in rows[1:]]
    return pd.DataFrame(data, columns=header)


# ----------------------------
# Helpers for text analytics
# ----------------------------
STOPWORDS = {
    "the", "and", "of", "to", "in", "a", "is", "that", "for", "with", "as", "on", "be", "are",
    "it", "this", "by", "or", "an", "from", "their", "they", "can", "should", "all", "not",
    "more", "such", "within", "into", "through", "which", "at", "than", "also", "have", "has",
    "will", "its", "between", "about", "how", "what", "where", "who", "while", "these", "those",
    "them", "being", "both", "other", "each", "may", "must", "do", "does", "did", "our", "your",
    "you", "we", "there", "here", "if", "but", "so", "up", "out", "over",
    "organization", "organizations", "structure", "process", "decision", "making",
    "members", "leadership",
}


def tokenize(text: str) -> list[str]:
    return [
        t
        for t in re.findall(r"[a-zA-Z][a-zA-Z\-']{2,}", str(text).lower())
        if t not in STOPWORDS
    ]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[\.!?])\s+", str(text).strip())
    return [p.strip() for p in parts if p.strip()]


def cosine_counter(c1: Counter, c2: Counter) -> float:
    if not c1 or not c2:
        return 0.0
    shared = set(c1) & set(c2)
    num = sum(c1[t] * c2[t] for t in shared)
    den = math.sqrt(sum(v * v for v in c1.values()) * sum(v * v for v in c2.values()))
    return num / den if den else 0.0


def top_terms(text: str, n: int = 10) -> list[str]:
    ctr = Counter(tokenize(text))
    return [w for w, _ in ctr.most_common(n)]


def build_cooc(text: str, terms: list[str]) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
    node_w = Counter()
    edge_w = Counter()
    termset = set(terms)

    for sent in split_sentences(text):
        toks = [t for t in tokenize(sent) if t in termset]
        uniq = sorted(set(toks))
        for t in uniq:
            node_w[t] += 1
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                edge_w[(uniq[i], uniq[j])] += 1

    return dict(node_w), dict(edge_w)


def radial_pos(terms: list[str], radius: float = 1.0) -> dict[str, tuple[float, float]]:
    pos = {}
    for i, t in enumerate(terms):
        ang = 2 * np.pi * i / len(terms) + 0.12
        pos[t] = (radius * np.cos(ang), radius * np.sin(ang))
    return pos


def robust_scale(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    allv = np.array(list(a.values) + list(b.values), dtype=float)
    lo, hi = np.quantile(allv, 0.05), np.quantile(allv, 0.95)

    def sc(x: float) -> float:
        if hi <= lo:
            return 50.0
        return 100 * float(np.clip((x - lo) / (hi - lo), 0, 1))

    return sc(float(a.median())), sc(float(b.median()))


# ----------------------------
# Figure 1
# ----------------------------
def make_retrieval_distribution(
    retrieval: pd.DataFrame,
    corpus: pd.DataFrame,
    out_dir: Path,
    doc_id_col: str,
    k_col: str,
    title_col: str | None,
    tradition_col: str | None,
) -> None:
    retr10 = retrieval[retrieval[k_col].astype(str) == "10"].copy()
    if retr10.empty:
        retr10 = retrieval.copy()

    counts = retr10[doc_id_col].value_counts()

    meta_cols = [doc_id_col]
    if title_col:
        meta_cols.append(title_col)
    if tradition_col:
        meta_cols.append(tradition_col)

    meta = corpus[meta_cols].copy()
    meta = meta.drop_duplicates(subset=[doc_id_col])

    if title_col:
        meta[title_col] = meta[title_col].replace("", "untitled")

    count_df = counts.rename_axis(doc_id_col).reset_index(name="retrieval_count").merge(
        meta, on=doc_id_col, how="left"
    )
    count_df = count_df.sort_values("retrieval_count", ascending=False)

    topn = 14
    top = count_df.head(topn).copy()
    other_n = int(count_df.iloc[topn:]["retrieval_count"].sum())

    if other_n > 0:
        row = {
            doc_id_col: "Other",
            "retrieval_count": other_n,
        }
        if title_col:
            row[title_col] = "Remaining documents"
        if tradition_col:
            row[tradition_col] = "mixed"

        top = pd.concat([top, pd.DataFrame([row])], ignore_index=True)

    labels = []
    for r in top.itertuples():
        did = getattr(r, doc_id_col)
        title = getattr(r, title_col) if title_col else did
        tradition = getattr(r, tradition_col) if tradition_col else ""
        label = f"{did} · {str(title).replace(chr(10), ' ')}"
        if did != "Other" and tradition:
            label += f" ({tradition})"
        labels.append(wrap(label, 46))

    vals = top["retrieval_count"].to_numpy()
    y = np.arange(len(top))[::-1]

    fig, ax = plt.subplots(figsize=(16, 11))
    for yi, xi, did in zip(y, vals, top[doc_id_col]):
        color = COLORS["pink"] if did == "Other" else COLORS["blue"]
        dot = COLORS["orange"] if did == "Other" else COLORS["red"]
        ax.hlines(yi, 0, xi, color=color, lw=3.0, alpha=0.9)
        ax.plot(xi, yi, "o", color=dot, ms=9)
        ax.text(xi + 1.0, yi, f"{int(xi)}", va="center", ha="left", fontsize=12, color=COLORS["ink"])

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Number of retrieval appearances across all qids")
    ax.set_title("Retrieval Distribution: Visibility Concentrates in a Small Document Core", loc="left", pad=18)
    ax.grid(axis="x", alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max(vals) * 1.25 if len(vals) else 1)
    plt.tight_layout(pad=2.0)

    for ext in ["png", "pdf"]:
        out = out_dir / f"fig_retrieval_distribution.{ext}"
        ensure_out(out)
        fig.savefig(out, dpi=320, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Figure 2
# ----------------------------
def make_field_of_visibility(
    analysis: pd.DataFrame,
    out_dir: Path,
    qid_col: str,
    baseline_col: str,
    rag_col: str,
) -> None:
    case = analysis[analysis[qid_col].astype(str).str.lower() == "q05"]
    if case.empty:
        case = analysis.iloc[[0]]

    case_row = case.iloc[0]
    baseline_text = str(case_row[baseline_col])
    rag_text = str(case_row[rag_col])

    base_terms = top_terms(baseline_text, 9)
    rag_terms = top_terms(rag_text, 10)

    base_nodes, base_edges = build_cooc(baseline_text, base_terms)
    rag_nodes, rag_edges = build_cooc(rag_text, rag_terms)

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    panels = [
        (
            axes[0],
            base_terms,
            base_nodes,
            base_edges,
            "Baseline visibility field\n(generic-managerial ordering)",
            COLORS["pink"],
            COLORS["gray"],
        ),
        (
            axes[1],
            rag_terms,
            rag_nodes,
            rag_edges,
            "RAG visibility field\n(theoretical-historical reordering)",
            COLORS["blue"],
            COLORS["red"],
        ),
    ]

    for ax, terms, nodes, edges, title, node_color, edge_color in panels:
        if not terms:
            ax.text(0.5, 0.5, "No terms available", ha="center", va="center")
            ax.axis("off")
            continue

        pos = radial_pos(terms, radius=1.05)

        for (u, v), w in sorted(edges.items(), key=lambda x: x[1]):
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            lw = 0.8 + 1.2 * w
            ax.plot(
                [x1, x2],
                [y1, y2],
                color=edge_color,
                alpha=0.25 + min(0.5, 0.08 * w),
                lw=lw,
                zorder=1,
            )

        for t in terms:
            x, y = pos[t]
            size = 500 + 190 * nodes.get(t, 1)
            ax.scatter([x], [y], s=size, color=node_color, edgecolor="white", linewidth=1.2, zorder=2)
            ax.text(x, y, wrap(t, 10), ha="center", va="center", fontsize=11, color=COLORS["ink"], zorder=3)

        ax.set_title(title, fontsize=15, loc="left", pad=14)
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.35, 1.35)
        ax.axis("off")

    fig.suptitle(
        "Field of Visibility (Organizational Structure Case)",
        fontsize=21,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    fig.text(
        0.02,
        0.02,
        "Nodes are salient terms from each answer; edges mark sentence-level co-appearance.\n"
        "RAG field shows denser links to historically specific political vocabulary.",
        fontsize=11,
        color=COLORS["gray"],
    )
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))

    for ext in ["png", "pdf"]:
        out = out_dir / f"fig_field_of_visibility.{ext}"
        ensure_out(out)
        fig.savefig(out, dpi=320, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Figure 3
# ----------------------------
def make_classification_visibility(
    corpus: pd.DataFrame,
    retrieval: pd.DataFrame,
    out_dir: Path,
    corpus_doc_id_col: str,
    retrieval_doc_id_col: str,
    retrieval_chunk_id_col: str | None,
    retrieval_qid_col: str | None,
) -> None:
    n_docs = int(corpus[corpus_doc_id_col].nunique())
    n_retrieved_docs = int(retrieval[retrieval_doc_id_col].nunique())
    n_chunks_visible = int(retrieval[retrieval_chunk_id_col].nunique()) if retrieval_chunk_id_col else int(retrieval.shape[0])
    n_pairs = int(retrieval.shape[0])
    n_queries = int(retrieval[retrieval_qid_col].nunique()) if retrieval_qid_col else 0

    stages = [
        ("Corpus\n(text archive)", n_docs, COLORS["gray"]),
        ("Classified chunks\n(metadata + segmentation)", n_chunks_visible, COLORS["pink"]),
        ("Embedding proximity\n(approximation space)", n_pairs, COLORS["orange"]),
        ("Retrieval selection\n(top-k filtering)", n_retrieved_docs, COLORS["blue"]),
        ("Visible context\nin generation", n_queries, COLORS["red"]),
    ]

    max_v = max(v for _, v, _ in stages) if stages else 1

    fig, ax = plt.subplots(figsize=(17, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    xs = np.linspace(0.08, 0.92, len(stages))
    box_w = 0.15

    for i, ((label, v, c), x) in enumerate(zip(stages, xs)):
        h = 0.18 + 0.34 * (v / max_v) if max_v else 0.18
        y = 0.5 - h / 2
        rect = Rectangle((x - box_w / 2, y), box_w, h, facecolor=c, alpha=0.22, edgecolor=c, linewidth=2.0)
        ax.add_patch(rect)
        ax.text(x, y + h + 0.03, label, ha="center", va="bottom", fontsize=12.5, color=COLORS["ink"], linespacing=1.2)
        ax.text(x, y + h / 2, f"n = {v}", ha="center", va="center", fontsize=13, fontweight="bold", color=COLORS["ink"])

        if i < len(stages) - 1:
            x2 = xs[i + 1] - box_w / 2 - 0.01
            x1 = x + box_w / 2 + 0.01
            arrow = FancyArrowPatch(
                (x1, 0.5),
                (x2, 0.5),
                arrowstyle="-|>",
                mutation_scale=18,
                lw=1.8,
                color=COLORS["ink"],
                alpha=0.8,
            )
            ax.add_patch(arrow)

    ax.set_title("Classification → Visibility: Mechanism of RAG Epistemic Ordering", loc="left", pad=16)
    ax.text(
        0.02,
        0.08,
        "Pasquinelli: classification + approximation (chunking, metadata, embedding neighborhoods)",
        fontsize=11,
        color=COLORS["blue"],
    )
    ax.text(
        0.02,
        0.04,
        "Foucault: ordering determines what becomes visible in discourse (retrieved context constrains generation).",
        fontsize=11,
        color=COLORS["red"],
    )

    plt.tight_layout(pad=2.0)

    for ext in ["png", "pdf"]:
        out = out_dir / f"fig_classification_visibility.{ext}"
        ensure_out(out)
        fig.savefig(out, dpi=320, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Figure 4
# ----------------------------
def make_transformation_diagram(
    analysis: pd.DataFrame,
    corpus: pd.DataFrame,
    out_dir: Path,
    baseline_col: str,
    rag_col: str,
    tradition_col: str | None,
    political_col: str | None,
) -> None:
    trad_lex = set()
    if tradition_col:
        trad_lex |= set(
            re.findall(r"[a-zA-Z\-]+", " ".join(corpus[tradition_col].astype(str).tolist()).lower())
        )
    trad_lex |= {"marxism", "maoism", "feminist", "anarchism", "confederalism", "anti", "imperialist", "cadres"}

    political_lex = set()
    if political_col:
        political_lex |= set(
            re.findall(r"[a-zA-Z\-]+", " ".join(corpus[political_col].astype(str).tolist()).lower())
        )
    political_lex |= {"class", "imperialism", "capitalism", "oppression", "liberation", "repression", "collective"}

    def score_grounding(text: str) -> float:
        toks = tokenize(text)
        if not toks:
            return 0.0
        return sum(t in trad_lex for t in toks) / len(toks)

    def score_political(text: str) -> float:
        toks = tokenize(text)
        if not toks:
            return 0.0
        return sum(t in political_lex for t in toks) / len(toks)

    def score_specificity(text: str) -> float:
        toks = tokenize(text)
        if not toks:
            return 0.0
        long_ratio = sum(len(t) >= 9 for t in toks) / len(toks)
        num_ratio = len(re.findall(r"\d+", str(text))) / max(1, len(toks))
        return long_ratio + 0.6 * num_ratio

    def score_coherence(text: str) -> float:
        sents = split_sentences(text)
        if len(sents) < 2:
            return 0.0
        sims = []
        for a, b in zip(sents, sents[1:]):
            sims.append(cosine_counter(Counter(tokenize(a)), Counter(tokenize(b))))
        return float(np.mean(sims)) if sims else 0.0

    rows = []
    for r in analysis.itertuples(index=False):
        baseline_text = str(getattr(r, baseline_col))
        rag_text = str(getattr(r, rag_col))
        rows.append(
            {
                "baseline_grounding": score_grounding(baseline_text),
                "rag_grounding": score_grounding(rag_text),
                "baseline_political": score_political(baseline_text),
                "rag_political": score_political(rag_text),
                "baseline_specificity": score_specificity(baseline_text),
                "rag_specificity": score_specificity(rag_text),
                "baseline_coherence": score_coherence(baseline_text),
                "rag_coherence": score_coherence(rag_text),
            }
        )

    met = pd.DataFrame(rows)

    pairs = [
        ("Theoretical grounding", *robust_scale(met["baseline_grounding"], met["rag_grounding"])),
        ("Political positioning", *robust_scale(met["baseline_political"], met["rag_political"])),
        ("Specificity", *robust_scale(met["baseline_specificity"], met["rag_specificity"])),
        ("Coherence", *robust_scale(met["baseline_coherence"], met["rag_coherence"])),
    ]

    fig, ax = plt.subplots(figsize=(13.5, 8.5))
    y = np.arange(len(pairs))[::-1]

    for yi, (dim, b, r) in zip(y, pairs):
        ax.plot([b, r], [yi, yi], color=COLORS["light_gray"], lw=4, zorder=1)
        ax.scatter([b], [yi], s=130, color=COLORS["pink"], edgecolor="white", linewidth=1.2, zorder=3)
        ax.scatter([r], [yi], s=130, color=COLORS["blue"], edgecolor="white", linewidth=1.2, zorder=3)
        ax.text(b - 2.2, yi + 0.18, f"{b:.0f}", ha="right", va="center", fontsize=11, color=COLORS["gray"])
        ax.text(r + 2.2, yi + 0.18, f"{r:.0f}", ha="left", va="center", fontsize=11, color=COLORS["gray"])

    ax.set_yticks(y)
    ax.set_yticklabels([p[0] for p in pairs])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Relative score (robustly scaled, median across prompts)")
    ax.set_title("Transformation Diagram: RAG as Epistemic Reconfiguration (not uniform improvement)", loc="left", pad=16)
    ax.grid(axis="x", alpha=0.45)
    ax.spines[["top", "right"]].set_visible(False)

    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["pink"], markeredgecolor="white", markersize=10, label="Baseline"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["blue"], markeredgecolor="white", markersize=10, label="RAG"),
        ],
        frameon=False,
        loc="lower right",
    )

    ax.text(
        0.0,
        -0.18,
        "Interpretation: RAG increases grounding/positioning and specificity, while coherence effects are mixed.",
        transform=ax.transAxes,
        fontsize=11,
        color=COLORS["gray"],
    )

    plt.tight_layout(pad=2.0)

    for ext in ["png", "pdf"]:
        out = out_dir / f"fig_transformation_diagram.{ext}"
        ensure_out(out)
        fig.savefig(out, dpi=320, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    base_dir = Path(__file__).resolve().parent

    corpus_path = base_dir / "corpus" / "corpus_relevant_filled.xlsx"
    retrieval_path = base_dir / "results" / "analysis" / "retrieval_results.csv"
    analysis_path = base_dir / "results" / "analysis" / "full_analysis_ready_deduped.xlsx"
    out_dir = base_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Concise data summary ===")
    print("corpus_path:", corpus_path)
    print("retrieval_path:", retrieval_path)
    print("analysis_path:", analysis_path)

    for p in [corpus_path, retrieval_path, analysis_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    corpus = read_xlsx(corpus_path)
    retrieval = pd.read_csv(retrieval_path)
    analysis = read_xlsx(analysis_path)

    print(f"corpus_relevant_filled.xlsx: {corpus.shape[0]} rows, {corpus.shape[1]} columns")
    print("  columns:", list(corpus.columns))

    print(f"\nretrieval_results.csv: {retrieval.shape[0]} rows, {retrieval.shape[1]} columns")
    print("  columns:", list(retrieval.columns))

    print(f"\nfull_analysis_ready_deduped.xlsx: {analysis.shape[0]} rows, {analysis.shape[1]} columns")
    print("  columns:", list(analysis.columns))

    corpus_doc_id_col = find_column(corpus, ["doc_id", "document_id"])
    title_col = find_column(corpus, ["title", "document_title"], required=False)
    tradition_col = find_column(corpus, ["theoretical_tradition", "tradition", "theoretical tradition"], required=False)
    political_col = find_column(corpus, ["political_positioning", "political position", "political"], required=False)

    retrieval_doc_id_col = find_column(retrieval, ["doc_id", "document_id"])
    retrieval_k_col = find_column(retrieval, ["k"])
    retrieval_chunk_id_col = find_column(retrieval, ["chunk_id", "chunk"], required=False)
    retrieval_qid_col = find_column(retrieval, ["qid", "query_id"], required=False)

    analysis_qid_col = find_column(analysis, ["qid", "query_id"], required=False)
    analysis_query_col = find_column(analysis, ["query", "question"], required=False)
    analysis_baseline_col = find_column(analysis, ["baseline_output", "baseline", "baseline_response"])
    analysis_rag_col = find_column(analysis, ["rag_output", "rag", "rag_response"])

    if tradition_col:
        print("  top theoretical_tradition:", dict(corpus[tradition_col].fillna("n/a").astype(str).value_counts().head(5)))
    if political_col:
        print("  top political_positioning:", dict(corpus[political_col].fillna("n/a").astype(str).value_counts().head(5)))

    if retrieval_qid_col:
        print("  qids:", retrieval[retrieval_qid_col].nunique())
    print("  docs retrieved:", retrieval[retrieval_doc_id_col].nunique())
    print("  k values:", sorted(pd.to_numeric(retrieval[retrieval_k_col], errors="coerce").dropna().unique().tolist()))

    if analysis_query_col:
        print("  sample queries:", analysis[analysis_query_col].head(4).tolist())

    print("\n=== Design choices ===")
    print("1) Retrieval Distribution: ranked lollipop chart with top documents + 'Other'.")
    print("2) Field of Visibility: side-by-side co-occurrence term fields for baseline vs RAG.")
    print("3) Classification → Visibility: staged mechanism panel with corpus/retrieval counts and theoretical bridge annotations.")
    print("4) Transformation Diagram: slope-style comparison across four coded dimensions using text-derived proxies.")

    make_retrieval_distribution(
        retrieval=retrieval,
        corpus=corpus,
        out_dir=out_dir,
        doc_id_col=retrieval_doc_id_col,
        k_col=retrieval_k_col,
        title_col=title_col,
        tradition_col=tradition_col,
    )

    make_field_of_visibility(
        analysis=analysis,
        out_dir=out_dir,
        qid_col=analysis_qid_col or analysis.columns[0],
        baseline_col=analysis_baseline_col,
        rag_col=analysis_rag_col,
    )

    make_classification_visibility(
        corpus=corpus,
        retrieval=retrieval,
        out_dir=out_dir,
        corpus_doc_id_col=corpus_doc_id_col,
        retrieval_doc_id_col=retrieval_doc_id_col,
        retrieval_chunk_id_col=retrieval_chunk_id_col,
        retrieval_qid_col=retrieval_qid_col,
    )

    make_transformation_diagram(
        analysis=analysis,
        corpus=corpus,
        out_dir=out_dir,
        baseline_col=analysis_baseline_col,
        rag_col=analysis_rag_col,
        tradition_col=tradition_col,
        political_col=political_col,
    )

    print("\nSaved files:")
    for name in [
        "fig_retrieval_distribution",
        "fig_field_of_visibility",
        "fig_classification_visibility",
        "fig_transformation_diagram",
    ]:
        for ext in ["png", "pdf"]:
            print(" -", out_dir / f"{name}.{ext}")

    print("\n=== Thesis mapping ===")
    print("1) Retrieval Distribution: demonstrates empirical concentration of retrieval visibility.")
    print("2) Field of Visibility: shows baseline managerial framing vs RAG's denser theoretical-historical field.")
    print("3) Classification → Visibility: formalizes mechanism from corpus ordering to generated discourse visibility.")
    print("4) Transformation Diagram: argues for reconfiguration (gains + trade-offs), not linear quality improvement.")


if __name__ == "__main__":
    main()