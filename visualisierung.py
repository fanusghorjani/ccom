import re
from collections import Counter
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# ----------------------------
# 1. Strong filtering config
# ----------------------------

# Generic stopwords + domain-generic words that made your current plot weak
BASE_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "are", "was", "were", "from", "into",
    "their", "there", "they", "them", "have", "has", "had", "will", "would", "should",
    "could", "can", "may", "might", "must", "also", "such", "more", "most", "less",
    "than", "then", "when", "where", "what", "which", "while", "because", "through",
    "within", "between", "about", "into", "onto", "over", "under", "after", "before",
    "being", "been", "very", "much", "many", "each", "some", "same", "both", "other",
    "these", "those", "whose", "your", "our", "its", "it's", "itself", "themselves",
    "himself", "herself", "ourselves", "myself", "yourself", "yourselves",
    "not", "only", "just", "even", "still", "well", "rather", "quite", "however",
    "therefore", "thus", "indeed", "yet",

    # weak content words that polluted your graph
    "help", "made", "make", "making", "clear", "achieved", "ensure", "approach",
    "information", "relatively", "important", "use", "used", "using", "way", "ways",
    "good", "better", "best", "need", "needs", "needed", "work", "working", "works",
    "form", "forms", "role", "roles", "question", "answer", "responses", "response",

    # generic organizational words that dominate too easily
    "organization", "organizations", "organizational", "structure", "structured",
    "group", "groups", "movement", "movements", "leadership", "leaders", "member",
    "members", "decision", "decisions", "decision-making", "power", "process",
    "processes", "responsibility", "responsibilities", "support", "people", "social",
}

# Keep these if they show up as meaningful items
WHITELIST_TERMS = {
    "mass line",
    "class consciousness",
    "political education",
    "democratic centralism",
    "central committee",
    "professional revolutionaries",
    "cadre organization",
    "collective leadership",
    "historical struggle",
    "revolutionary practice",
    "anti oppression",
    "oppression",
    "class struggle",
    "collective discipline",
    "accountability",
    "centralization",
    "decentralization",
    "coordination",
    "efficiency",
    "collective participation",
}

# Extra domain phrases worth detecting as units
PHRASE_PATTERNS = [
    r"\bmass line\b",
    r"\bclass consciousness\b",
    r"\bpolitical education\b",
    r"\bdemocratic centralism\b",
    r"\bcentral committee\b",
    r"\bprofessional revolutionaries\b",
    r"\bclass struggle\b",
    r"\brevolutionary practice\b",
    r"\bcollective leadership\b",
    r"\bcollective participation\b",
    r"\banti[- ]oppression\b",
    r"\bpower imbalance[s]?\b",
    r"\binternal accountability\b",
    r"\bsecurity culture\b",
    r"\bpolitical line\b",
    r"\bcadre organization\b",
    r"\borganizational discipline\b",
]


# ----------------------------
# 2. Text normalization
# ----------------------------

def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_phrases(text: str, phrase_patterns: Iterable[str]) -> List[str]:
    text = normalize_text(text)
    found = []
    for pattern in phrase_patterns:
        found.extend(re.findall(pattern, text))
    return found


def tokenize_clean(text: str) -> List[str]:
    text = normalize_text(text)
    tokens = re.findall(r"[a-z][a-z\-]{2,}", text)

    cleaned = []
    for tok in tokens:
        tok = tok.strip("-")
        if not tok or tok in BASE_STOPWORDS:
            continue
        cleaned.append(tok)
    return cleaned


# ----------------------------
# 3. Candidate extraction
# ----------------------------

def extract_candidates(text: str) -> List[str]:
    """
    Mix of:
    - curated multiword phrases
    - cleaned unigram tokens
    """
    phrases = extract_phrases(text, PHRASE_PATTERNS)
    tokens = tokenize_clean(text)
    return phrases + tokens


def aggregate_term_counts(texts: Iterable[str]) -> Counter:
    counter = Counter()
    for text in texts:
        counter.update(extract_candidates(text))
    return counter


# ----------------------------
# 4. Contrastive scoring
# ----------------------------

def tfidf_contrast_terms(
    baseline_texts: List[str],
    rag_texts: List[str],
    top_n: int = 12,
    min_doc_freq: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      baseline_terms_df, rag_terms_df
    with contrastive TF-IDF scores.
    """
    all_texts = [normalize_text(t) for t in baseline_texts + rag_texts]
    labels = ["baseline"] * len(baseline_texts) + ["rag"] * len(rag_texts)

    # custom tokenizer using our filtering
    def analyzer(doc: str) -> List[str]:
        return extract_candidates(doc)

    vec = TfidfVectorizer(
        analyzer=analyzer,
        lowercase=False,
        min_df=min_doc_freq,
        max_df=0.9,
        ngram_range=(1, 1),  # phrases are already injected by extract_candidates
    )

    X = vec.fit_transform(all_texts)
    terms = np.array(vec.get_feature_names_out())

    baseline_mask = np.array([lab == "baseline" for lab in labels])
    rag_mask = np.array([lab == "rag" for lab in labels])

    baseline_mean = np.asarray(X[baseline_mask].mean(axis=0)).ravel()
    rag_mean = np.asarray(X[rag_mask].mean(axis=0)).ravel()

    diff_baseline = baseline_mean - rag_mean
    diff_rag = rag_mean - baseline_mean

    baseline_df = pd.DataFrame({
        "term": terms,
        "baseline_mean": baseline_mean,
        "rag_mean": rag_mean,
        "contrast": diff_baseline,
    }).sort_values("contrast", ascending=False)

    rag_df = pd.DataFrame({
        "term": terms,
        "baseline_mean": baseline_mean,
        "rag_mean": rag_mean,
        "contrast": diff_rag,
    }).sort_values("contrast", ascending=False)

    # final semantic filtering
    def keep_term(term: str) -> bool:
        if term in WHITELIST_TERMS:
            return True
        if term in BASE_STOPWORDS:
            return False
        if len(term) < 4:
            return False
        if term.isdigit():
            return False
        return True

    baseline_df = baseline_df[baseline_df["term"].map(keep_term)].head(top_n).reset_index(drop=True)
    rag_df = rag_df[rag_df["term"].map(keep_term)].head(top_n).reset_index(drop=True)

    return baseline_df, rag_df


# ----------------------------
# 5. Case-specific extraction
# ----------------------------

def get_case_texts(
    analysis_df: pd.DataFrame,
    qid: str | None = None,
    query_contains: str | None = None,
    baseline_col: str = "baseline_output",
    rag_col: str = "rag_output",
    qid_col: str = "qid",
    query_col: str = "query",
) -> Tuple[List[str], List[str]]:
    """
    Pull texts for one case or for matched rows.
    """
    df = analysis_df.copy()

    if qid is not None and qid_col in df.columns:
        df = df[df[qid_col].astype(str).str.lower() == qid.lower()]

    if query_contains is not None and query_col in df.columns:
        needle = query_contains.lower()
        df = df[df[query_col].astype(str).str.lower().str.contains(needle, na=False)]

    if df.empty:
        raise ValueError("No matching rows found for the requested case.")

    baseline_texts = df[baseline_col].fillna("").astype(str).tolist()
    rag_texts = df[rag_col].fillna("").astype(str).tolist()
    return baseline_texts, rag_texts


# ----------------------------
# 6. Pretty final selection
# ----------------------------

def dedupe_semantic_near_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove trivial duplicates like 'decision' vs 'decisions', or phrase covered by parts.
    Simple heuristic, safe enough for plotting.
    """
    chosen = []
    seen_roots = set()

    def rootish(term: str) -> str:
        t = term.lower().strip()
        t = re.sub(r"\b(s|es|ing|ed)\b", "", t)
        t = t.replace("-", " ")
        return t

    for _, row in df.iterrows():
        term = row["term"]
        root = rootish(term)
        if root in seen_roots:
            continue
        seen_roots.add(root)
        chosen.append(row)

    if not chosen:
        return df.head(0)

    return pd.DataFrame(chosen).reset_index(drop=True)


def get_clean_contrast_terms_for_plot(
    analysis_df: pd.DataFrame,
    qid: str = "q01",
    query_contains: str | None = "revolutionary organization be structured",
    top_n: int = 8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Final function to call for your figure.
    """
    baseline_texts, rag_texts = get_case_texts(
        analysis_df,
        qid=qid,
        query_contains=query_contains,
        baseline_col="baseline_output",
        rag_col="rag_output",
        qid_col="qid",
        query_col="query",
    )

    baseline_df, rag_df = tfidf_contrast_terms(
        baseline_texts=baseline_texts,
        rag_texts=rag_texts,
        top_n=top_n * 3,   # overgenerate first
        min_doc_freq=1,
    )

    baseline_df = dedupe_semantic_near_duplicates(baseline_df).head(top_n)
    rag_df = dedupe_semantic_near_duplicates(rag_df).head(top_n)

    return baseline_df, rag_df


# ----------------------------
# 7. Example usage
# ----------------------------

import pandas as pd

analysis = pd.read_excel(
    r"C:\Users\SinaElBasiouni\Documents\3_Semester\Thesis\data\class_conscious_organization_files\results\analysis\full_analysis_ready_deduped.xlsx"
)

# For q01 if that is:
# "How should a revolutionary organization be structured?"
baseline_terms_df, rag_terms_df = get_clean_contrast_terms_for_plot(
    analysis_df=analysis,
    qid="q01",
    query_contains="revolutionary organization be structured",
    top_n=8,
)

print("\nBaseline terms for plot:")
print(baseline_terms_df[["term", "contrast"]])

print("\nRAG terms for plot:")
print(rag_terms_df[["term", "contrast"]])