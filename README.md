# Retrieval-Augmented Generation as an Epistemic Intervention

Case study repository for a master's thesis on **class-conscious organizational management** and the role of **retrieval-augmented generation (RAG)** as an epistemic intervention in language-model outputs.

## What this repo contains

This repository includes:

- a curated political/theoretical text corpus,
- preprocessing and chunking scripts,
- dense retrieval infrastructure (Sentence Transformers + FAISS),
- controlled generation scripts for baseline vs. RAG,
- analysis outputs, figures, and evaluation artifacts.

In short, this is an end-to-end research pipeline for comparing:

1. **Baseline generation** (no external retrieval context), and
2. **RAG generation** (retrieved corpus chunks injected into prompts).

---

## Research question

> To what extent does retrieval-augmented generation function as a form of epistemic correction by reshaping the structure of knowledge in LLM outputs compared to a baseline without retrieval?

The project treats retrieval not only as a technical component, but as a mechanism that structures visibility and therefore influences what is sayable, grounded, and comparable in model responses.

---

## Repository structure

```text
.
├── corpus/
│   ├── raw_pdfs/            # Source PDFs
│   ├── raw_txt/             # Extracted text before cleaning
│   ├── clean_txt/           # Cleaned text used for chunking
│   └── references.bib       # Corpus bibliography
├── data/
│   └── queries/
│       └── eval_queries.jsonl
├── processed/
│   └── chunks/
│       ├── chunks.jsonl
│       ├── embeddings.npy
│       ├── embeddings_meta.jsonl
│       └── faiss.index
├── results/
│   ├── generations/
│   ├── analysis/
│   └── evaluation/
├── src/
│   ├── preprocessing/
│   ├── retrieval/
│   ├── generation/
│   └── analysis/
└── run_pipeline.py
```

---

## Method overview

Pipeline flow:

```text
corpus → cleaning → chunking → embedding → indexing → retrieval → generation → analysis
```

### Conditions

- **Condition 1 (Baseline):** answer questions without retrieved context.
- **Condition 2 (RAG):** retrieve top-k chunks and inject them into prompt context.

### Fixed generation settings

Both conditions keep comparable defaults:

- `temperature = 0.2`
- `top_p = 0.9`
- `max_new_tokens = 350`
- `seed = 42`

### Retriever settings

- Embedding model: `sentence-transformers/all-mpnet-base-v2`
- Similarity backend: FAISS `IndexFlatIP` (cosine-equivalent with normalized vectors)

---

## Quickstart

### 1) Create environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Ensure generation backend is available

Generation scripts use an Ollama-compatible API by default:

- default base URL: `http://127.0.0.1:11434`
- default model: `mistral:latest`

You can override with environment variables:

```bash
export OLLAMA_BASE_URL=http://127.0.0.1:11434
export OLLAMA_MODEL=mistral:latest
```

### 3) Run full pipeline

```bash
python run_pipeline.py
```

---

## Running stages manually

### Preprocessing

```bash
python src/preprocessing/clean_texts.py
python src/preprocessing/chunk_texts.py
```

### Retrieval index build

```bash
python src/retrieval/embed_chunks.py
python src/retrieval/build_faiss.py
```

### Retrieval evaluation (optional)

```bash
python src/retrieval/eval_retrieval.py
python src/retrieval/analyze_retrieval.py
```

### Generation

```bash
python src/generation/baseline_generate.py
python src/generation/rag_generate.py
```

---

## Key inputs and outputs

### Inputs

- Corpus texts: `corpus/raw_txt/` and `corpus/raw_pdfs/`
- Evaluation queries: `data/queries/eval_queries.jsonl`

### Intermediate artifacts

- `processed/chunks/chunks.jsonl`
- `processed/chunks/embeddings.npy`
- `processed/chunks/embeddings_meta.jsonl`
- `processed/chunks/faiss.index`

### Main outputs

- `baseline_results.csv`
- `rag_results.csv`
- `retrieval_results.csv`
- additional analysis/evaluation files in `results/`

---

## Reproducibility notes

- Retrieval and generation scripts use fixed defaults to improve run-to-run comparability.
- Results can still vary across machines/accelerators/model versions.
- This repository contains pre-generated artifacts (`processed/`, parts of `results/`) to support immediate inspection.

---

## Related documentation

- `corpus/README.md` — corpus curation and role
- `src/preprocessing/README.md` — cleaning/chunking details
- `src/retrieval/README.md` — embedding/retrieval details
- `src/generation/README.md` — baseline vs. RAG generation details
- `data/queries/README.md` — evaluation query set rationale

---

## Citation / usage

If you use this repository in academic work, please cite the associated thesis/project and clearly distinguish between:

1. model-internal (parametric) knowledge, and
2. externally retrieved corpus-grounded context.

