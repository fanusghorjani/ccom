# Retrieval-Augmented Generation as an Epistemic Intervention  
## A Case Study on Class-Conscious Organizational Management

Case study repository for a master's thesis on **class-conscious organizational management** and the role of **retrieval-augmented generation (RAG)** as an epistemic intervention in language-model outputs.

---

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

## Method overview

Pipeline flow:

corpus → cleaning → chunking → embedding → indexing → retrieval → generation → analysis

### Conditions

- **Condition 1 (Baseline):** answer questions without retrieved context  
- **Condition 2 (RAG):** retrieve top-k chunks and inject them into prompt context  

### Fixed generation settings

Both conditions use identical parameters:

- temperature = 0.2  
- top_p = 0.9  
- max_new_tokens = 350  
- seed = 42  

### Retriever settings

- Embedding model: sentence-transformers/all-mpnet-base-v2  
- Similarity backend: FAISS IndexFlatIP  
- Similarity metric: cosine similarity (via normalized vectors)  

---

## Quickstart

### 1) Create environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt