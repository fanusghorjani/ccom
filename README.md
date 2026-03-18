# Retrieval-Augmented Generation as an Epistemic Intervention

## Overview

This repository contains the implementation of a Retrieval-Augmented Generation (RAG) pipeline developed as part of a master's thesis.

The project investigates whether retrieval can function as an **epistemic correction mechanism** for large language models. The empirical case focuses on class-conscious organizational management questions within a curated corpus of political and critical theory texts.

The current repository reflects the **implementation and initial results stage**. Evaluation is ongoing.

---

## Research Question

The central research question of this project is:

> To what extent does a RAG pipeline act as an epistemic correction for a Mistral-7B LLM, compared to a no-RAG baseline — on the example of class-conscious organizational management questions?

This project builds on a critical understanding of AI systems as socio-technical infrastructures and evaluates model outputs in terms of their **epistemic structure**, not only correctness.

---

## Theoretical Framing

The thesis is situated at the intersection of:

* Critical Data Studies
* Political Theory
* Applied Natural Language Processing

Large language models are understood as systems that reorganize and reproduce knowledge rather than generating new epistemic insight. Following this perspective, retrieval is conceptualized as a potential intervention that can re-anchor generated outputs in external, theory-grounded knowledge.

---

## Current Project Status

The repository currently includes:

* Implementation of a RAG pipeline (`rag_generate.py`)
* Generated outputs for two experimental conditions:

  * Baseline (no retrieval): `rag_baseline_results.csv`
  * RAG condition: `rag_results.csv`
* Initial qualitative comparison:

  * `comparison_table.md`
  * `comparison_full.md`

The formal evaluation framework is under development.

---

## Experimental Setup

Two experimental conditions are compared:

### 1. Baseline (No Retrieval)

* Direct generation with Mistral-7B
* No access to external documents
* Serves as control condition

### 2. RAG (Retrieval-Augmented Generation)

* Retrieval of relevant text chunks from a curated corpus
* Retrieved context is injected into the prompt
* Model generates grounded responses

All other parameters are held constant across both conditions.

---

## Data

The dataset consists of a curated selection of texts related to class-conscious organizational management and critical theory.

The corpus includes foundational works from:

* Marxist theory
* Feminist theory
* Anti-colonial and revolutionary theory

The texts are preprocessed and segmented into retrievable chunks used by the RAG pipeline.

*Note: The dataset is purpose-built and not intended to be exhaustive.* 

---

## Pipeline Overview

The current pipeline consists of:

1. Text preprocessing (PDF → clean text)
2. Chunking of documents
3. Embedding-based retrieval
4. Answer generation using retrieved context

---

## Evaluation (Work in Progress)

Evaluation is currently ongoing and follows a mixed approach:

* **Qualitative assessment (primary method)**

  * Grounding in source texts
  * Use of domain-specific concepts
  * Coherence and relevance

* **Simple automated signal (secondary)**

  * Classification of outputs as in-domain vs. out-of-domain

At this stage, the repository includes preliminary comparison files:

* `comparison_table.md`
* `comparison_full.md`

Final evaluation criteria and interpretation will be developed in the thesis.

---

## Reproducibility

To run the generation pipeline:

```bash
python rag_generate.py
```

Environment setup is defined in the `Dockerfile`.

---

## Notes

* Cache files and model weights are excluded via `.gitignore`
* Outputs are stored as CSV files for transparency
* The repository reflects an intermediate research stage

---

## Author

Fanus Ghorjani
Master’s Thesis, Hertie School

