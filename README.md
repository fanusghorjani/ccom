# Retrieval-Augmented Generation as an Epistemic Intervention
### A Case Study on Class-Conscious Organizational Management

This repository contains the full experimental pipeline and analysis for a master's thesis investigating retrieval-augmented generation (RAG) as an epistemic intervention in large language models (LLMs).

---

## Research Focus

Large Language Models do not only retrieve information — they actively structure and produce knowledge. Their outputs are shaped by probabilistic inference, data bias, and infrastructural constraints.

This project introduces the concept of **Epistemic Correction**, defined as:

> an intervention into the structure of knowledge production in LLMs by reorganizing visibility, relational positioning, and comparability through curated retrieval.

The central research question is:

> To what extent does retrieval-augmented generation function as a form of epistemic correction by reshaping the structure of knowledge in LLM outputs compared to a baseline without retrieval?

---

## Experimental Design

The study follows a **controlled comparative setup**:

- Same model: **Mistral-7B**
- Two conditions:
  - Baseline (no retrieval)
  - RAG (retrieval-augmented generation)
- Same prompts, parameters, and evaluation set
- Only difference: **injected retrieved context**

---

## Evaluation Setup

- **30 evaluation questions**
- Covering practical and theoretical aspects of class-conscious organizing
- Structured across multiple thematic domains

### Analytical Dimensions

Outputs are evaluated along four epistemic dimensions:

1. Theoretical grounding
2. Political positioning
3. Specificity
4. Epistemic coherence

Each dimension is coded using a simple ordinal scheme:

- 0 = absent
- 1 = partial
- 2 = strong presence

---

## Dataset

- **49 curated documents**
- Focus: class-conscious organization and critical theory
- Includes Marxist and Black radical traditions

The dataset is:

- not representative
- deliberately constructed
- a **situated epistemic intervention**

> The dataset itself functions as an epistemic interface that shapes what becomes visible and articulable within the model.

---

## Pipeline

The full pipeline is implemented in this repository:

```text
corpus → preprocessing → chunking → embeddings → retrieval → generation → analysis → evaluation
```
