# Corpus Documentation

## Overview
This directory contains the textual corpus used for the Retrieval-Augmented Generation (RAG) pipeline implemented in this project.

The corpus serves as the **epistemic foundation** of the system: it defines the external knowledge base from which context is retrieved and injected into the language model during generation.

---

## Composition of the Corpus
The corpus consists of a curated selection of texts related to:

* Class-conscious organizational practice
* Marxist theory
* Feminist theory
* Anti-colonial and revolutionary thought

These texts were selected to reflect a specific theoretical and political perspective relevant to the research question.

---

## Data Structure

* `references.bib`
  Bibliographic metadata for all texts included in the corpus.

---

## Selection Criteria
The corpus is **not exhaustive**, but intentionally constructed.
Selection was guided by:

* Relevance to class-conscious organizing practices
* Theoretical depth and conceptual clarity
* Representation of critical perspectives on organization, power, and social change

This curated approach reflects the assumption that retrieval systems do not operate on neutral data, but on **structured and situated knowledge sources**.

---

## Epistemic Role in the Project
Within this project, the corpus is not treated as a passive dataset.
Instead, it functions as:

* a **source of theoretical grounding**
* a **constraint on model output**
* a **mechanism of epistemic intervention** in the generation process

The RAG pipeline retrieves fragments from this corpus to influence how answers are constructed, thereby shaping the epistemic structure of the model’s responses.

---

## Notes
* The corpus was preprocessed (cleaning, chunking) before being used in the retrieval pipeline.
* Only derived representations (chunks, embeddings) are used during model inference.
* The `.bib` file provides transparency regarding the sources and enables traceability.

---
