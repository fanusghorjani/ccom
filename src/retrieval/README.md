# Retrieval

This module implements the retrieval pipeline used in the Retrieval-Augmented Generation (RAG) condition.

It encodes text chunks into vector representations, builds a similarity index, and retrieves the most relevant chunks for a given query.


## Overview

The retrieval process consists of the following steps:

```text
chunks → embeddings → FAISS index → query → top-k chunks
```


## Scripts

### `embed_chunks.py`

Encodes all text chunks into dense vector embeddings using a pretrained sentence transformer model.

**Input:**

* `processed/chunks/chunks.jsonl`

**Output:**

* `processed/chunks/embeddings.npy`
* `processed/chunks/embeddings_meta.jsonl`

The embeddings are normalized, enabling cosine similarity search via inner product.


### `build_faiss.py`

Builds a similarity search index over the embeddings using FAISS.

**Input:**

* `processed/chunks/embeddings.npy`

**Output:**

* `processed/chunks/faiss.index`

The index uses inner product similarity (equivalent to cosine similarity for normalized embeddings).


### `eval_retrieval.py`

Runs retrieval for a set of evaluation queries and stores the results.

**Input:**

* `processed/chunks/faiss.index`
* `processed/chunks/embeddings_meta.jsonl`
* `processed/chunks/chunks.jsonl`
* `data/queries/eval_queries.jsonl`

**Output:**

* `retrieval_results.csv`

For each query, the script retrieves the top-k most relevant chunks (k = 5, 10) and records:

* similarity score
* document ID
* chunk ID
* rank

This script implements the main retrieval process used in the analysis.


### `test_retrieval.py`

Runs retrieval for a single query and prints the top results.

Used for debugging and qualitative inspection of retrieved chunks.

### `analyze_retrieval.py`

Analyzes retrieval results to identify structural patterns.

Examples of analysis:

* document dominance (which texts are retrieved most often)
* concentration of results across documents
* distribution of retrieved sources

This supports the evaluation of retrieval bias and visibility effects.

## Retrieval Logic

Retrieval is performed using dense vector similarity:

1. Queries are encoded into embeddings
2. The FAISS index is queried
3. Top-k most similar chunks are returned

```python
scores, indices = index.search(query_vector, k)
```

This operation determines which parts of the corpus become visible to the model.

## Notes

* Retrieval is implemented within `eval_retrieval.py` and `test_retrieval.py` rather than as a separate function.
* The system uses dense retrieval based on semantic similarity rather than keyword matching.
* The retrieved chunks define the context available to the model in the RAG condition.

From an epistemic perspective, retrieval is the central mechanism that structures visibility: it determines which knowledge becomes accessible and how different sources are represented in generated outputs.
