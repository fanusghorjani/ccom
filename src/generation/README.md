# Generation

This module handles the generation of model outputs for the two experimental conditions used in this study.

## Experimental Conditions

The generation process consists of two controlled conditions:

1. **Baseline**

   * The model generates responses without external context.
   * Outputs rely solely on parametric model knowledge.

2. **RAG (Retrieval-Augmented Generation)**

   * Relevant text chunks are retrieved from a curated corpus.
   * Retrieved content is injected into the prompt before generation.
   * Introduces non-parametric memory into the generation process.

Both conditions use identical model parameters.
The only experimental difference is the inclusion of retrieved context.


## Scripts

* `01_generate_baseline.py`
  Generates responses without retrieval.

* `02_generate_rag.py`
  Generates responses with retrieved context.

## Input

* `data/evaluation_questions.csv`
  Contains the 30 evaluation queries.

* (RAG only) retrieved chunks from `src/retrieval/`

## Output

* `results/baseline_outputs.csv`
* `results/rag_outputs.csv`

Each output file contains model responses for all evaluation questions.

## Usage

Run from project root:

```bash
python src/generation/01_generate_baseline.py
python src/generation/02_generate_rag.py
```

Alternatively, run the full pipeline:

```bash
python run_pipeline.py
```

## Notes

* Outputs are generated under controlled conditions to allow direct comparison.
* Due to the probabilistic nature of LLMs, results should be interpreted as indicative patterns rather than fixed outcomes.
* This module is part of a proof-of-concept implementation for analyzing epistemic differences between baseline and retrieval-augmented generation.
