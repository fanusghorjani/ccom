# Queries

This directory contains the evaluation queries used to test the model under both baseline and retrieval-augmented (RAG) conditions.

The queries form a structured evaluation set designed to probe how the model responds to questions related to class-conscious organizing.

---

## File

* `eval_queries.jsonl`
  Contains all evaluation queries in JSONL format.

Each entry follows the structure:

```json
{
  "qid": "q01",
  "query": "How should a revolutionary organization be structured?"
}
```

---

## Purpose

The queries define the input to the system and structure the evaluation of generated outputs.

They are used for:

* retrieval (selecting relevant chunks)
* generation (baseline vs. RAG comparison)
* qualitative and quantitative analysis

## Thematic Structure

The evaluation set consists of 30 queries organized into six thematic dimensions:

### I. Organizational Structure

Focus on internal organization, leadership, and decision-making.

### II. Discipline & Internal Culture

Focus on internal norms, accountability, and collective practice.

### III. Anti-Oppression & Internal Power

Focus on internal hierarchies, discrimination, and power relations.

### IV. Theory & Practice

Focus on political education, learning, and the relationship between theory and action.

### V. Mass Line & Social Base

Focus on the relationship between organizations and broader social groups.

### VI. Power, Strategy, Repression & Social Reproduction

Focus on strategy, repression, material conditions, and global struggle.

## Notes

* The queries are not randomly generated but manually constructed as part of the experimental design.
* They reflect key dimensions of class-conscious organizing and are aligned with the analytical framework used in the thesis.
* The thematic structure is used for interpretation and analysis but is not encoded in the JSONL file itself.
* All queries are used consistently across baseline and RAG conditions, ensuring comparability.

## Role in the System

The queries serve as the starting point of the pipeline:

```text
query → retrieval → context → generation → output → analysis
```

They therefore play a central role in determining which knowledge becomes visible and how the model responds under different conditions.
