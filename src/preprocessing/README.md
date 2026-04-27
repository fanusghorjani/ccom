# Preprocessing

This module prepares the raw corpus for retrieval. It cleans extracted text and segments documents into overlapping chunks suitable for embedding.

## Overview

```text
raw_txt → cleaned text → paragraph-aware chunks (jsonl)
```

## Scripts

### `clean_texts.py`

Normalizes raw text extracted from PDFs.

**Input:**

* `corpus/raw_txt/*.txt`

**Output:**

* `corpus/clean_txt/*.txt`

Operations:

* repair common mojibake patterns (e.g. `â€™` → `’`)
* normalize line endings
* rejoin words split across hyphenated line breaks
* collapse repeated whitespace
* drop standalone page-number lines
* collapse excessive blank lines

### `chunk_texts.py`

Splits cleaned documents into chunks for retrieval.

**Input:**

* `corpus/clean_txt/*.txt`

**Output:**

* `processed/chunks/chunks.jsonl` — one JSON record per chunk with fields `doc_id`, `chunk_id`, `text`, `word_count`.

Chunking parameters (defined at the top of the script):

* `TARGET_WORDS = 400` — target chunk size
* `MIN_WORDS = 200` — small tail chunks are merged into the previous chunk
* `MAX_WORDS = 520` — hard cap to prevent oversized chunks
* `OVERLAP_WORDS = 60` — token overlap between adjacent chunks

The chunker is paragraph-aware (it tries to keep paragraphs intact) and applies heuristics to strip table-of-contents blocks and dot-leader TOC lines.

## Usage

Run from project root:

```bash
python src/preprocessing/clean_texts.py
python src/preprocessing/chunk_texts.py
```
