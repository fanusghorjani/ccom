#!/usr/bin/env python3
"""Generate Condition 2 (RAG) answers for eval queries.

This script retrieves top-k chunks from a FAISS index and injects them into a
fixed prompt template before generation. It is designed to be comparable with
Condition 1 baseline outputs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import random
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# Frozen experiment parameters.
# Keep these constant for reproducibility across Condition 2 runs.
# -----------------------------
SYSTEM_PROMPT = "You are a helpful assistant. Answer clearly and concretely."
TEMPERATURE = 0.2
TOP_P = 0.9
MAX_NEW_TOKENS = 350
SEED = 42
TOP_K = 5
RETRIEVER_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

DEFAULT_INPUT = Path("eval_queries.jsonl")
DEFAULT_OUTPUT = Path("rag_results.csv")
DEFAULT_ERRORS = Path("rag_errors.jsonl")
DEFAULT_INDEX = Path("chunks") / "faiss.index"
DEFAULT_META = Path("chunks") / "embeddings_meta.jsonl"
DEFAULT_CHUNKS = Path("chunks") / "chunks.jsonl"


@dataclass
class GenSettings:
    temperature: float = TEMPERATURE
    top_p: float = TOP_P
    max_new_tokens: int = MAX_NEW_TOKENS
    seed: int = SEED


@dataclass
class RetrievedChunk:
    rank: int
    score: float
    doc_id: str
    chunk_id: str
    text: str


class GenerationBackend:
    def model_name(self) -> str:
        raise NotImplementedError

    def generate(self, user_content: str, settings: GenSettings) -> str:
        raise NotImplementedError


class OllamaBackend(GenerationBackend):
    """Backend for local/remote Ollama server (`ollama serve`)."""

    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def model_name(self) -> str:
        return self.model

    def generate(self, user_content: str, settings: GenSettings) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "options": {
                "temperature": settings.temperature,
                "top_p": settings.top_p,
                "num_predict": settings.max_new_tokens,
                "seed": settings.seed,
            },
        }
        data = _post_json(f"{self.base_url}/api/chat", payload)
        return data["message"]["content"].strip()


def _post_json(url: str, payload: Dict) -> Dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_queries(path: Path) -> List[Dict[str, str]]:
    queries = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "qid" not in obj or "query" not in obj:
                raise ValueError(f"Invalid input at line {line_no}: expected qid/query")
            queries.append({"qid": str(obj["qid"]), "query": str(obj["query"])})
    return queries


def load_meta(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_chunk_texts(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f if line.strip()]


def retrieve_topk(
    query: str,
    top_k: int,
    index: faiss.Index,
    retriever: SentenceTransformer,
    meta: Sequence[Dict[str, str]],
    chunk_texts: Sequence[str],
) -> List[RetrievedChunk]:
    q_vec = retriever.encode([query], normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(q_vec, top_k)

    out: List[RetrievedChunk] = []
    for rank, (idx, score) in enumerate(zip(idxs[0], scores[0]), start=1):
        i = int(idx)
        if i < 0 or i >= len(meta):
            continue
        m = meta[i]
        out.append(
            RetrievedChunk(
                rank=rank,
                score=float(score),
                doc_id=str(m.get("doc_id", "")),
                chunk_id=str(m.get("chunk_id", "")),
                text=str(chunk_texts[i]),
            )
        )
    return out


def build_rag_user_prompt(question: str, retrieved: Sequence[RetrievedChunk]) -> str:
    # Prompt format is constant for Condition 2:
    # - includes question
    # - includes retrieved context blocks in rank order
    # - includes a fixed instruction on how to use the context
    lines = [
        "Use the retrieved context to answer the question.",
        "If the context is insufficient, say what is missing.",
        "",
        f"Question: {question}",
        "",
        "Retrieved context:",
    ]

    for c in retrieved:
        lines.append(f"[{c.rank}] doc_id={c.doc_id} chunk_id={c.chunk_id} score={c.score:.6f}")
        lines.append(c.text)
        lines.append("")

    lines.append("Answer:")
    return "\n".join(lines)


def iter_generate_rag(
    backend: GenerationBackend,
    queries: Iterable[Dict[str, str]],
    settings: GenSettings,
    top_k: int,
    index: faiss.Index,
    retriever: SentenceTransformer,
    meta: Sequence[Dict[str, str]],
    chunk_texts: Sequence[str],
    retries: int,
    retry_delay_sec: float,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    rows: List[Dict[str, str]] = []
    errors: List[Dict[str, str]] = []

    for item in queries:
        qid = item["qid"]
        question = item["query"]
        timestamp_utc = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()

        retrieved = retrieve_topk(
            query=question,
            top_k=top_k,
            index=index,
            retriever=retriever,
            meta=meta,
            chunk_texts=chunk_texts,
        )
        user_prompt = build_rag_user_prompt(question=question, retrieved=retrieved)

        answer = None
        last_error = None
        for attempt in range(1, retries + 2):
            try:
                answer = backend.generate(user_content=user_prompt, settings=settings)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt <= retries:
                    jitter = random.uniform(0, 0.25)
                    time.sleep(retry_delay_sec + jitter)

        if answer is None:
            errors.append(
                {
                    "qid": qid,
                    "question": question,
                    "error": last_error or "unknown error",
                    "top_k": top_k,
                    "timestamp_utc": timestamp_utc,
                }
            )
            continue

        rows.append(
            {
                "qid": qid,
                "question": question,
                "answer": answer,
                "model_name": backend.model_name(),
                "temperature": settings.temperature,
                "top_p": settings.top_p,
                "max_new_tokens": settings.max_new_tokens,
                "seed": settings.seed,
                "top_k": top_k,
                "retriever_model": RETRIEVER_MODEL_NAME,
                "retrieved_doc_ids": "|".join(c.doc_id for c in retrieved),
                "retrieved_chunk_ids": "|".join(c.chunk_id for c in retrieved),
                "retrieved_scores": "|".join(f"{c.score:.6f}" for c in retrieved),
                "timestamp_utc": timestamp_utc,
            }
        )

    return rows, errors


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    fieldnames = [
        "qid",
        "question",
        "answer",
        "model_name",
        "temperature",
        "top_p",
        "max_new_tokens",
        "seed",
        "top_k",
        "retriever_model",
        "retrieved_doc_ids",
        "retrieved_chunk_ids",
        "retrieved_scores",
        "timestamp_utc",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_errors_jsonl(path: Path, errors: List[Dict[str, str]]) -> None:
    if not errors:
        if path.exists():
            path.unlink()
        return
    with path.open("w", encoding="utf-8") as f:
        for err in errors:
            f.write(json.dumps(err, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Condition 2 RAG answers.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to eval_queries.jsonl")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to output CSV")
    p.add_argument("--errors", type=Path, default=DEFAULT_ERRORS, help="Path to JSONL error log")
    p.add_argument("--index", type=Path, default=DEFAULT_INDEX, help="Path to FAISS index")
    p.add_argument("--meta", type=Path, default=DEFAULT_META, help="Path to embeddings metadata")
    p.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS, help="Path to chunks jsonl")
    p.add_argument("--top-k", type=int, default=TOP_K, help="Top-k chunks to retrieve")
    p.add_argument("--backend", choices=["ollama"], default="ollama", help="LLM backend")
    p.add_argument("--model", default=None, help="Ollama model name")
    p.add_argument("--base-url", default=None, help="Override Ollama base URL")
    p.add_argument("--retries", type=int, default=2, help="Retries after first failed request")
    p.add_argument("--retry-delay", type=float, default=1.5, help="Base retry delay in seconds")
    return p.parse_args()


def build_backend(args: argparse.Namespace) -> GenerationBackend:
    model = args.model or os.getenv("OLLAMA_MODEL", "mistral:latest")
    base_url = args.base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    return OllamaBackend(model=model, base_url=base_url)


def validate_inputs(args: argparse.Namespace) -> None:
    for p in [args.input, args.index, args.meta, args.chunks]:
        if not p.exists():
            raise SystemExit(f"Missing required file: {p.resolve()}")
    if args.top_k < 1:
        raise SystemExit("--top-k must be >= 1")


def main() -> None:
    args = parse_args()
    validate_inputs(args)

    settings = GenSettings()
    backend = build_backend(args)

    queries = load_queries(args.input)
    index = faiss.read_index(str(args.index))
    meta = load_meta(args.meta)
    chunk_texts = load_chunk_texts(args.chunks)
    retriever = SentenceTransformer(RETRIEVER_MODEL_NAME)

    rows, errors = iter_generate_rag(
        backend=backend,
        queries=queries,
        settings=settings,
        top_k=args.top_k,
        index=index,
        retriever=retriever,
        meta=meta,
        chunk_texts=chunk_texts,
        retries=args.retries,
        retry_delay_sec=args.retry_delay,
    )

    write_csv(args.output, rows)
    write_errors_jsonl(args.errors, errors)

    print(f"Processed questions: {len(rows)} / {len(queries)}")
    print(f"Output CSV: {args.output}")
    print(f"Failures: {len(errors)}")
    if errors:
        print(f"Error log: {args.errors}")


if __name__ == "__main__":
    main()
