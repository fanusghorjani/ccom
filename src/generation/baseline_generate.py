#!/usr/bin/env python3
"""Generate Condition 1 (baseline, no retrieval) answers for eval queries.

This script intentionally sends ONLY each question to the LLM (no chunks, no
retrieved context) so results are comparable against retrieval-augmented runs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# -----------------------------
# Frozen experiment parameters.
# Keep these constant for reproducibility across Condition 1 runs.
# -----------------------------
SYSTEM_PROMPT = "You are a helpful assistant. Answer clearly and concretely."
TEMPERATURE = 0.2
TOP_P = 0.9
MAX_NEW_TOKENS = 350
SEED = 42

DEFAULT_INPUT = Path("eval_queries.jsonl")
DEFAULT_OUTPUT = Path("rag_baseline_results.csv")
DEFAULT_ERRORS = Path("baseline_errors.jsonl")


@dataclass
class GenSettings:
    temperature: float = TEMPERATURE
    top_p: float = TOP_P
    max_new_tokens: int = MAX_NEW_TOKENS
    seed: int = SEED


class GenerationBackend:
    """Interface for model backends used by this baseline generator."""

    def model_name(self) -> str:
        raise NotImplementedError

    def generate(self, question: str, settings: GenSettings) -> str:
        raise NotImplementedError


class OllamaBackend(GenerationBackend):
    """Backend for local Ollama server (`ollama serve`)."""

    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def model_name(self) -> str:
        return self.model

    def generate(self, question: str, settings: GenSettings) -> str:
        # We fix system prompt and generation options for reproducibility.
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
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


class OpenAICompatibleBackend(GenerationBackend):
    """Backend for OpenAI API or any OpenAI-compatible chat endpoint.

    Supports:
    - OpenAI API (`https://api.openai.com/v1`)
    - llama.cpp server in OpenAI-compatible mode
    - vLLM / LM Studio / other compatible servers
    """

    def __init__(self, model: str, base_url: str, api_key: str):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def model_name(self) -> str:
        return self.model

    def generate(self, question: str, settings: GenSettings) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "max_tokens": settings.max_new_tokens,
            # Kept fixed; ignored by providers that do not support deterministic seed.
            "seed": settings.seed,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = _post_json(f"{self.base_url}/chat/completions", payload, headers=headers)
        return data["choices"][0]["message"]["content"].strip()


def _post_json(url: str, payload: Dict, headers: Optional[Dict[str, str]] = None) -> Dict:
    body = json.dumps(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    req = urllib.request.Request(url, data=body, headers=request_headers, method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
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


def iter_generate(
    backend: GenerationBackend,
    queries: Iterable[Dict[str, str]],
    settings: GenSettings,
    retries: int,
    retry_delay_sec: float,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    rows: List[Dict[str, str]] = []
    errors: List[Dict[str, str]] = []

    for item in queries:
        qid = item["qid"]
        question = item["query"]
        timestamp_utc = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()

        # Prompt format is constant and contains only the question.
        # System: You are a helpful assistant. Answer clearly and concretely.
        # User: <QUESTION>
        # Assistant: <ANSWER>
        # (no retrieval context allowed in this condition)
        answer = None
        last_error = None

        for attempt in range(1, retries + 2):
            try:
                answer = backend.generate(question=question, settings=settings)
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


def build_backend(args: argparse.Namespace) -> GenerationBackend:
    if args.backend == "ollama":
        model = args.model or os.getenv("OLLAMA_MODEL", "mistral:latest")
        base_url = args.base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        return OllamaBackend(model=model, base_url=base_url)

    if args.backend == "openai_compatible":
        model = args.model or os.getenv("LLM_MODEL") or "gpt-4o-mini"
        base_url = args.base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for backend=openai_compatible")
        return OpenAICompatibleBackend(model=model, base_url=base_url, api_key=api_key)

    raise ValueError(f"Unsupported backend: {args.backend}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate baseline (no retrieval) answers.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to eval_queries.jsonl")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to output CSV")
    p.add_argument(
        "--errors",
        type=Path,
        default=DEFAULT_ERRORS,
        help="Path to JSONL file with failed qids",
    )
    p.add_argument(
        "--backend",
        choices=["ollama", "openai_compatible"],
        default="ollama",
        help="LLM backend type",
    )
    p.add_argument("--model", default=None, help="Model name for chosen backend")
    p.add_argument("--base-url", default=None, help="Override backend base URL")
    p.add_argument("--api-key", default=None, help="API key (for openai_compatible backend)")
    p.add_argument("--retries", type=int, default=2, help="Retries after first failed request")
    p.add_argument("--retry-delay", type=float, default=1.5, help="Base delay between retries")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    settings = GenSettings()

    queries = load_queries(args.input)
    backend = build_backend(args)

    rows, errors = iter_generate(
        backend=backend,
        queries=queries,
        settings=settings,
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
