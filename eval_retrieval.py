import json
import csv
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# === Paths ===
INDEX_PATH = Path("chunks") / "faiss.index"
META_PATH = Path("chunks") / "embeddings_meta.jsonl"
CHUNKS_PATH = Path("chunks") / "chunks.jsonl"
QUERY_PATH = Path("eval_queries.jsonl")
OUT_PATH = Path("retrieval_results.csv")

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

TOPK_LIST = [5, 10]

def main():
    print("Loading index...")
    index = faiss.read_index(str(INDEX_PATH))

    print("Loading metadata and chunks...")
    meta = [json.loads(l) for l in META_PATH.open("r", encoding="utf-8")]
    chunks = [json.loads(l)["text"] for l in CHUNKS_PATH.open("r", encoding="utf-8")]

    print("Loading queries...")
    queries = [json.loads(l) for l in QUERY_PATH.open("r", encoding="utf-8")]

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Running retrieval...")
    rows = []

    for q in queries:
        qid = q["qid"]
        query_text = q["query"]

        q_vec = model.encode([query_text], normalize_embeddings=True).astype("float32")

        max_k = max(TOPK_LIST)
        scores, indices = index.search(q_vec, max_k)

        for k in TOPK_LIST:
            for rank in range(k):
                idx = indices[0][rank]
                score = float(scores[0][rank])
                m = meta[idx]

                rows.append({
                    "qid": qid,
                    "query": query_text,
                    "k": k,
                    "rank": rank + 1,
                    "score": score,
                    "doc_id": m["doc_id"],
                    "chunk_id": m["chunk_id"],
                    "word_count": m["word_count"]
                })

    print("Writing CSV...")
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("Done.")
    print(f"Output saved to: {OUT_PATH.resolve()}")

if __name__ == "__main__":
    main()