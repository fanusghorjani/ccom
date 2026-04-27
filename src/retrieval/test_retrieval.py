import json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
INDEX_PATH = Path("processed") / "chunks" / "faiss.index"
META_PATH = Path("processed") / "chunks" / "embeddings_meta.jsonl"
CHUNKS_PATH = Path("processed") / "chunks" / "chunks.jsonl"

TOPK = 5
QUERY = "How should a revolutionary organization structure cadre work and discipline?"

def main():
    index = faiss.read_index(str(INDEX_PATH))

    # load meta + chunk texts (same order)
    meta = [json.loads(l) for l in META_PATH.open("r", encoding="utf-8")]
    texts = [json.loads(l)["text"] for l in CHUNKS_PATH.open("r", encoding="utf-8")]

    model = SentenceTransformer(MODEL_NAME)
    q = model.encode([QUERY], normalize_embeddings=True).astype("float32")

    scores, idxs = index.search(q, TOPK)
    print("QUERY:", QUERY)
    print("\nTOPK RESULTS:")
    for rank, (i, s) in enumerate(zip(idxs[0], scores[0]), start=1):
        m = meta[i]
        snippet = texts[i].replace("\n", " ")
        snippet = snippet[:300] + "..."
        print(f"\n#{rank} score={s:.4f} {m['chunk_id']} ({m['doc_id']})")
        print(snippet)

if __name__ == "__main__":
    main()