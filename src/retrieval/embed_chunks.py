import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# Paths
CHUNKS_PATH = Path("processed") / "chunks" / "chunks.jsonl"
OUT_DIR = Path("processed") / "chunks"
EMB_PATH = OUT_DIR / "embeddings.npy"
META_PATH = OUT_DIR / "embeddings_meta.jsonl"

# Model (robust default, good quality)
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE = 64
NORMALIZE = True

def main():
    if not CHUNKS_PATH.exists():
        raise SystemExit(f"Missing file: {CHUNKS_PATH.resolve()}")

    OUT_DIR.mkdir(exist_ok=True)

    # Load texts + ids
    texts = []
    meta = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            texts.append(r["text"])
            meta.append({"doc_id": r["doc_id"], "chunk_id": r["chunk_id"], "word_count": r.get("word_count")})

    print(f"Loaded chunks: {len(texts)}")

    # Encode
    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=NORMALIZE,
    )

    emb = np.asarray(emb, dtype=np.float32)
    np.save(EMB_PATH, emb)

    # Save meta mapping (same order as embeddings.npy rows)
    with META_PATH.open("w", encoding="utf-8") as out:
        for m in meta:
            out.write(json.dumps(m, ensure_ascii=False) + "\n")

    print("Done.")
    print(f"Embeddings shape: {emb.shape}")
    print(f"Saved: {EMB_PATH.resolve()}")
    print(f"Saved: {META_PATH.resolve()}")

if __name__ == "__main__":
    main()