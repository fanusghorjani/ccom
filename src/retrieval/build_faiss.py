from pathlib import Path
import numpy as np
import faiss
EMB_PATH = Path("processed") / "chunks" / "embeddings.npy"
INDEX_PATH = Path("processed") / "chunks" / "faiss.index"

def main():
    if not EMB_PATH.exists():
        raise SystemExit(f"Missing: {EMB_PATH.resolve()}")

    X = np.load(EMB_PATH).astype("float32")
    n, d = X.shape
    print("Loaded embeddings:", X.shape)

    # If embeddings were normalized, inner product == cosine similarity
    index = faiss.IndexFlatIP(d)
    index.add(X)
    # falls ordner mal fehlt
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_PATH))
    print("Saved index:", INDEX_PATH.resolve())
    print("Index size:", index.ntotal)

if __name__ == "__main__":
    main()