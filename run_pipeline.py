import subprocess

def run_step(description, command):
    print(f"\n--- {description} ---")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Error in step: {description}")
        exit(1)

if __name__ == "__main__":

    # 1. Preprocessing
    run_step("Cleaning texts", "python src/preprocessing/clean_texts.py")

    # 2. Chunking
    run_step("Chunking texts", "python src/preprocessing/chunk_texts.py")

    # 3. Embedding
    run_step("Creating embeddings", "python src/retrieval/embed_chunks.py")

    # 4. Build index
    run_step("Building FAISS index", "python src/retrieval/build_faiss.py")

    # 5. Baseline generation
    run_step("Running baseline generation", "python src/generation/baseline_generate.py")

    # 6. RAG generation
    run_step("Running RAG generation", "python src/generation/rag_generate.py")

    print("\nPipeline completed successfully.")
