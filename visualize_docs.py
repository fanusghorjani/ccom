import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    # Load the deduped dataset
    file_path = './results/evaluation/full_analysis_ready_deduped.xlsx'
    print(f"Loading data from {file_path}...")
    df = pd.read_excel(file_path)

    # We need to extract the 'rag_retrieved_doc_ids' and 'rag_retrieved_scores'
    # The format is string separated by '|'
    
    doc_scores = []
    
    for _, row in df.iterrows():
        docs = str(row.get('rag_retrieved_doc_ids', '')).split('|')
        scores = str(row.get('rag_retrieved_scores', '')).split('|')
        
        # Make sure they have the same length
        if len(docs) == len(scores) and docs[0] != 'nan':
            for doc, score in zip(docs, scores):
                try:
                    doc_scores.append({
                        'Document ID': doc.strip(),
                        'Retrieval Score': float(score.strip())
                    })
                except ValueError:
                    continue

    if not doc_scores:
        print("No valid document info found.")
        return

    # Convert to DataFrame
    ds_df = pd.DataFrame(doc_scores)

    # Compute summary statistics for each document
    # How many times it was used, and what's the average matching score
    summary_df = ds_df.groupby('Document ID').agg(
        Frequency=('Document ID', 'count'),
        Mean_Score=('Retrieval Score', 'mean'),
        Max_Score=('Retrieval Score', 'max')
    ).reset_index().sort_values(by='Frequency', ascending=False)
    
    print("\n--- Document Usage Summary ---")
    print(summary_df.head(20))
    print("------------------------------\n")

    # 1. Plot Document Frequencies
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=summary_df.head(20), # Top 20 most frequent
        x='Frequency',
        y='Document ID',
        palette='Blues_d'
    )
    plt.title('Top 20 Most Retrieved Documents (by Frequency)')
    plt.tight_layout()
    plt.savefig('doc_frequencies.png')
    print("Saved plot -> doc_frequencies.png")

    # 2. Plot Document Scores distribution
    # Only keep the top 10 most frequent docs to avoid clutter
    top_docs = summary_df.head(10)['Document ID'].tolist()
    filtered_ds = ds_df[ds_df['Document ID'].isin(top_docs)]

    plt.figure(figsize=(12, 8))
    sns.boxplot(
        data=filtered_ds, 
        x='Retrieval Score', 
        y='Document ID',
        order=top_docs,
        palette='Set2'
    )
    plt.title('Distribution of Retrieval Scores for Top 10 Documents')
    plt.tight_layout()
    plt.savefig('doc_scores_box.png')
    print("Saved plot -> doc_scores_box.png")

if __name__ == "__main__":
    main()
    main()
