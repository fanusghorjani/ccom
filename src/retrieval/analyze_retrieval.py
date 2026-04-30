import pandas as pd

df = pd.read_csv("retrieval_results.csv")

def top_docs(k: int, topn: int = 15):
    sub = df[df["k"] == k]
    counts = sub["doc_id"].value_counts().head(topn)
    return counts

print("\n=== Dominance: doc_id frequency in results ===")
for k in sorted(df["k"].unique()):
    print(f"\nTop doc_ids for k={k}:")
    print(top_docs(k).to_string())

print("\n=== Share of top-5 results coming from top documents (k=5 only) ===")
sub5 = df[df["k"] == 5]
total5 = len(sub5)
top5_doc = sub5["doc_id"].value_counts().head(5)
print((top5_doc / total5).to_string())

print("\n=== Per-question concentration (k=10): how many unique docs appear in top-10? ===")
sub10 = df[df["k"] == 10]
uniq_docs_per_q = sub10.groupby("qid")["doc_id"].nunique().describe()
print(uniq_docs_per_q.to_string())

print("\n=== Top document per question (k=5) ===")
top5 = df[df["k"] == 5]
top_doc_per_q = (
    top5.groupby("qid")["doc_id"]
    .agg(lambda x: x.value_counts().index[0])
)

print(top_doc_per_q.to_string())