from app.retrieval.reranker import Reranker

query = "How many casual leave days do employees receive?"

docs = hybrid.search(
    query,
    query_embedding,
    k=5
)

reranker = Reranker()

results = reranker.rerank(
    query,
    docs,
    top_k=3
)

print("\nRe-ranked Results\n")

for i, doc in enumerate(results, start=1):
    print("=" * 80)
    print(f"Rank {i}")
    print(doc.file_name)
    print(doc.text)