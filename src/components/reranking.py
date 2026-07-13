def reciprocal_rank_fusion(bm25_ranks, vector_ranks, k=60):
    scores = {}
    for rank, doc_id in enumerate(bm25_ranks):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, doc_id in enumerate(vector_ranks):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
