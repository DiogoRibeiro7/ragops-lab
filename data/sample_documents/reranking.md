# Reranking

Reranking adds a second, more accurate scoring stage after an initial retriever returns a candidate set. The first stage optimises for recall and speed, pulling perhaps the top fifty chunks cheaply; the reranker then reorders that shortlist for precision before the chunks reach the generator.

The dominant reranking architecture is the cross-encoder. Unlike a bi-encoder, which embeds the query and each passage independently, a cross-encoder reads the query and a passage together and scores their relevance jointly. This joint attention captures fine-grained interactions that independent embeddings miss, which makes cross-encoders markedly more accurate.

The cost of that accuracy is latency, because the model must run once per query-passage pair and cannot precompute passage vectors. The standard pattern is therefore retrieve-then-rerank: use a fast retriever to shrink the candidate pool to a few dozen chunks, then apply the cross-encoder only to that shortlist. This keeps the expensive model off the full corpus while still delivering most of its precision benefit.
