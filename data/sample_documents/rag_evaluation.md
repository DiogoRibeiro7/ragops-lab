# RAG Evaluation

RAG evaluation should measure retrieval quality and answer quality separately, because a good answer depends first on retrieving the right evidence and then on using it faithfully.

Retrieval metrics judge the ranked list of chunks. Recall at k measures whether the relevant chunks appear in the top k results, and is the metric to watch when a downstream generator can tolerate some noise. Mean reciprocal rank rewards placing the first relevant chunk near the top of the list. Context precision measures how many retrieved chunks are actually relevant, while context recall measures how much of the needed evidence was retrieved.

Answer metrics judge the generated text against the retrieved evidence. Faithfulness measures whether every claim in the answer is supported by the context, and is the primary defence against hallucination. Citation support checks that each cited chunk was actually retrieved. Answer relevance measures whether the response addresses the question. Operational metrics such as latency and cost per answer complete the picture, because a faithful answer that is too slow or too expensive is still a failure in production.
