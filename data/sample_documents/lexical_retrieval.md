# Lexical Retrieval and BM25

Lexical retrieval ranks documents by exact term overlap between the query and the indexed text. BM25 is the dominant lexical scoring function. It extends TF-IDF with two ideas: term-frequency saturation, so repeated terms give diminishing returns, and document-length normalisation, so long passages are not unfairly rewarded for containing more words.

BM25 exposes two tuning parameters. The k1 parameter controls how quickly term-frequency saturates; typical values fall between 1.2 and 2.0. The b parameter controls how aggressively length normalisation is applied, where b equal to zero disables it and b equal to one applies it fully. A common default is k1 of 1.5 and b of 0.75.

The strength of lexical retrieval is precision on rare keywords, identifiers, and exact phrases. Its weakness is vocabulary mismatch: a query for "car" will not match a passage that only says "automobile", because BM25 has no notion of meaning. This is the gap that vector retrieval is designed to close.
