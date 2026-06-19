# Chunking Strategies

Chunking splits documents into passages small enough to retrieve precisely and to fit inside a model's context window. The chunk size and the overlap between adjacent chunks are the two parameters that most affect retrieval quality.

Small chunks improve precision because each passage covers a single idea, so a retrieved chunk is more likely to be fully relevant. The cost is recall: an answer that spans several sentences may be split across chunk boundaries and never retrieved as a whole. Large chunks have the opposite profile, capturing more context at the expense of diluting the relevant sentence with unrelated text.

Overlap mitigates boundary effects by repeating a window of text between neighbouring chunks, so a fact that sits at a boundary still appears intact in at least one chunk. Typical configurations use an overlap of ten to twenty percent of the chunk size. Chunking can operate on characters, tokens, or semantic units such as sentences and paragraphs; sentence-aware chunking usually preserves meaning better than fixed-width splitting.
