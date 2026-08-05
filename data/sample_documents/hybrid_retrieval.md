# Hybrid Retrieval

Hybrid retrieval combines lexical and vector scores so that a system benefits from exact keyword matching and semantic recall at the same time. The lexical component anchors rare terms and identifiers, while the vector component closes the vocabulary-mismatch gap.

A simple and effective strategy is weighted score fusion. Each retriever's scores are normalised to a comparable range, then blended with a weight alpha, so the final score is alpha times the lexical score plus one minus alpha times the vector score. Setting alpha to one recovers pure lexical search and setting it to zero recovers pure vector search, which makes alpha a single knob for tuning the trade-off.

An alternative is Reciprocal Rank Fusion, which combines ranked lists using the reciprocal of each item's rank rather than its raw score. Rank fusion avoids the need to calibrate score scales across retrievers and is often more stable when the two systems produce very different score distributions.
