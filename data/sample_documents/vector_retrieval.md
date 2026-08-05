# Vector Retrieval and Embeddings

Vector retrieval represents text as dense embeddings and ranks passages by similarity in that vector space. An embedding model maps a chunk of text to a fixed-length vector so that semantically related passages land close together, even when they share no words.

Cosine similarity is the standard scoring metric for embeddings. It measures the angle between two vectors and ignores their magnitude, which makes it robust to differences in passage length. Scores range from minus one to one, and normalised embeddings let cosine similarity be computed as a simple dot product.

Because comparing a query against every stored vector is expensive at scale, production systems use approximate nearest neighbour search. Index structures such as HNSW graphs and inverted file indexes trade a small amount of recall for a large gain in latency. The main weakness of vector retrieval is the opposite of lexical search: it can miss exact keyword matches and rare identifiers, retrieving passages that are merely topically related.
