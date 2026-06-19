# Architecture

```text
Raw documents
  -> Ingestion
  -> Chunk store
  -> Retriever
  -> Generator
  -> Evaluator
  -> Trace store
  -> API / dashboard
```

The central design principle is evaluation-first development. Every generated answer should be linked to retrieved context and every failure should be measurable.
