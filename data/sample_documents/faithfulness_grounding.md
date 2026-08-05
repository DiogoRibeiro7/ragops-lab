# Faithfulness and Grounding

Grounding is the discipline of making a language model answer only from supplied evidence rather than from its parametric memory. A grounded RAG system passes retrieved context into the prompt and instructs the model to use nothing else, which keeps answers traceable to a source.

Faithfulness is the property that every claim in an answer is entailed by the retrieved context. A hallucination is an unfaithful claim: a statement that is fluent and plausible but unsupported by the evidence. Faithfulness can be estimated by decomposing an answer into atomic claims and checking each one against the context, either with lexical overlap heuristics or with a model acting as a judge.

Citations make grounding auditable. When a system returns the chunk identifiers it relied on, a reviewer can verify each claim against its source, and the system can reject answers that cite chunks which were never retrieved. Refusal is the correct behaviour when no retrieved chunk supports an answer; a system that refuses on weak evidence is safer than one that guesses, especially for unanswerable questions.
