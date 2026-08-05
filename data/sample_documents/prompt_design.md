# Prompt Design for Grounded Generation

The prompt is the contract between a retrieval system and the language model. For grounded generation it must do three things: supply the retrieved evidence, constrain the model to use only that evidence, and specify the exact output format the system expects.

A system instruction sets the rules. A typical grounding instruction tells the model to answer only from the provided contexts, to cite the chunk identifiers it relied on, and to refuse when the evidence is insufficient. Placing each context behind its identifier lets the model reference sources precisely and lets the system validate the citations it returns.

Structured output makes generation safe to parse. Asking the model to return JSON with an answer, a list of citations, and a refusal flag turns a free-form response into a checkable object. The system can then validate the schema, reject citations that point to chunks which were never retrieved, and mark an answer as grounded only when it is both cited and not a refusal. A strict output contract is what lets evaluation run automatically over thousands of answers.
