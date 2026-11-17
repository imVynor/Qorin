# Prompt Changes & Synthesis Guidelines

## Objective

Document the prompt engineering and system instruction updates applied to guide QORIN toward evidence-grounded synthesis and explicit attribution.

---

## 1. Post-Tool Turn System Prompt Injection

In `backend/app.py` (`chat_stream`), whenever a tool call completes and execution transitions to the final synthesis round, the generation configuration dynamically appends strict synthesis guidelines:

```python
if current_round > 0 and tool_traces:
    # Dynamically inject strict grounding guidelines for final synthesis turn
    config.system_prompt = (
        f"{effective_system_prompt}\n\n"
        "STRICT GROUNDING & SYNTHESIS REQUIREMENTS:\n"
        "1. Base all statements strictly on the retrieved source passages. Do not invent dates, numbers, positions, or affiliations.\n"
        "2. Cite the source IDs (e.g. [SOURCE_1]) immediately after each claim.\n"
        "3. Do not introduce unrelated facts or repeat prior conversational claims."
    )
```

---

## 2. Dynamic Tool Observation Format

Before Stage 16B, tool outputs were passed as generic text strings:
```text
<tool_response>
https://en.wikipedia.org/wiki/Donald_Trump
Donald John Trump is the 47th president...
</tool_response>
```

Under Stage 16B, the `EvidenceBundle` renders structured passages with explicit `[SOURCE_X]` headers, titles, URLs, and entity tags:
```text
Retrieved 3 authoritative sources for query 'who is donald trump':

[SOURCE_1] Donald Trump - Wikipedia
URL: https://en.wikipedia.org/wiki/Donald_Trump
Key Entities: 1946, 2017, 2021, 47th president, Donald Trump, United States, Republican Party
Evidence Passage: Donald John Trump (born June 14, 1946) is an American politician, media personality, and businessman who is the 47th president of the United States. A member of the Republican Party, he served as the 45th president from 2017 to 2021.
```

---

## 3. Conservative Fallback Synthesis Template

When validation detects unsupported entities or contradiction in the model's generated response, `synthesize_conservative_grounded_answer()` produces a 100% verified response using this declarative template:

```text
Based on verified sources ([SOURCE_1], [SOURCE_2]): <Verified Declarative Sentence 1> [SOURCE_1]. <Verified Declarative Sentence 2> [SOURCE_2].
```

If retrieved sources contain zero relevant facts or query terms, the system declares uncertainty explicitly:
```text
Retrieved sources do not contain sufficient evidence to answer: '<user_query>'.
```
