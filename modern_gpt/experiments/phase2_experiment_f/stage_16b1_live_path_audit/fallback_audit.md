# Fallback Synthesis Audit & Safety Verification

## Line-by-Line Audit of `synthesize_conservative_grounded_answer()`

The fallback synthesis routine in `backend/tools/grounding_validator.py` was audited line by line to verify that every asserted statement has traceable supporting evidence, zero dangling pronouns, and independent validation.

---

## 1. Traceability Matrix

For every sentence emitted by the fallback synthesis:

| Target Query | Emitted Sentence | Supporting Record | Supporting Source Passage | Assertion Supported |
| :--- | :--- | :---: | :--- | :---: |
| **"Who is Jensen Huang?"** | `Jensen Huang is a Taiwanese-born American entrepreneur who cofounded the American semiconductor company NVIDIA Corporation (NVDA) [SOURCE_2].` | `SOURCE_2` (Britannica) | *"Jensen Huang is a Taiwanese-born American entrepreneur who cofounded the American semiconductor company NVIDIA Corporation (NVDA)."* | **100%** |
| **"Who is Donald Trump?"** | `Donald John Trump (born June 14, 1946) is an American politician, media personality, and businessman who is the 47th president of the United States [SOURCE_1].` | `SOURCE_1` (Wikipedia) | *"Donald John Trump (born June 14, 1946) is an American politician, media personality, and businessman who is the 47th president of the United States."* | **100%** |
| **"What is the capital of Australia?"** | `As Australia's capital , Canberra is the most important centre for much of Australia's political reportage... [SOURCE_1].` | `SOURCE_1` (Wikipedia Canberra) | *"As Australia's capital, Canberra is the most important centre for much of Australia's political reportage..."* | **100%** |
| **"what was the result of the 2022 FIFA World Cup?"** | `The final match of the 2022 FIFA World Cup... was contested by Argentina and defending champions France [SOURCE_3].` | `SOURCE_3` (Wikipedia Final) | *"The final match of the 2022 FIFA World Cup... was contested by Argentina and defending champions France."* | **100%** |

---

## 2. Key Safety Invariants Implemented

1. **Dangling Pronoun Elimination**:
   Any sentence beginning with `He `, `She `, `It `, `They `, `His `, or `Her ` is strictly rejected unless the target entity is explicitly named within that same sentence. This completely eliminates the Stage 16B defect where Robert Trump's sentence (*"He was the younger brother of Donald Trump"*) was falsely spliced into Donald Trump's answer.

2. **Target Entity Scoping**:
   For entity inquiries (`Who is X?`, `Where is X?`), candidate records are checked for whether the query entity is the actual subject of the record. Sentences from secondary individuals who merely mention the query entity are filtered out.

3. **Single Best Defining Sentence for Biographical Inquiries**:
   Rather than concatenating snippets from multiple different people, single-entity queries select the single highest-confidence declarative definition from the primary authoritative source.

4. **Independent Validation Loop**:
   Before `synthesize_conservative_grounded_answer()` returns, it executes `EvidenceGroundingValidator.validate()` on its own constructed text. If validation fails, it safely returns explicit uncertainty:
   `"Retrieved sources do not contain sufficient evidence to answer: '<user_query>'."`
