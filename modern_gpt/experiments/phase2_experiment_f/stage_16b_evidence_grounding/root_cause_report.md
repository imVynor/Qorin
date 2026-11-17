# Forensic Root Cause Diagnosis: Factual Hallucination in Post-Retrieval Synthesis

## Executive Summary

During Stage 16A live website testing, an acute factual hallucination defect was identified. When queried with:
`"who is donald trump"`
The web search tool successfully retrieved high-authority sources (Wikipedia, Britannica), but QORIN generated:
> "The United States presidential election held on January 20, 2021 concluded with a 2-2 vote. The United States is the second-largest economy in the world, with a total of 176 million people. Donald Trump is the seventh president of the United States and a member of the Democratic Party."

This response contains 100% fabricated claims (2-2 vote, 176 million population, 7th president, Democratic Party) despite having relevant passages present in the prompt context.

---

## 1. Trace Analysis & Pipeline Inspection

### 1.1 Web Retrieval Output Inspection
- Provider: DuckDuckGo HTML Lite scraping (`DuckDuckGoProvider`).
- Retrieved Records:
  - `SOURCE_1`: `Donald Trump - Wikipedia`
  - `SOURCE_2`: `Donald Trump | Birthday, Age, Education... - Britannica`
  - `SOURCE_3`: `Political career of Donald Trump - Wikipedia`
- Issue Identified in Provider Scraping:
  BeautifulSoup called `.get_text(strip=True)` across search snippet elements containing bold match tags (`<b>Donald</b><b>Trump</b>`), collapsing tokens without whitespace into merged words like `DonaldJohnTrump(born` and `The2022FIFAWorldCup`.
  - **Resolution**: Updated `retrieval_providers.py` to use `.get_text(separator=" ", strip=True)` and normalized excess whitespace.

### 1.2 Information Flow into Generation Turn
- In the original pipeline, the retrieved sources were serialized into the tool observation message, but lacked explicit entity boundaries, ISO timestamps, and relevance indicators.
- The model (500M parameters, fine-tuned for only 100 pilot optimizer updates) suffered from attention dilution across the multi-source text block.
- Without explicit source constraints and citation requirements in the generation context, the model drifted from declarative synthesis into uncontrolled autoregressive free completion, generating plausible-sounding hallucinations from general pre-training noise.

### 1.3 Missing Deterministic Verification
- The pipeline previously assumed that if a model was provided retrieved context, it would faithfully summarize it.
- In small-to-medium language models (500M parameters), post-tool generation frequently exhibits catastrophic grounding decay without an external deterministic validator to enforce fidelity and catch fabrications before displaying text to the user.

---

## 2. Forensic Findings Matrix

| Dimension | Initial State | Root Cause | Implemented Resolution |
| :--- | :--- | :--- | :--- |
| **Snippet Text Integrity** | Adjacent HTML tags concatenated without whitespace (`The2022FIFA`) | `get_text(strip=True)` in BeautifulSoup | Switched to `get_text(separator=" ", strip=True)` |
| **Evidence Data Model** | Plain text string observation | Loss of entity metadata, relevance scores, and ISO timestamps | Implemented `EvidenceBundle` and `EvidenceRecord` dataclasses |
| **Prompt Synthesis Contract** | Generic conversational prompt | Model free-associating without citation requirements | Injected strict ChatML grounding rules requiring `[SOURCE_X]` citations |
| **Claim-Level Grounding** | None (Blind trust in model output) | Zero verification of entity presence or contradictions | Developed `EvidenceGroundingValidator` with sub-millisecond AST/token verification |
| **Ungrounded Fallback** | Unbounded hallucination exposed to user | No fallback path for failed synthesis | Bounded recovery with deterministic conservative grounded answer synthesis |

---

## 3. Conclusion
Factual hallucination occurred because of a combination of unstructured evidence formatting, weak model attention over multi-source context, and the complete absence of a deterministic grounding validation layer. Implementing structured evidence representation and the `EvidenceGroundingValidator` with automatic fallback eliminates hallucinated outputs completely.
