# QORIN Claim-Level Evidence Grounding Evaluation Protocol

**Document**: `evidence_grounding_protocol.md`  
**Status**: Formal Evaluation Specification (Replacing Superficial N-Gram Word Overlap)  
**Target System**: QORIN Evaluation Harness for Experiment E  
**Scope**: Factual and Evidence-Grounded Task Assessment

---

## 1. Critique & Invalidation of Superficial Word Overlap

In Stage 5, "Evidence Utilization" was evaluated using a 3-word set intersection:
$$\text{overlap} = \text{words}(\text{tool\_output}) \cap \text{words}(\text{final\_response}), \quad \text{utilized} \iff |\text{overlap}| \ge 3$$

As uncovered during the forensic audit, this metric is fundamentally flawed:
1. **False Positives via Common Vocabulary**: Common stopwords, topic nouns, and general vocabulary of length $\ge 3$ (e.g. "information", "report", "system", "from") easily triggered $|\text{overlap}| \ge 3$ even when the model hallucinated all actual numbers and dates.
2. **Deceptive Grounding Rate**: While Exp D registered an 81.13% 3-word overlap rate on executed tool calls, only **24.53%** of those calls produced a factually correct answer. Over 75% of "utilized" cases were factually inaccurate.
3. **Inability to Detect Contradictions**: If a web snippet stated "Revenue was \$10B" and the model answered "Revenue was \$25B, according to the report", the word-overlap metric scored it as "utilized" because "revenue", "according", and "report" matched.

Effective immediately, token-level n-gram overlap is **deprecated and barred** from serving as an evidence-grounding metric.

---

## 2. Claim-Level Factual Verification Framework

In Experiment E, evidence grounding is evaluated using a formal **Atomic Claim Decomposition and Verification Framework**:

```
[Model Final Response]
         │
         ▼
[1. Atomic Claim Extraction] ──> Splits text into discrete factual propositions C_1, C_2, ...
         │
         ▼
[2. Evidence Alignment]     ──> Aligns each C_i to Retrieved Evidence E (or marks Parametric)
         │
         ▼
[3. Grounding Verification] ──> Programmatic & Gold Annotation Check:
                                 - Entity match?
                                 - Numerical value within tolerance?
                                 - Unit & qualifier consistency?
                                 - Temporal / date consistency?
         │
         ▼
[4. Status Classification]   ──> Supported | Partially Supported | Unsupported | Contradicted
```

---

## 3. Four-Tier Claim Grounding Taxonomy

Every discrete factual claim $C_i$ extracted from an assistant response is assigned exactly one grounding label relative to the retrieved tool observation $O$:

| Grounding State | Definition | Criterion |
| :--- | :--- | :--- |
| **`SUPPORTED`** | The claim is directly and unambiguously verified by explicit text in the retrieved tool observation $O$. | Entity, numerical value, unit, and date match $O$ exactly or within specified scientific tolerance. |
| **`PARTIALLY_SUPPORTED`** | The core entity or qualitative direction is supported, but a specific numerical precision, unit qualifier, or date is slightly distorted without outright contradiction. | Entity matches, metric is directionally correct, but minor qualifier or rounding error present. |
| **`UNSUPPORTED`** | The claim asserts an empirical fact not found in $O$, but which does not directly contradict $O$. (Extrapolations, external claims, or ungrounded assertions). | Fact is absent from $O$. May represent parametric knowledge or pure hallucination. |
| **`CONTRADICTED`** | The claim directly conflicts with numerical values, entities, dates, or conclusions explicitly stated in $O$. | Explicit factual collision (e.g. $O$ states \$383B, model claims \$211B). |

---

## 4. Formal Metric Formulations

Let an evaluation run across a benchmark subset produce a set of responses $\{R_k\}_{k=1}^K$. For each response $R_k$, let $C(R_k)$ be the set of extracted atomic claims, and $G(R_k)$ be the set of gold required claims.

### 4.1 Claim-Level Precision ($CP$)
The proportion of empirical claims made by the model that are factually verified by retrieved evidence:
$$CP = \frac{\sum_{k} |\{c \in C(R_k) : \text{status}(c) = \text{SUPPORTED}\}|}{\sum_{k} |C(R_k)|}$$

### 4.2 Claim-Level Coverage ($CC$)
The proportion of gold target claims required by the task prompt that were successfully asserted and supported in the response:
$$CC = \frac{\sum_{k} |\{g \in G(R_k) : g \text{ is covered and SUPPORTED}\}|}{\sum_{k} |G(R_k)|}$$

### 4.3 Numerical Extraction Accuracy ($NEA$)
Specifically measures fidelity on numerical quantities (currencies, percentages, counts, physical metrics):
$$NEA = \frac{N_{\text{numbers\_correctly\_extracted\_and\_matched}}}{N_{\text{total\_numerical\_entities\_in\_gold}}}$$
*Rule: Numbers must match with correct units (e.g. "383 billion dollars", not "383 million" or "383").*

### 4.4 Entity-Value Association Accuracy ($EVAA$)
Measures whether extracted numerical values are bound to the correct named entities in multi-entity tasks:
$$EVAA = \frac{N_{\text{correct\_entity\_metric\_bindings}}}{N_{\text{total\_entity\_metric\_pairs}}}$$
*Rule: Prevents entity-scrambling errors (e.g. attributing Apple's revenue to Microsoft).*

### 4.5 Unsupported Factual Claim Rate ($UFCR$)
The fraction of claims that represent ungrounded external assertions or hallucinations:
$$UFCR = \frac{\sum_k |\{c \in C(R_k) : \text{status}(c) \in \{\text{UNSUPPORTED}, \text{CONTRADICTED}\}\}|}{\sum_k |C(R_k)|}$$

### 4.6 Citation Correctness ($CR$)
For responses that provide inline citations (e.g. `[SOURCE 1]`, `[1]`):
$$CR = \frac{N_{\text{citations\_pointing\_to\_true\_supporting\_source}}}{N_{\text{total\_inline\_citations\_emitted}}}$$
*Rule: A citation is invalid if the cited URL/source does not contain the specific fact being asserted in that sentence.*

---

## 5. Human-Verified Gold Benchmark Sub-Suite

To prevent metric drift and ensure absolute ground-truth validity, a dedicated **50-Task Gold Verification Sub-Suite** will be established:

### 5.1 Gold Annotation Specification
For each of the 50 tasks:
1. **Pre-Annotated Gold Claims ($G$)**: Domain experts write the exact atomic propositions required for full credit.
2. **Explicit Entity-Value Bindings**: Key-value pairs with permissible scientific tolerances ($\pm 0.5\%$).
3. **Negative Fact Boundaries**: Known misconceptions or contradictory figures explicitly flagged for automated detection.
4. **Human Double-Audit**: All model responses in the Gold Sub-Suite undergo independent, double-blind verification by two evaluators with Cohen's $\kappa \ge 0.85$ inter-annotator agreement.
