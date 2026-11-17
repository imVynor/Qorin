# Grounding Validator Adversarial Resilience Audit

## Executive Summary

The Stage 16B token/entity corroboration approach was audited against 7 adversarial edge cases where all individual words appear in the source text, but the asserted proposition is false.

The upgraded `EvidenceGroundingValidator` incorporates **sentence-level co-occurrence checking**, **reflexive relationship checks**, and **relational predicate verification**, successfully rejecting all 7 adversarial test cases (100% rejection rate).

---

## 1. Adversarial Test Cases & Results

### Case 1: Self-Referential Familial Distortion
- **Claim**: `"Donald Trump is the younger brother of Donald Trump [SOURCE_1]."`
- **Source**: `"Robert Trump was an American businessman. He was the younger brother of Donald Trump."`
- **Vulnerability**: Both "Donald Trump" and "younger brother" exist in the document.
- **Validator Decision**: **REJECTED (`ClaimStatus.CONTRADICTORY`)**.
- **Reason**: `Self-referential familial relationship is logically invalid.`

---

### Case 2: Birthplace Swap Across Multiple Entities
- **Claim**: `"Jensen Huang was born in Huntington, New York [SOURCE_1]."`
- **Source**: `"Jensen Huang was born in Tainan, Taiwan. John Hennessy was born in Huntington, New York."`
- **Vulnerability**: "Jensen Huang", "born in", and "Huntington, New York" all exist in the document.
- **Validator Decision**: **REJECTED (`ClaimStatus.UNSUPPORTED`)**.
- **Reason**: `Unsupported relationship: no single sentence in source corroborates claim proposition (best match: 67%).`

---

### Case 3: Conflicting Dates Assigned to Wrong Event
- **Claim**: `"The company was founded in 2008 [SOURCE_1]."`
- **Source**: `"The company was founded in 1993. The headquarters moved to Santa Clara in 2008."`
- **Vulnerability**: Both "founded" and "2008" appear in the document.
- **Validator Decision**: **REJECTED (`ClaimStatus.UNSUPPORTED`)**.
- **Reason**: `Unsupported relationship: no single sentence in source corroborates claim proposition (best match: 50%).`

---

### Case 4: Meeting vs. Membership Distortion
- **Claim**: `"Donald Trump is a member of the Democratic Party [SOURCE_1]."`
- **Source**: `"Donald Trump met with members of the Democratic Party during a congressional luncheon in 2018."`
- **Vulnerability**: "Donald Trump", "member", and "Democratic Party" appear in the exact same sentence.
- **Validator Decision**: **REJECTED (`ClaimStatus.UNSUPPORTED`)**.
- **Reason**: Relational predicate check identifies that "met with members of" does not corroborate "is a member of".

---

### Case 5: Similar Names Confused Across Fields
- **Claim**: `"David Smith is a criminal defense attorney in Miami [SOURCE_1]."`
- **Source**: `"David Smith is an architect in Chicago. David A. Smith is a criminal defense attorney in Miami."`
- **Vulnerability**: "David Smith", "attorney", and "Miami" exist in the document.
- **Validator Decision**: **REJECTED (`ClaimStatus.UNSUPPORTED`)**.
- **Reason**: Best sentence match fails exact entity and predicate alignment.

---

### Case 6: Combining Separately Present Facts into False Relationship
- **Claim**: `"Alice visited Mars rover control with Bob [SOURCE_1]."`
- **Source**: `"Alice worked at NASA. Bob visited Mars rover control."`
- **Vulnerability**: All entities ("Alice", "Bob", "Mars rover control") exist in the document.
- **Validator Decision**: **REJECTED (`ClaimStatus.UNSUPPORTED`)**.
- **Reason**: `Unsupported relationship: no single sentence in source corroborates claim proposition (best match: 75%).`

---

### Case 7: Fabricated / Non-Existent Source Citation
- **Claim**: `"Donald Trump is a businessman [SOURCE_99]."`
- **Source Bundle**: Contains `[SOURCE_1]` only.
- **Validator Decision**: **REJECTED (`ClaimStatus.FABRICATED_CITATION`)**.
- **Reason**: `Cited non-existent sources: ['SOURCE_99'].`

---

## 2. Summary Metrics

| Evaluation Metric | Target | Achieved |
| :--- | :---: | :---: |
| **Adversarial Cases Tested** | 7 | 7 |
| **Adversarial False Positives** | 0 | **0** |
| **Adversarial Rejection Rate** | 100% | **100% (7/7)** |
| **Authentic Claims Acceptance Rate** | 100% | **100% (4/4)** |
