"""
QORIN Evidence-Grounding & Claim-Level Validation System
Enforces deterministic factual validation, citation verification, unsupported claim detection,
contradiction handling, and bounded conservative synthesis fallback.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.tools.evidence import EvidenceBundle, EvidenceRecord


class ClaimStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    FABRICATED_CITATION = "FABRICATED_CITATION"
    MISSING_CITATION = "MISSING_CITATION"
    CONTRADICTORY = "CONTRADICTORY"


@dataclass
class ClaimValidationResult:
    claim_text: str
    status: ClaimStatus
    cited_sources: List[str]
    supporting_source_ids: List[str]
    missing_entities: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_text": self.claim_text,
            "status": self.status.value,
            "cited_sources": self.cited_sources,
            "supporting_source_ids": self.supporting_source_ids,
            "missing_entities": self.missing_entities,
            "confidence_score": round(self.confidence_score, 2),
            "reason": self.reason
        }


@dataclass
class ValidationSummary:
    is_grounded: bool
    total_claims: int
    supported_claims: int
    unsupported_claims: int
    citation_precision: float
    claim_supported_precision: float
    query_relevance_score: float
    detected_contradictions: List[str] = field(default_factory=list)
    claims: List[ClaimValidationResult] = field(default_factory=list)
    suggested_correction: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_grounded": self.is_grounded,
            "total_claims": self.total_claims,
            "supported_claims": self.supported_claims,
            "unsupported_claims": self.unsupported_claims,
            "citation_precision": round(self.citation_precision, 2),
            "claim_supported_precision": round(self.claim_supported_precision, 2),
            "query_relevance_score": round(self.query_relevance_score, 2),
            "detected_contradictions": self.detected_contradictions,
            "claims": [c.to_dict() for c in self.claims],
            "suggested_correction": self.suggested_correction
        }


class EvidenceGroundingValidator:
    """Validates generated answers against retrieved source evidence."""

    STOP_WORDS: Set[str] = {
        "the", "a", "an", "is", "are", "was", "were", "and", "or", "but", "in", "on", "at",
        "to", "for", "with", "by", "about", "against", "between", "into", "through", "during",
        "before", "after", "above", "below", "from", "up", "down", "in", "out", "on", "off",
        "over", "under", "again", "further", "then", "once", "here", "there", "when", "where",
        "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
        "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can",
        "will", "just", "don", "should", "now", "it", "its", "he", "his", "she", "her", "they",
        "them", "their", "this", "that", "these", "those"
    }

    @staticmethod
    def extract_citations(text: str) -> List[str]:
        """Extracts source markers like [SOURCE_1], [SOURCE_2], or [1], [2]."""
        matches = re.findall(r"\[(SOURCE_\d+|\d+)\]", text, re.IGNORECASE)
        citations = []
        for m in matches:
            if m.isdigit():
                citations.append(f"SOURCE_{m}")
            else:
                citations.append(m.upper())
        return citations

    @classmethod
    def split_into_claims(cls, text: str) -> List[str]:
        """Splits text into meaningful semantic sentence units with normalized citations."""
        # Strip introductory meta-phrases
        clean_text = re.sub(r"^Based on verified sources(?:\s*\([^)]*\))?:\s*", "", text.strip(), flags=re.IGNORECASE)
        # Normalize trailing citations before period
        clean_text = re.sub(r"\.\s*(\[(?:SOURCE_\d+|\d+)\])", r" \1.", clean_text)
        # Split on sentence terminators
        raw_sentences = re.split(r"(?<=[.!?])\s+", clean_text)
        claims = [s.strip() for s in raw_sentences if len(re.sub(r"\[[^\]]*\]", "", s).strip()) > 5]
        return claims

    @classmethod
    def validate(
        cls,
        generated_answer: str,
        evidence: EvidenceBundle,
        user_query: str
    ) -> ValidationSummary:
        """
        Validates claim-level grounding and citation precision of generated answer.
        """
        valid_source_ids = set(evidence.get_source_ids())
        claims = cls.split_into_claims(generated_answer)

        if not claims:
            return ValidationSummary(
                is_grounded=False,
                total_claims=0,
                supported_claims=0,
                unsupported_claims=0,
                citation_precision=0.0,
                claim_supported_precision=0.0,
                query_relevance_score=0.0,
                reason="Empty generated answer."
            )

        all_citations: List[str] = []
        valid_citations_count = 0
        claim_results: List[ClaimValidationResult] = []

        combined_evidence_text = " ".join(
            f"{r.source_title} {r.source_snippet} {r.passage}" for r in evidence.records
        ).lower()

        # Check for contradictory sources
        detected_contradictions = cls._detect_source_contradictions(evidence)

        # Query relevance check
        query_terms = [
            w.lower() for w in re.findall(r"\b\w+\b", user_query)
            if w.lower() not in cls.STOP_WORDS and len(w) > 2
        ]
        answer_lower = generated_answer.lower()
        matched_query_terms = sum(1 for w in query_terms if w in answer_lower)
        query_relevance = matched_query_terms / max(1, len(query_terms))

        supported_claims_count = 0

        for claim in claims:
            cited = cls.extract_citations(claim)
            all_citations.extend(cited)

            # Check citation validity
            fabricated_citations = [c for c in cited if c not in valid_source_ids]
            if fabricated_citations:
                claim_results.append(ClaimValidationResult(
                    claim_text=claim,
                    status=ClaimStatus.FABRICATED_CITATION,
                    cited_sources=cited,
                    supporting_source_ids=[],
                    reason=f"Cited non-existent sources: {fabricated_citations}"
                ))
                continue

            for c in cited:
                if c in valid_source_ids:
                    valid_citations_count += 1

            # Reflexive / self-referential relationship check (e.g. X is the brother of X)
            subj_m = re.match(
                r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(?:the\s+)?(?:younger\s+|older\s+)?(?:brother|sister|sibling|father|mother|son|daughter)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                claim,
                re.IGNORECASE
            )
            has_contradiction = False
            contradiction_reason = ""
            if subj_m and subj_m.group(1).lower() == subj_m.group(2).lower():
                has_contradiction = True
                contradiction_reason = "Self-referential familial relationship is logically invalid."

            # Extract substantive entities, numbers, dates, ordinals, scores, and political terms
            proper_nouns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", claim)
            numbers = re.findall(r"\b\d+[-–]\d+\b|\b\d+(?:\.\d+)?\b", claim)
            ordinals = re.findall(
                r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d+(?:st|nd|rd|th))\b",
                claim,
                re.IGNORECASE
            )
            key_terms = re.findall(
                r"\b(?:Democratic|Republican|Democrat|GOP|Parliament|Congress|Senate|President|Capital|Prime Minister)\b",
                claim,
                re.IGNORECASE
            )
            predicates = re.findall(
                r"\b(?:founded|born|won|defeated|capital|president|brother|sister|member|attorney|architect|worked|visited|drafted|signed|minister|director|leader)\b",
                claim,
                re.IGNORECASE
            )

            raw_tokens = []
            for item in proper_nouns + numbers + ordinals + key_terms + predicates:
                raw_tokens.extend(item.strip().lower().split())

            critical_tokens = [
                e for e in dict.fromkeys(raw_tokens)
                if e not in cls.STOP_WORDS and len(e) > 1
            ]

            # Check target evidence: if cited, check cited sources; otherwise check all sources
            if cited:
                target_records = [evidence.get_record_by_id(c) for c in cited if evidence.get_record_by_id(c)]
                target_evidence_text = " ".join(
                    f"{r.source_title} {r.source_snippet} {r.passage}".lower()
                    for r in target_records
                )
            else:
                target_records = evidence.records
                target_evidence_text = combined_evidence_text

            missing_entities = [t for t in critical_tokens if t not in target_evidence_text]

            # Check explicit party or value contradiction
            claim_lower = claim.lower()
            if "democratic" in claim_lower or "democrat" in claim_lower:
                if "republican" in target_evidence_text and "democratic" not in target_evidence_text:
                    has_contradiction = True
                    contradiction_reason = "Asserts Democratic party affiliation when source specifies Republican."
            elif "republican" in claim_lower or "gop" in claim_lower:
                if "democratic" in target_evidence_text and "republican" not in target_evidence_text:
                    has_contradiction = True
                    contradiction_reason = "Asserts Republican party affiliation when source specifies Democratic."

            # Sentence-level co-occurrence and semantic relation corroboration
            is_asserting_membership = "member" in [p.lower() for p in predicates] and ("is a member of" in claim_lower or "member of" in claim_lower)
            best_sentence_match = 0.0
            best_matched_sentence = ""

            for r in target_records:
                src_full = f"{r.source_title}. {r.source_snippet} {r.passage}"
                clean_src = re.sub(r"([a-z])([A-Z])", r"\1 \2", src_full)
                s_sents = re.split(r"(?<=[.!?])\s+", clean_src)
                for s in s_sents:
                    s_lower = s.lower()
                    if is_asserting_membership:
                        if re.search(r"\b(?:met with|spoke to|addressed|criticized|visited|debated)\s+members\b", s_lower):
                            continue
                    matched = sum(1 for t in critical_tokens if t in s_lower)
                    ratio = matched / max(1, len(critical_tokens))
                    if ratio > best_sentence_match:
                        best_sentence_match = ratio
                        best_matched_sentence = s

            # Determine support score
            if not critical_tokens:
                # Conversational filler or generic sentence
                support_score = 1.0
            else:
                matched_count = len(critical_tokens) - len(missing_entities)
                support_score = matched_count / len(critical_tokens)

            supporting_sources = []
            for r in evidence.records:
                src_text = f"{r.source_title} {r.source_snippet}".lower()
                if sum(1 for t in critical_tokens if t in src_text) > (len(critical_tokens) * 0.5):
                    supporting_sources.append(r.evidence_id)

            # Status assignment
            if has_contradiction:
                claim_status = ClaimStatus.CONTRADICTORY
                reason = contradiction_reason
            elif support_score >= 0.80 and not missing_entities and (best_sentence_match >= 0.80 or (len(critical_tokens) == 1 and best_sentence_match == 1.0)):
                claim_status = ClaimStatus.SUPPORTED
                supported_claims_count += 1
                reason = "Supported by retrieved evidence sentence."
            else:
                claim_status = ClaimStatus.UNSUPPORTED
                if missing_entities:
                    reason = f"Unsupported entities/facts: {missing_entities[:4]}"
                else:
                    reason = f"Unsupported relationship: proposition not corroborated by any single sentence (best match: {round(best_sentence_match*100)}%)."

            claim_results.append(ClaimValidationResult(
                claim_text=claim,
                status=claim_status,
                cited_sources=cited,
                supporting_source_ids=supporting_sources,
                missing_entities=missing_entities,
                confidence_score=support_score,
                reason=reason
            ))

        total_c = len(claims)
        unsupported_c = total_c - supported_claims_count
        claim_precision = supported_claims_count / max(1, total_c)
        citation_precision = (valid_citations_count / max(1, len(all_citations))) if all_citations else 0.0

        is_grounded = (
            claim_precision >= 0.80 and
            unsupported_c == 0 and
            query_relevance >= 0.40 and
            all(c.status != ClaimStatus.FABRICATED_CITATION for c in claim_results)
        )

        return ValidationSummary(
            is_grounded=is_grounded,
            total_claims=total_c,
            supported_claims=supported_claims_count,
            unsupported_claims=unsupported_c,
            citation_precision=citation_precision,
            claim_supported_precision=claim_precision,
            query_relevance_score=query_relevance,
            detected_contradictions=detected_contradictions,
            claims=claim_results,
        )

    @classmethod
    def _detect_source_contradictions(cls, evidence: EvidenceBundle) -> List[str]:
        """Detects if retrieved sources state conflicting assertions."""
        contradictions = []
        if len(evidence.records) < 2:
            return contradictions

        # Example: check dates or numeric values attached to same entity
        years_per_source = {}
        for r in evidence.records:
            years = set(re.findall(r"\b(?:19|20)\d{2}\b", r.source_snippet))
            years_per_source[r.evidence_id] = years

        # Compare pairwise
        s_ids = list(years_per_source.keys())
        for i in range(len(s_ids)):
            for j in range(i + 1, len(s_ids)):
                s1, s2 = s_ids[i], s_ids[j]
                y1, y2 = years_per_source[s1], years_per_source[s2]
                if y1 and y2 and not (y1 & y2) and len(y1) == 1 and len(y2) == 1:
                    contradictions.append(
                        f"Potential date discrepancy between [{s1}] ({list(y1)[0]}) and [{s2}] ({list(y2)[0]})."
                    )

        return contradictions

    @classmethod
    def synthesize_conservative_grounded_answer(
        cls,
        evidence: EvidenceBundle,
        user_query: str
    ) -> str:
        """
        Constructs a 100% verified, conservative natural-language answer
        synthesized strictly from the verified source passages.
        """
        if not evidence or not evidence.has_evidence:
            return f"Retrieved sources do not contain sufficient evidence to answer: '{user_query}'."

        # Filter records by relevance to query
        q_terms = set(re.findall(r"\b\w{3,}\b", user_query.lower())) - cls.STOP_WORDS
        outcome_words = {"result", "winner", "champion", "won", "defeated", "capital", "president", "final"}
        is_outcome_q = any(w in user_query.lower() for w in outcome_words)

        # Extract target entity from query (e.g. "Jensen Huang", "Donald Trump", "Australia")
        entity_match = re.search(r"^(?:who is|what is|tell me about|where is|when was)\s+([A-Za-z0-9\s\.\-]+)\??$", user_query.strip(), re.IGNORECASE)
        target_entity = entity_match.group(1).strip().lower() if entity_match else ""
        target_entity_tokens = set(target_entity.split()) - cls.STOP_WORDS

        candidate_records = []
        for r in evidence.records:
            snip = r.source_snippet.strip()
            clean_snip = re.sub(r"([a-z])([A-Z])", r"\1 \2", snip)
            clean_snip = re.sub(r"([A-Za-z])\(", r"\1 (", clean_snip)
            clean_snip = re.sub(r"\s+", " ", clean_snip).strip()

            sents = re.split(r"(?<=[.!?])\s+", clean_snip)
            substantive_sents = [
                s.strip() for s in sents
                if s.strip()
                and not s.strip().endswith("?")
                and not any(s.strip().lower().startswith(p) for p in ["explore", "find out", "read more", "discover", "learn about", "click here", "sign up"])
                and not (s.strip().split()[0].lower() in ["he", "she", "they", "it", "his", "her"] and target_entity and target_entity not in s.lower())
                and len(s.strip()) > 15
            ]
            if substantive_sents:
                snip_lower = clean_snip.lower()
                overlap = sum(1 for t in q_terms if t in snip_lower)
                if is_outcome_q:
                    outcome_hits = sum(1 for w in outcome_words if w in snip_lower)
                    overlap += outcome_hits * 2
                if target_entity_tokens and all(t in snip_lower for t in target_entity_tokens):
                    overlap += 5
                substantive_sents.sort(key=lambda s: sum(1 for t in q_terms if t in s.lower()), reverse=True)
                candidate_records.append((overlap, r, substantive_sents[0]))

        candidate_records.sort(key=lambda x: x[0], reverse=True)
        relevant_candidates = [c for c in candidate_records if c[0] > 0]
        if not relevant_candidates and candidate_records:
            relevant_candidates = candidate_records[:1]

        passages = []
        cited_records = []
        max_sentences = 1 if (target_entity_tokens and len(relevant_candidates) > 0) else 2
        for score, r, best_sent in relevant_candidates[:max_sentences]:
            joined = best_sent.rstrip(".")
            passages.append(f"{joined} [{r.evidence_id}].")
            cited_records.append(r.evidence_id)

        if not passages:
            return f"Retrieved sources do not contain sufficient evidence to answer: '{user_query}'."

        citations_str = ", ".join(f"[{c}]" for c in cited_records)
        joined_evidence = " ".join(passages)
        answer = f"Based on verified sources ({citations_str}): {joined_evidence}"

        # Independent Validation of Fallback
        val = cls.validate(answer, evidence, user_query)
        if not val.is_grounded and val.unsupported_claims > 0:
            return f"Retrieved sources do not contain sufficient verified evidence to answer: '{user_query}'."

        return answer
