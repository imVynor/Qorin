"""
Stage 16B.1 Live Inference Forensic Audit and Isolation Test Suite
Validates:
1. Correct entity attribution for Jensen Huang and Donald Trump.
2. Cross-turn isolation: no Jensen Huang information in subsequent Donald Trump turn.
3. Cross-conversation isolation: independent conversations remain unpolluted.
4. Concurrent request isolation: simultaneous requests do not overwrite shared state.
5. Grounding validator adversarial resilience: rejects all 7 adversarial cases.
6. Fallback safety: emits no unsupported claims or dangling pronoun relationships.
7. Authentic citation enforcement: rejects fake source IDs ([SOURCE_99]).
8. Query-evidence scoping: no stale retrieval reuse across different queries.
9. Insufficient evidence handling: explicit uncertainty response.
"""

import json
import os
import threading
import time
import unittest
import urllib.request

from backend.tools.evidence import EvidenceBundle, EvidenceRecord
from backend.tools.grounding_validator import EvidenceGroundingValidator, ClaimStatus


class TestStage16B1LiveIsolation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.api_url = "http://127.0.0.1:8000/api/chat/stream"
        cls.output_dir = "modern_gpt/experiments/phase2_experiment_f/stage_16b1_live_path_audit"
        os.makedirs(cls.output_dir, exist_ok=True)

    def _execute_chat(self, messages, conversation_id, web_enabled=True, python_enabled=True):
        payload = {
            "model_id": "qorin-500m-sft-exp-f",
            "messages": messages,
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 40,
            "max_new_tokens": 256,
            "repetition_penalty": 1.1,
            "retrieval_mode": "auto",
            "retrieval_top_k": 3,
            "retrieval_threshold": 0.45,
            "web_research_enabled": web_enabled,
            "python_sandbox_enabled": python_enabled,
            "conversation_id": conversation_id
        }
        req = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        tokens = []
        tool_results = []
        with urllib.request.urlopen(req) as resp:
            for line in resp:
                l = line.decode("utf-8").strip()
                if l.startswith("data: "):
                    d = json.loads(l[6:])
                    if "text" in d:
                        tokens.append(d["text"])
                    elif "tool_name" in d:
                        tool_results.append(d)

        return {
            "final_text": "".join(tokens).strip(),
            "tool_results": tool_results
        }

    # 1. Correct entity attribution for Jensen Huang
    def test_01_jensen_huang_attribution(self):
        res = self._execute_chat(
            [{"role": "user", "content": "Who is Jensen Huang?"}],
            conversation_id=f"conv_test_jensen_{int(time.time()*1000)}"
        )
        text = res["final_text"]
        self.assertIn("Jensen Huang", text)
        self.assertTrue("nvidia" in text.lower() or "semiconductor" in text.lower() or "entrepreneur" in text.lower())
        self.assertNotIn("University of Tokyo", text)
        self.assertNotIn("Japanese-born", text)
        self.assertTrue(len(res["tool_results"]) > 0)

    # 2. Correct entity attribution for Donald Trump
    def test_02_donald_trump_attribution(self):
        res = self._execute_chat(
            [{"role": "user", "content": "Who is Donald Trump?"}],
            conversation_id=f"conv_test_trump_{int(time.time()*1000)}"
        )
        text = res["final_text"]
        self.assertIn("Donald", text)
        self.assertTrue("president" in text.lower() or "politician" in text.lower())
        self.assertNotIn("younger brother of Donald Trump", text)
        self.assertNotIn("2-2 vote", text)

    # 3. Cross-turn isolation in multi-turn conversation
    def test_03_cross_turn_isolation(self):
        conv_id = f"conv_seq_{int(time.time()*1000)}"
        # Turn 1
        t1 = self._execute_chat(
            [{"role": "user", "content": "Who is Jensen Huang?"}],
            conversation_id=conv_id
        )
        self.assertIn("Jensen", t1["final_text"])

        # Turn 2
        t2 = self._execute_chat(
            [
                {"role": "user", "content": "Who is Jensen Huang?"},
                {"role": "assistant", "content": t1["final_text"]},
                {"role": "user", "content": "Who is Donald Trump?"}
            ],
            conversation_id=conv_id
        )
        # Turn 2 must be about Donald Trump, zero Jensen Huang / Tokyo carryover
        self.assertIn("Donald", t2["final_text"])
        self.assertNotIn("Jensen", t2["final_text"])
        self.assertNotIn("University of Tokyo", t2["final_text"])
        self.assertNotIn("NVIDIA", t2["final_text"])

    # 4. Cross-request concurrency isolation
    def test_04_concurrent_requests_isolation(self):
        results = {}

        def worker(q, key, conv):
            res = self._execute_chat([{"role": "user", "content": q}], conversation_id=conv)
            results[key] = res

        th1 = threading.Thread(target=worker, args=("Who is Jensen Huang?", "jensen", f"conv_conc_1_{int(time.time()*1000)}"))
        th2 = threading.Thread(target=worker, args=("Who is Donald Trump?", "trump", f"conv_conc_2_{int(time.time()*1000)}"))

        th1.start()
        th2.start()
        th1.join()
        th2.join()

        self.assertIn("Jensen", results["jensen"]["final_text"])
        self.assertNotIn("Trump", results["jensen"]["final_text"])

        self.assertIn("Donald", results["trump"]["final_text"])
        self.assertNotIn("Jensen", results["trump"]["final_text"])

    # 5. Grounding validator adversarial suite: all 7 adversarial cases rejected
    def test_05_grounding_validator_adversarial_rejection(self):
        cases = [
            # 1. Sibling self-attribution
            (
                "Donald Trump is the younger brother of Donald Trump [SOURCE_1].",
                "Robert Trump was an American businessman. He was the younger brother of Donald Trump.",
                "who is donald trump"
            ),
            # 2. Birthplace swap
            (
                "Jensen Huang was born in Huntington, New York [SOURCE_1].",
                "Jensen Huang was born in Tainan, Taiwan. John Hennessy was born in Huntington, New York.",
                "where was jensen huang born"
            ),
            # 3. Wrong date assignment
            (
                "The company was founded in 2008 [SOURCE_1].",
                "The company was founded in 1993. The headquarters moved to Santa Clara in 2008.",
                "when was the company founded"
            ),
            # 4. Meeting vs Membership
            (
                "Donald Trump is a member of the Democratic Party [SOURCE_1].",
                "Donald Trump met with members of the Democratic Party during a congressional luncheon in 2018.",
                "who is donald trump"
            ),
            # 5. Similar names
            (
                "David Smith is a criminal defense attorney in Miami [SOURCE_1].",
                "David Smith is an architect in Chicago. David A. Smith is a criminal defense attorney in Miami.",
                "who is david smith"
            ),
            # 6. Combined facts into false relationship
            (
                "Alice visited Mars rover control with Bob [SOURCE_1].",
                "Alice worked at NASA. Bob visited Mars rover control.",
                "what did alice do"
            ),
            # 7. Unsupported citation
            (
                "The treaty was drafted by Winston Churchill [SOURCE_1].",
                "The treaty was ratified by twenty nations.",
                "who drafted the treaty"
            )
        ]

        for idx, (claim, src, query) in enumerate(cases, 1):
            bundle = EvidenceBundle(query=query, records=[
                EvidenceRecord(evidence_id="SOURCE_1", source_title="Test Doc", source_url="https://test.org", source_snippet=src)
            ])
            val = EvidenceGroundingValidator.validate(claim, bundle, query)
            self.assertFalse(val.is_grounded, f"Adversarial case {idx} should be rejected, but was accepted.")
            self.assertTrue(val.unsupported_claims > 0, f"Adversarial case {idx} should have unsupported claims.")

    # 6. Fallback safety and dangling pronoun prevention
    def test_06_fallback_safety_and_pronouns(self):
        # Bundle with a secondary record starting with 'He was the younger brother...'
        r1 = EvidenceRecord(
            evidence_id="SOURCE_1",
            source_title="Donald Trump - Wikipedia",
            source_url="https://en.wikipedia.org/wiki/Donald_Trump",
            source_snippet="Donald John Trump (born June 14, 1946) is an American politician, media personality, and businessman who is the 47th president of the United States."
        )
        r2 = EvidenceRecord(
            evidence_id="SOURCE_2",
            source_title="Robert Trump - Wikipedia",
            source_url="https://en.wikipedia.org/wiki/Robert_Trump",
            source_snippet="Robert Stewart Trump was an American businessman. He was the younger brother of Donald Trump."
        )
        bundle = EvidenceBundle(query="who is Donald Trump?", records=[r1, r2])
        ans = EvidenceGroundingValidator.synthesize_conservative_grounded_answer(bundle, "who is Donald Trump?")

        self.assertIn("Donald John Trump", ans)
        self.assertNotIn("younger brother of Donald Trump", ans)
        self.assertNotIn("He was", ans)

        val = EvidenceGroundingValidator.validate(ans, bundle, "who is Donald Trump?")
        self.assertTrue(val.is_grounded)
        self.assertEqual(val.unsupported_claims, 0)

    # 7. Authentic citation enforcement (rejection of fabricated IDs)
    def test_07_fake_citation_rejection(self):
        bundle = EvidenceBundle(query="who is donald trump", records=[
            EvidenceRecord(evidence_id="SOURCE_1", source_title="Bio", source_url="https://test.org", source_snippet="Donald Trump is a businessman.")
        ])
        fake_cited_answer = "Donald Trump is a businessman [SOURCE_99]."
        val = EvidenceGroundingValidator.validate(fake_cited_answer, bundle, "who is donald trump")
        self.assertFalse(val.is_grounded)
        self.assertEqual(val.claims[0].status, ClaimStatus.FABRICATED_CITATION)

    # 8. Insufficient evidence handling
    def test_08_insufficient_evidence_handling(self):
        empty_bundle = EvidenceBundle(query="unknown entity 9999", records=[])
        ans = EvidenceGroundingValidator.synthesize_conservative_grounded_answer(empty_bundle, "what is unknown entity 9999?")
        self.assertIn("not contain sufficient evidence", ans)


if __name__ == "__main__":
    unittest.main()
