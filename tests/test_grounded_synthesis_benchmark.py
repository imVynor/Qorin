"""
QORIN Stage 16B: 8-Case Evidence-Grounded Synthesis Benchmark
Tests:
Case A: Who is Donald Trump?
Case B: Who is the current president of the United States?
Case C: What is the capital of Australia?
Case D: What was the result of the 2022 FIFA World Cup?
Case E: Compare two sources that provide contradictory information.
Case F: Answer a question where the retrieved evidence is insufficient.
Case G: A factual question with several irrelevant retrieved passages.
Case H: A factual question with no RAG chunks but successful Web retrieval.
"""

import json
import os
import time
import unittest
import urllib.request

from backend.tools.evidence import EvidenceBundle, EvidenceRecord
from backend.tools.grounding_validator import EvidenceGroundingValidator, ClaimStatus
from backend.tools.retrieval_providers import UnifiedWebResearcher


class TestEvidenceGroundedSynthesis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.web = UnifiedWebResearcher()
        cls.output_dir = "modern_gpt/experiments/phase2_experiment_f/stage_16b_evidence_grounding"
        os.makedirs(cls.output_dir, exist_ok=True)
        cls.benchmark_results = {}

    def _execute_chat_stream(self, prompt: str, override: str = "WEB") -> dict:
        t0 = time.perf_counter()
        payload = {
            "model_id": "qorin-500m-sft-exp-f",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 40,
            "max_new_tokens": 256,
            "repetition_penalty": 1.1,
            "web_research_enabled": True,
            "python_sandbox_enabled": False,
            "manual_tool_override": override,
            "conversation_id": f"conv_bench_{int(time.time()*1000)}"
        }
        req = urllib.request.Request(
            "http://127.0.0.1:8000/api/chat/stream",
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
                    elif "tool_name" in d and d.get("status") == "completed":
                        tool_results.append(d)

        dur_ms = (time.perf_counter() - t0) * 1000.0
        final_text = "".join(tokens).strip()
        return {
            "prompt": prompt,
            "final_text": final_text,
            "tool_results": tool_results,
            "latency_ms": round(dur_ms, 2)
        }

    # Case A: Who is Donald Trump?
    def test_case_a_donald_trump(self):
        res = self._execute_chat_stream("who is donald trump")
        self.assertIn("Donald", res["final_text"])
        self.assertIn("president", res["final_text"].lower())
        self.assertNotIn("2-2 vote", res["final_text"])
        self.assertNotIn("seventh president", res["final_text"])
        self.assertNotIn("176 million", res["final_text"])

        # Check tool results have sources
        self.assertTrue(len(res["tool_results"]) > 0)
        tool_res = res["tool_results"][0]
        records = tool_res.get("output_payload", {}).get("records", [])
        eb = EvidenceBundle(
            query="who is donald trump",
            records=[EvidenceRecord.from_dict(r) for r in records]
        )
        val = EvidenceGroundingValidator.validate(res["final_text"], eb, "who is donald trump")
        self.assertTrue(val.is_grounded)
        self.assertEqual(val.unsupported_claims, 0)

        self.__class__.benchmark_results["case_a"] = {
            "prompt": "who is donald trump",
            "final_text": res["final_text"],
            "latency_ms": res["latency_ms"],
            "tool_results": tool_res,
            "validation": val.to_dict()
        }

    # Case B: Who is the current president of the United States?
    def test_case_b_current_president(self):
        res = self._execute_chat_stream("who is the current president of the United States?")
        self.assertTrue(len(res["final_text"]) > 10)
        self.assertIn("president", res["final_text"].lower())
        tool_res = res["tool_results"][0] if res["tool_results"] else {}
        records = tool_res.get("output_payload", {}).get("records", [])
        eb = EvidenceBundle(
            query="who is the current president of the United States?",
            records=[EvidenceRecord.from_dict(r) for r in records]
        )
        val = EvidenceGroundingValidator.validate(res["final_text"], eb, "who is the current president of the United States?")
        self.assertTrue(val.is_grounded)

        self.__class__.benchmark_results["case_b"] = {
            "prompt": "who is the current president of the United States?",
            "final_text": res["final_text"],
            "latency_ms": res["latency_ms"],
            "tool_results": tool_res,
            "validation": val.to_dict()
        }

    # Case C: What is the capital of Australia?
    def test_case_c_capital_of_australia(self):
        res = self._execute_chat_stream("what is the capital of Australia?")
        self.assertIn("Canberra", res["final_text"])
        tool_res = res["tool_results"][0] if res["tool_results"] else {}
        records = tool_res.get("output_payload", {}).get("records", [])
        eb = EvidenceBundle(
            query="what is the capital of Australia?",
            records=[EvidenceRecord.from_dict(r) for r in records]
        )
        val = EvidenceGroundingValidator.validate(res["final_text"], eb, "what is the capital of Australia?")
        self.assertTrue(val.is_grounded)

        self.__class__.benchmark_results["case_c"] = {
            "prompt": "what is the capital of Australia?",
            "final_text": res["final_text"],
            "latency_ms": res["latency_ms"],
            "tool_results": tool_res,
            "validation": val.to_dict()
        }

    # Case D: What was the result of the 2022 FIFA World Cup?
    def test_case_d_fifa_world_cup(self):
        res = self._execute_chat_stream("what was the result of the 2022 FIFA World Cup?")
        self.assertIn("Argentina", res["final_text"])
        tool_res = res["tool_results"][0] if res["tool_results"] else {}
        records = tool_res.get("output_payload", {}).get("records", [])
        eb = EvidenceBundle(
            query="what was the result of the 2022 FIFA World Cup?",
            records=[EvidenceRecord.from_dict(r) for r in records]
        )
        val = EvidenceGroundingValidator.validate(res["final_text"], eb, "what was the result of the 2022 FIFA World Cup?")
        self.assertTrue(val.is_grounded)

        self.__class__.benchmark_results["case_d"] = {
            "prompt": "what was the result of the 2022 FIFA World Cup?",
            "final_text": res["final_text"],
            "latency_ms": res["latency_ms"],
            "tool_results": tool_res,
            "validation": val.to_dict()
        }

    # Case E: Compare two sources that provide contradictory information
    def test_case_e_contradictory_sources(self):
        # Construct synthetic contradictory bundle
        r1 = EvidenceRecord(
            evidence_id="SOURCE_1",
            source_title="Historical Archive A",
            source_url="https://history-a.org/event",
            source_snippet="The treaty was signed in Paris in 1919."
        )
        r2 = EvidenceRecord(
            evidence_id="SOURCE_2",
            source_title="Historical Archive B",
            source_url="https://history-b.org/event",
            source_snippet="The treaty was signed in London in 1920."
        )
        bundle = EvidenceBundle(query="where and when was the treaty signed?", records=[r1, r2])
        contradictions = EvidenceGroundingValidator._detect_source_contradictions(bundle)
        self.assertTrue(len(contradictions) > 0)
        self.assertIn("Potential date discrepancy", contradictions[0])

        ans = EvidenceGroundingValidator.synthesize_conservative_grounded_answer(bundle, "where and when was the treaty signed?")
        self.assertIn("[SOURCE_1]", ans)
        self.assertIn("[SOURCE_2]", ans)

        val = EvidenceGroundingValidator.validate(ans, bundle, "where and when was the treaty signed?")
        self.assertTrue(val.is_grounded)

        self.__class__.benchmark_results["case_e"] = {
            "prompt": "where and when was the treaty signed?",
            "contradictions_detected": contradictions,
            "final_text": ans,
            "validation": val.to_dict()
        }

    # Case F: Answer a question where the retrieved evidence is insufficient
    def test_case_f_insufficient_evidence(self):
        empty_bundle = EvidenceBundle(query="unknown query xyz12345", records=[])
        ans = EvidenceGroundingValidator.synthesize_conservative_grounded_answer(empty_bundle, "what is xyz12345?")
        self.assertIn("not contain sufficient evidence", ans)

        self.__class__.benchmark_results["case_f"] = {
            "prompt": "what is xyz12345?",
            "final_text": ans,
            "status": "PASSED - Acknowledged insufficient evidence explicitly"
        }

    # Case G: A factual question with several irrelevant retrieved passages
    def test_case_g_irrelevant_passages_filtering(self):
        r_good = EvidenceRecord(
            evidence_id="SOURCE_1",
            source_title="Canberra Geography",
            source_url="https://geo.org/canberra",
            source_snippet="Canberra is the federal capital city of the Commonwealth of Australia."
        )
        r_noise_1 = EvidenceRecord(
            evidence_id="SOURCE_2",
            source_title="Baking Recipes",
            source_url="https://recipes.org/cake",
            source_snippet="Preheat the oven to 350 degrees F and whisk flour and sugar."
        )
        r_noise_2 = EvidenceRecord(
            evidence_id="SOURCE_3",
            source_title="Used Car Listings",
            source_url="https://cars.org/listing",
            source_snippet="2015 Sedan in good condition with low mileage and new tires."
        )
        noisy_bundle = EvidenceBundle(query="capital of Australia", records=[r_noise_1, r_good, r_noise_2])
        ans = EvidenceGroundingValidator.synthesize_conservative_grounded_answer(noisy_bundle, "capital of Australia")
        self.assertIn("Canberra", ans)
        self.assertNotIn("flour", ans)
        self.assertNotIn("Sedan", ans)

        val = EvidenceGroundingValidator.validate(ans, noisy_bundle, "capital of Australia")
        self.assertTrue(val.is_grounded)

        self.__class__.benchmark_results["case_g"] = {
            "prompt": "capital of Australia",
            "final_text": ans,
            "status": "PASSED - Irrelevant passages filtered, grounded assertion retained",
            "validation": val.to_dict()
        }

    # Case H: A factual question with no RAG chunks but successful Web retrieval
    def test_case_h_no_rag_successful_web(self):
        # Query that bypasses internal RAG
        res = self._execute_chat_stream("who was the first person to walk on the moon?")
        self.assertIn("Armstrong", res["final_text"])
        self.assertTrue(len(res["tool_results"]) > 0)

        tool_res = res["tool_results"][0]
        records = tool_res.get("output_payload", {}).get("records", [])
        eb = EvidenceBundle(
            query="who was the first person to walk on the moon?",
            records=[EvidenceRecord.from_dict(r) for r in records]
        )
        val = EvidenceGroundingValidator.validate(res["final_text"], eb, "who was the first person to walk on the moon?")
        self.assertTrue(val.is_grounded)

        self.__class__.benchmark_results["case_h"] = {
            "prompt": "who was the first person to walk on the moon?",
            "final_text": res["final_text"],
            "latency_ms": res["latency_ms"],
            "validation": val.to_dict()
        }

    @classmethod
    def tearDownClass(cls):
        out_file = os.path.join(cls.output_dir, "regression_results.json")
        with open(out_file, "w") as f:
            json.dump(cls.benchmark_results, f, indent=2)
        print(f"Regression results saved to {out_file}")


if __name__ == "__main__":
    unittest.main()
