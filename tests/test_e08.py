from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qwen_serve_lab.e08 import compare_e08_runs
from qwen_serve_lab.e08_admission import (
    AdmissionController,
    E08Config,
    POLICIES,
    build_e08_trace,
    prepare_e08_traces,
    validate_e08_trace,
)
from qwen_serve_lab.e08_readiness import write_e08_readiness_report
from qwen_serve_lab.e08_runner import (
    build_prompt_token_ids,
    request_streaming_completion,
    summarize_e08_requests,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/admission/e08.toml"


def _synthetic_runs(config: E08Config) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    goodput = {
        "unbounded": 1.00,
        "fixed_concurrency": 1.05,
        "token_aware": 1.20,
    }
    fairness = {
        "unbounded": 0.75,
        "fixed_concurrency": 0.88,
        "token_aware": 0.90,
    }
    for profile in (item for item in config.profiles if item.formal):
        for policy in POLICIES:
            for repetition in range(1, config.repetitions + 1):
                trace_hash = f"trace-{profile.name}-{repetition}"
                runs.append(
                    {
                        "kind": "e08_admission_run",
                        "_path": f"{profile.name}-{policy}-r{repetition}.json",
                        "profile": profile.name,
                        "formal": True,
                        "policy": policy,
                        "repetition": repetition,
                        "seed": config.seeds[repetition - 1],
                        "experiment_config_sha256": config.source_sha256,
                        "server_profile": config.server_profile,
                        "server_config_sha256": config.server_config_sha256,
                        "trace_sha256": trace_hash,
                        "offered_duration_seconds": 180.0,
                        "summary": {
                            "arrivals": 360,
                            "admitted": 300,
                            "rejected": 60,
                            "completed": 300,
                            "failed": 0,
                            "admitted_error_rate": 0.0,
                            "request_throughput": goodput[policy],
                            "request_goodput": goodput[policy],
                            "slo_yield": 0.6,
                            "p95_ttft_ms": 800.0,
                            "p95_tpot_ms": 40.0,
                            "p95_e2e_ms": 5000.0,
                            "p95_dispatch_lag_ms": 5.0,
                            "class_metrics": {
                                "short": {"slo_yield": 0.6},
                                "medium": {"slo_yield": 0.6},
                                "long": {"slo_yield": 0.6},
                                "jain_slo_yield": fairness[policy],
                            },
                        },
                        "controller": {
                            "peak_active": 8,
                            "peak_inflight_tokens": 4096,
                            "current_token_budget": 4096,
                            "budget_decreases": 1,
                            "budget_increases": 1,
                        },
                        "telemetry": {
                            "peak_memory_used_mib": 11000.0,
                            "mean_gpu_utilization_percent": 90.0,
                            "max_temperature_c": 65.0,
                        },
                        "valid": True,
                    }
                )
    return runs


class E08ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = E08Config.from_file(CONFIG_PATH)

    def test_frozen_config_has_three_policies_and_27_formal_runs(self) -> None:
        formal = [profile for profile in self.config.profiles if profile.formal]
        self.assertEqual(self.config.policies, POLICIES)
        self.assertEqual(len(formal), 3)
        self.assertTrue(all(profile.duration_seconds >= 180 for profile in formal))
        self.assertEqual(len(formal) * len(POLICIES) * self.config.repetitions, 27)
        self.assertEqual(self.config.server_profile, "e06_bt2048_apc_off")

    def test_trace_is_deterministic_hash_authenticated_and_mixed(self) -> None:
        first = build_e08_trace(self.config, "mixed_overload", 1)
        second = build_e08_trace(self.config, "mixed_overload", 1)
        self.assertEqual(first["trace_sha256"], second["trace_sha256"])
        validate_e08_trace(first, self.config)
        counts = {name: 0 for name in ("short", "medium", "long")}
        for request in first["requests"]:
            counts[request["workload"]] += 1
        total = sum(counts.values())
        self.assertLessEqual(abs(counts["short"] - total * 0.5), 10)
        self.assertLessEqual(abs(counts["medium"] - total * 0.3), 10)
        self.assertLessEqual(abs(counts["long"] - total * 0.2), 10)

    def test_trace_preparation_writes_every_profile_and_repetition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = prepare_e08_traces(CONFIG_PATH, temporary)
            self.assertEqual(len(paths), len(self.config.profiles) * 3)
            payload = json.loads(paths[0].read_text())
            validate_e08_trace(payload, self.config)

    def test_fixed_concurrency_rejects_fifth_active_request(self) -> None:
        controller = AdmissionController("fixed_concurrency", self.config)
        leases = [
            controller.try_acquire(f"request-{index}", "short", 256)
            for index in range(5)
        ]
        self.assertTrue(all(leases[:4]))
        self.assertIsNone(leases[4])
        for lease in leases[:4]:
            assert lease is not None
            controller.release(lease, slo_met=True)
        self.assertEqual(controller.snapshot()["active"], 0)

    def test_token_controller_uses_budget_and_aimd_feedback(self) -> None:
        controller = AdmissionController("token_aware", self.config)
        long = controller.try_acquire("long", "long", 2304)
        medium = controller.try_acquire("medium", "medium", 768)
        overflow = controller.try_acquire("overflow", "long", 2304)
        self.assertIsNotNone(long)
        self.assertIsNotNone(medium)
        self.assertIsNone(overflow)
        assert long is not None and medium is not None
        controller.release(long, slo_met=False)
        decreased = controller.snapshot()["current_token_budget"]
        self.assertLess(decreased, self.config.initial_token_budget)
        controller.release(medium, slo_met=True)
        for index in range(self.config.success_window - 1):
            lease = controller.try_acquire(f"success-{index}", "short", 256)
            self.assertIsNotNone(lease)
            assert lease is not None
            controller.release(lease, slo_met=True)
        self.assertGreater(controller.snapshot()["current_token_budget"], decreased)

    def test_request_summary_counts_rejection_as_zero_slo_yield(self) -> None:
        results = [
            {
                "id": "short-ok",
                "workload": "short",
                "admitted": True,
                "status": "completed",
                "slo_met": True,
                "ttft_ms": 100.0,
                "tpot_ms": 10.0,
                "e2e_ms": 1000.0,
                "dispatch_lag_ms": 1.0,
            },
            {
                "id": "medium-reject",
                "workload": "medium",
                "admitted": False,
                "status": "rejected",
                "slo_met": False,
                "dispatch_lag_ms": 1.0,
            },
            {
                "id": "long-ok",
                "workload": "long",
                "admitted": True,
                "status": "completed",
                "slo_met": True,
                "ttft_ms": 900.0,
                "tpot_ms": 40.0,
                "e2e_ms": 5000.0,
                "dispatch_lag_ms": 2.0,
            },
        ]
        summary = summarize_e08_requests(results, 10.0)
        self.assertEqual(summary["arrivals"], 3)
        self.assertEqual(summary["rejected"], 1)
        self.assertAlmostEqual(summary["request_goodput"], 0.2)
        self.assertEqual(summary["class_metrics"]["medium"]["slo_yield"], 0.0)

    def test_prompt_tokens_are_exact_and_exclude_special_ids(self) -> None:
        class Tokenizer:
            vocab_size = 2048
            all_special_ids = [300, 301]

        first = build_prompt_token_ids(Tokenizer(), 512, 123)
        second = build_prompt_token_ids(Tokenizer(), 512, 123)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 512)
        self.assertFalse({300, 301} & set(first))

    def test_streaming_client_extracts_usage_ttft_and_tpot(self) -> None:
        class Response:
            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def __iter__(self):
                return iter(
                    [
                        b'data: {"choices":[{"text":"a","finish_reason":null}]}\n',
                        b'data: {"choices":[{"text":"b","finish_reason":"length"}]}\n',
                        b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n',
                        b'data: [DONE]\n',
                    ]
                )

        with patch("qwen_serve_lab.e08_runner.urlopen", return_value=Response()):
            result = request_streaming_completion(
                "http://127.0.0.1:8000",
                "model",
                [1, 2, 3],
                2,
                123,
                10.0,
            )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["completion_tokens"], 2)
        self.assertEqual(result["prompt_tokens"], 3)
        self.assertEqual(result["finish_reason"], "length")
        self.assertGreaterEqual(result["ttft_ms"], 0)
        self.assertGreaterEqual(result["tpot_ms"], 0)

    def test_complete_synthetic_matrix_passes_frozen_gate(self) -> None:
        flat, comparison, passed = compare_e08_runs(
            _synthetic_runs(self.config), self.config
        )
        self.assertTrue(passed)
        self.assertEqual(len(flat), 27)
        overload = [row for row in comparison if row["gate_scope"] == "OVERLOAD"]
        self.assertEqual(len(overload), 2)
        self.assertTrue(all(row["scientific_gate"] == "PASS" for row in overload))

    def test_trace_mismatch_cannot_pass_as_paired_evidence(self) -> None:
        runs = _synthetic_runs(self.config)
        target = next(
            run
            for run in runs
            if run["profile"] == "mixed_burst"
            and run["policy"] == "token_aware"
            and run["repetition"] == 1
        )
        target["trace_sha256"] = "different"
        _, comparison, passed = compare_e08_runs(runs, self.config)
        self.assertFalse(passed)
        burst = next(row for row in comparison if row["profile"] == "mixed_burst")
        self.assertEqual(burst["evidence"], "INCOMPLETE")
        self.assertEqual(burst["scientific_gate"], "FAIL")

    def test_readiness_is_explicitly_not_gpu_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            json_path, _, passed = write_e08_readiness_report(
                ROOT, CONFIG_PATH, temporary
            )
            report = json.loads(json_path.read_text())
            self.assertTrue(passed)
            self.assertEqual(report["status"], "READY_FOR_GPU")
            self.assertEqual(report["gpu_execution"], "DEFERRED")
            self.assertEqual(report["scientific_result"], "NOT_RUN")
            self.assertEqual(report["planned_formal_runs"], 27)


if __name__ == "__main__":
    unittest.main()
