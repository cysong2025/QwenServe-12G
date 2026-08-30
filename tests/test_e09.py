from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qwen_serve_lab.e09 import compare_e09_runs
from qwen_serve_lab.e09_admission import (
    E09AdmissionController,
    E09Config,
    FORMAL_PROFILES,
    POLICIES,
    build_e09_trace,
    prepare_e09_traces,
    validate_e09_trace,
)
from qwen_serve_lab.e09_readiness import write_e09_readiness_report
from qwen_serve_lab.e09_runner import summarize_e09_requests


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/admission/e09.toml"


def _synthetic_runs(config: E09Config) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    goodput = {
        "unbounded": 1.30,
        "fixed_concurrency": 1.40,
        "deadline_aware": 1.60,
    }
    fairness = {
        "unbounded": 0.86,
        "fixed_concurrency": 0.90,
        "deadline_aware": 0.88,
    }
    for profile in config.profiles:
        if not profile.formal:
            continue
        for repetition in range(1, config.repetitions + 1):
            trace = build_e09_trace(config, profile.name, repetition)
            trace_hash = trace["trace_sha256"]
            for policy in POLICIES:
                runs.append(
                    {
                        "_path": f"synthetic/{profile.name}/{policy}/r{repetition}.json",
                        "kind": "e09_admission_run",
                        "profile": profile.name,
                        "formal": True,
                        "policy": policy,
                        "repetition": repetition,
                        "seed": config.seeds[repetition - 1],
                        "experiment_config_sha256": config.source_sha256,
                        "server_config_sha256": config.server_config_sha256,
                        "trace_sha256": trace_hash,
                        "offered_duration_seconds": 180.0,
                        "summary": {
                            "arrivals": 360,
                            "admitted": 288,
                            "rejected": 72,
                            "completed": 288,
                            "failed": 0,
                            "admitted_error_rate": 0.0,
                            "request_throughput": goodput[policy],
                            "request_goodput": goodput[policy],
                            "slo_yield": goodput[policy] / 2,
                            "p95_ttft_ms": 800.0,
                            "p95_tpot_ms": 40.0,
                            "p95_e2e_ms": 5000.0,
                            "p95_arrival_lag_ms": 5.0,
                            "p95_admitted_queue_wait_ms": (
                                650.0 if policy == "deadline_aware" else 0.0
                            ),
                            "max_admitted_queue_wait_ms": (
                                790.0 if policy == "deadline_aware" else 0.0
                            ),
                            "deadline_rejections": (
                                72 if policy == "deadline_aware" else 0
                            ),
                            "class_metrics": {
                                "short": {"slo_yield": 0.8},
                                "medium": {"slo_yield": 0.75},
                                "long": {"slo_yield": 0.7},
                                "jain_slo_yield": fairness[policy],
                            },
                        },
                        "controller": {
                            "peak_active": 8,
                            "peak_inflight_tokens": 8192,
                            "peak_queue_depth": 8,
                            "backfilled": 1,
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


class E09ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = E09Config.from_file(CONFIG_PATH)

    def test_frozen_config_has_independent_holdouts_and_27_runs(self) -> None:
        formal = [profile for profile in self.config.profiles if profile.formal]
        self.assertEqual(self.config.policies, POLICIES)
        self.assertEqual(tuple(profile.name for profile in formal), FORMAL_PROFILES)
        self.assertEqual(len(formal) * len(POLICIES) * self.config.repetitions, 27)
        self.assertNotIn(self.config.calibration_seed, self.config.seeds)
        self.assertEqual(self.config.deadline_max_inflight, 8)
        self.assertEqual(self.config.deadline_max_queue_wait_ms, 800)

    def test_trace_is_deterministic_authenticated_and_seed_separated(self) -> None:
        first = build_e09_trace(self.config, "holdout_periodic_burst", 1)
        second = build_e09_trace(self.config, "holdout_periodic_burst", 1)
        calibration = build_e09_trace(self.config, "calibration_burst", 1)
        self.assertEqual(first["trace_sha256"], second["trace_sha256"])
        self.assertNotEqual(first["seed"], calibration["seed"])
        validate_e09_trace(first, self.config)
        counts = {name: 0 for name in ("short", "medium", "long")}
        for request in first["requests"]:
            counts[request["workload"]] += 1
        total = sum(counts.values())
        self.assertLessEqual(abs(counts["short"] - total * 0.5), 10)
        self.assertLessEqual(abs(counts["medium"] - total * 0.3), 10)
        self.assertLessEqual(abs(counts["long"] - total * 0.2), 10)

    def test_trace_preparation_writes_one_calibration_and_nine_holdouts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = prepare_e09_traces(CONFIG_PATH, temporary)
            self.assertEqual(len(paths), 10)
            payload = json.loads(paths[0].read_text())
            validate_e09_trace(payload, self.config)

    def test_fixed_and_deadline_dispatch_ceilings_are_independent(self) -> None:
        fixed = E09AdmissionController("fixed_concurrency", self.config)
        fixed_leases = [
            fixed.try_acquire(f"fixed-{index}", "short", 256)
            for index in range(5)
        ]
        self.assertTrue(all(fixed_leases[:4]))
        self.assertIsNone(fixed_leases[4])
        self.assertEqual(fixed.snapshot()["rejected"], 1)
        for lease in fixed_leases[:4]:
            assert lease is not None
            fixed.release(lease, slo_met=True)

        deadline = E09AdmissionController("deadline_aware", self.config)
        deadline_leases = [
            deadline.try_acquire(f"deadline-{index}", "short", 256)
            for index in range(9)
        ]
        self.assertTrue(all(deadline_leases[:8]))
        self.assertIsNone(deadline_leases[8])
        self.assertEqual(deadline.snapshot()["rejected"], 0)
        deadline.record_deadline_expiry()
        self.assertEqual(deadline.snapshot()["deadline_expired"], 1)
        for lease in deadline_leases[:8]:
            assert lease is not None
            deadline.release(lease, slo_met=True)

    def test_deadline_token_ceiling_can_backpressure_long_requests(self) -> None:
        controller = E09AdmissionController("deadline_aware", self.config)
        leases = [
            controller.try_acquire(f"long-{index}", "long", 2304)
            for index in range(4)
        ]
        self.assertTrue(all(leases[:3]))
        self.assertIsNone(leases[3])
        for lease in leases[:3]:
            assert lease is not None
            controller.release(lease, slo_met=True)

    def test_summary_counts_deadline_expiry_as_zero_yield(self) -> None:
        results = [
            {
                "id": "short-ok",
                "workload": "short",
                "admitted": True,
                "status": "completed",
                "slo_met": True,
                "ttft_ms": 700.0,
                "tpot_ms": 20.0,
                "e2e_ms": 2000.0,
                "dispatch_lag_ms": 2.0,
                "arrival_lag_ms": 2.0,
                "queue_wait_ms": 500.0,
                "rejection_reason": None,
            },
            {
                "id": "long-expired",
                "workload": "long",
                "admitted": False,
                "status": "rejected",
                "slo_met": False,
                "dispatch_lag_ms": 3.0,
                "arrival_lag_ms": 3.0,
                "queue_wait_ms": 603.0,
                "rejection_reason": "queue_deadline_expired",
            },
        ]
        summary = summarize_e09_requests(results, 10.0)
        self.assertEqual(summary["arrivals"], 2)
        self.assertEqual(summary["deadline_rejections"], 1)
        self.assertAlmostEqual(summary["request_goodput"], 0.1)
        self.assertEqual(summary["class_metrics"]["long"]["slo_yield"], 0.0)
        self.assertEqual(summary["max_admitted_queue_wait_ms"], 500.0)

    def test_complete_synthetic_matrix_passes_frozen_positive_gate(self) -> None:
        flat, comparison, passed = compare_e09_runs(
            _synthetic_runs(self.config), self.config
        )
        self.assertTrue(passed)
        self.assertEqual(len(flat), 27)
        gated = [row for row in comparison if row["gate_scope"] == "GATED_BURST"]
        self.assertEqual(len(gated), 2)
        self.assertTrue(all(row["scientific_gate"] == "PASS" for row in gated))

    def test_trace_mismatch_cannot_pass_as_paired_evidence(self) -> None:
        runs = _synthetic_runs(self.config)
        target = next(
            run
            for run in runs
            if run["profile"] == "holdout_shock_burst"
            and run["policy"] == "deadline_aware"
            and run["repetition"] == 1
        )
        target["trace_sha256"] = "different"
        _, comparison, passed = compare_e09_runs(runs, self.config)
        self.assertFalse(passed)
        shock = next(
            row for row in comparison if row["profile"] == "holdout_shock_burst"
        )
        self.assertEqual(shock["evidence"], "INCOMPLETE")
        self.assertEqual(shock["scientific_gate"], "FAIL")

    def test_insufficient_actual_headroom_cannot_pass(self) -> None:
        runs = _synthetic_runs(self.config)
        for run in runs:
            if run["profile"] == "holdout_periodic_burst":
                run["summary"]["arrivals"] = 250
        _, comparison, passed = compare_e09_runs(runs, self.config)
        self.assertFalse(passed)
        periodic = next(
            row for row in comparison if row["profile"] == "holdout_periodic_burst"
        )
        self.assertEqual(periodic["offered_headroom_gate"], "FAIL")

    def test_readiness_is_explicitly_not_gpu_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            json_path, _, passed = write_e09_readiness_report(
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
