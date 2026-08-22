from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

from qwen_serve_lab.e05_quality import (
    compare_e05_quality,
    load_e05_quality_dataset,
    run_e05_quality,
    summarize_e05_human_review,
)
from qwen_serve_lab.config import config_sha256


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/e05_ai_infra_quality.json"


def _valid_response(*args: object) -> dict[str, object]:
    del args
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "root_cause": "cuda_oom",
                            "actions": [
                                "reduce_batch_size",
                                "enable_gradient_checkpointing",
                            ],
                            "dangerous_command": False,
                        }
                    )
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }


class E05QualityTests(unittest.TestCase):
    def test_dataset_is_balanced_and_has_fifty_cases(self) -> None:
        _, metadata, cases = load_e05_quality_dataset(DATASET)

        self.assertEqual(len(cases), 50)
        self.assertEqual(set(metadata["label_counts"].values()), {5})
        self.assertEqual(len(metadata["root_cause_labels"]), 10)

    def test_quality_runner_writes_complete_strict_schema_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            marker = root / "active.json"
            marker.write_text(
                json.dumps(
                    {
                        "profile": "e05_kv_bf16",
                        "pid": os.getpid(),
                        "server_config_sha256": config_sha256(
                            ROOT / "configs/serve/e05_kv_bf16.toml"
                        ),
                    }
                )
            )

            output_path, valid = run_e05_quality(
                state="bf16",
                dataset_path=DATASET,
                result_root=root / "results",
                active_server_path=marker,
                request_completion=_valid_response,
            )

            evidence = json.loads(output_path.read_text())
            self.assertTrue(valid)
            self.assertEqual(evidence["completed"], 50)
            self.assertTrue(all(row["schema_pass"] for row in evidence["results"]))

    def test_identical_correct_results_pass_automated_gate_and_make_blind_review(
        self,
    ) -> None:
        dataset_sha, metadata, cases = load_e05_quality_dataset(DATASET)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result_root = root / "results"
            result_paths = {}
            for state in ("bf16", "fp8"):
                state_dir = result_root / state
                state_dir.mkdir(parents=True)
                rows = []
                for case in cases:
                    parsed = {
                        "root_cause": case["expected_root_cause"],
                        "actions": case["expected_actions"],
                        "dangerous_command": False,
                    }
                    rows.append(
                        {
                            **case,
                            "generated": json.dumps(parsed),
                            "parsed": parsed,
                            "schema_pass": True,
                            "dangerous_command_detected": False,
                            "error": None,
                        }
                    )
                document = {
                    "kind": "e05_quality_run",
                    "state": state,
                    "valid": True,
                    "dataset_sha256": dataset_sha,
                    "root_cause_labels": metadata["root_cause_labels"],
                    "action_labels": metadata["action_labels"],
                    "results": rows,
                }
                path = state_dir / f"20260821-e05-quality-{state}.json"
                path.write_text(json.dumps(document))
                result_paths[state] = path

            json_path, _, passed = compare_e05_quality(
                result_root=result_root,
                output_dir=root / "report",
                bf16_result=result_paths["bf16"],
                fp8_result=result_paths["fp8"],
            )

            summary = json.loads(json_path.read_text())
            self.assertTrue(passed)
            self.assertEqual(summary["automated_status"], "PASS")
            review_path = root / "report/human_review.csv"
            with review_path.open(newline="") as handle:
                review_rows = list(csv.DictReader(handle))
            self.assertEqual(len(review_rows), 50)

            for row in review_rows:
                row["preferred"] = "TIE"
                row["score_a_1_to_5"] = "5"
                row["score_b_1_to_5"] = "5"
            with review_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]))
                writer.writeheader()
                writer.writerows(review_rows)
            _, _, human_passed = summarize_e05_human_review(
                review_path,
                root / "report/human_review_key.json",
                root / "report",
            )
            self.assertTrue(human_passed)


if __name__ == "__main__":
    unittest.main()
