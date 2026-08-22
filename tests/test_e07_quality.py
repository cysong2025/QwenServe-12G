from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

from qwen_serve_lab.config import config_sha256
from qwen_serve_lab.e05_quality import load_e05_quality_dataset
from qwen_serve_lab.e07_quality import (
    compare_e07_quality,
    run_e07_quality,
    summarize_e07_human_review,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/e05_ai_infra_quality.json"


def _quality_document(state: str, degrade: bool) -> dict[str, object]:
    dataset_hash, metadata, cases = load_e05_quality_dataset(DATASET)
    labels = metadata["root_cause_labels"]
    action_labels = metadata["action_labels"]
    rows = []
    for index, case in enumerate(cases):
        wrong_root = degrade and index < 20
        wrong_actions = degrade and index < 30
        root = (
            labels[(labels.index(case["expected_root_cause"]) + 1) % len(labels)]
            if wrong_root
            else case["expected_root_cause"]
        )
        expected_actions = case["expected_actions"]
        alternatives = [item for item in action_labels if item not in expected_actions]
        actions = alternatives[:2] if wrong_actions else expected_actions
        parsed = {
            "root_cause": root,
            "actions": actions,
            "dangerous_command": False,
        }
        rows.append(
            {
                "id": case["id"],
                "incident": case["incident"],
                "prompt_sha256": case["prompt_sha256"],
                "expected_root_cause": case["expected_root_cause"],
                "expected_actions": expected_actions,
                "generated": json.dumps(parsed),
                "parsed": parsed,
                "schema_pass": True,
                "dangerous_command_detected": False,
                "error": None,
            }
        )
    return {
        "schema_version": 1,
        "kind": "e07_quality_run",
        "state": state,
        "valid": True,
        "server_profile": f"e07_{state}",
        "server_config_sha256": config_sha256(
            ROOT / f"configs/serve/e07_{state}.toml"
        ),
        "dataset_sha256": dataset_hash,
        "root_cause_labels": labels,
        "action_labels": action_labels,
        "lora_manifest_sha256": "adapter-hash" if state == "lora" else None,
        "lora_weights_sha256": "weights-hash" if state == "lora" else None,
        "results": rows,
    }


class E07QualityTests(unittest.TestCase):
    def test_quality_gain_and_blinded_review_complete_the_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_root = root / "results"
            (result_root / "base").mkdir(parents=True)
            (result_root / "lora").mkdir(parents=True)
            (result_root / "base/20260101T000000Z-e07-quality-base.json").write_text(
                json.dumps(_quality_document("base", degrade=True))
            )
            (result_root / "lora/20260101T000000Z-e07-quality-lora.json").write_text(
                json.dumps(_quality_document("lora", degrade=False))
            )
            output = root / "reports"
            _, _, passed = compare_e07_quality(result_root, output)
            self.assertTrue(passed)
            summary = json.loads((output / "quality.json").read_text())
            self.assertEqual(summary["automated_status"], "PASS")
            self.assertGreaterEqual(
                summary["lora"]["root_cause_macro_f1"]
                - summary["base"]["root_cause_macro_f1"],
                0.10,
            )

            key = json.loads((output / "human_review_key.json").read_text())
            with (output / "human_review.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                review_rows = list(csv.DictReader(handle))
            for row in review_rows:
                if key["a_state_by_id"][row["id"]] == "base":
                    row.update(
                        preferred="B", score_a_1_to_5="3", score_b_1_to_5="5"
                    )
                else:
                    row.update(
                        preferred="A", score_a_1_to_5="5", score_b_1_to_5="3"
                    )
            with (output / "human_review.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]))
                writer.writeheader()
                writer.writerows(review_rows)
            _, _, human_passed = summarize_e07_human_review(
                output / "human_review.csv",
                output / "human_review_key.json",
                output,
            )
            self.assertTrue(human_passed)
            finalized = json.loads((output / "quality.json").read_text())
            self.assertEqual(finalized["overall_status"], "PASS")

    def test_quality_runner_records_lora_adapter_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            active = root / "active.json"
            active.write_text(
                json.dumps(
                    {
                        "profile": "e07_lora",
                        "server_config_sha256": config_sha256(
                            ROOT / "configs/serve/e07_lora.toml"
                        ),
                        "pid": os.getpid(),
                        "lora_name": "ai-infra-triage-r8",
                        "lora_manifest_sha256": "adapter-hash",
                        "lora_weights_sha256": "weights-hash",
                    }
                )
            )
            _, _, cases = load_e05_quality_dataset(DATASET)
            by_prompt = {case["user_prompt"]: case for case in cases}
            requested_models: list[str] = []

            def request(
                base_url: str,
                model: str,
                system_prompt: str,
                user_prompt: str,
                seed: int,
                timeout: float,
            ) -> dict[str, object]:
                requested_models.append(model)
                case = by_prompt[user_prompt]
                content = json.dumps(
                    {
                        "root_cause": case["expected_root_cause"],
                        "actions": case["expected_actions"],
                        "dangerous_command": False,
                    }
                )
                return {
                    "choices": [
                        {
                            "message": {"content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {},
                }

            output_path, valid = run_e07_quality(
                "lora",
                DATASET,
                root / "results",
                active_server_path=active,
                request_completion=request,
            )
            result = json.loads(output_path.read_text())
            self.assertTrue(valid)
            self.assertEqual(result["completed"], 50)
            self.assertEqual(result["lora_manifest_sha256"], "adapter-hash")
            self.assertEqual(result["lora_weights_sha256"], "weights-hash")
            self.assertEqual(set(requested_models), {"ai-infra-triage-r8"})


if __name__ == "__main__":
    unittest.main()
