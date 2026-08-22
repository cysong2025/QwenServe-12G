from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from qwen_serve_lab.e06_canary import (
    STATES,
    compare_e06_canary,
    run_e06_canary,
)
from qwen_serve_lab.results import ResultError


ROOT = Path(__file__).resolve().parents[1]


def canary_document(state: str, changed_output: bool = False) -> dict[str, object]:
    generated = "wrong" if changed_output else "expected"
    return {
        "schema_version": 1,
        "kind": "e06_correctness_canary",
        "state": state,
        "dataset_sha256": "same-dataset",
        "valid": True,
        "prefix_metrics": {
            "query_tokens": 100.0,
            "hit_tokens": 50.0,
            "hit_rate_percent": 50.0,
        },
        "results": [
            {
                "id": "case-1",
                "group": "lookup",
                "prompt_sha256": "prompt-1",
                "expected": "expected",
                "generated": generated,
                "expected_match": generated == "expected",
            },
            {
                "id": "case-2",
                "group": "lookup",
                "prompt_sha256": "prompt-2",
                "expected": "answer",
                "generated": "shared-base-error",
                "expected_match": False,
            },
        ],
    }


class E06CanaryTests(unittest.TestCase):
    def _write_documents(
        self, root: Path, changed_state: str | None = None
    ) -> dict[str, Path]:
        paths = {}
        for state in STATES:
            path = root / f"{state}.json"
            path.write_text(
                json.dumps(
                    canary_document(state, changed_output=state == changed_state)
                ),
                encoding="utf-8",
            )
            paths[state] = path
        return paths

    def test_shared_base_error_does_not_fail_configuration_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = self._write_documents(root)

            json_path, markdown_path, passed = compare_e06_canary(
                result_root=root,
                output_dir=root / "reports",
                result_paths=paths,
            )

            self.assertTrue(passed)
            summary = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["configuration_equivalence_status"], "PASS")
            self.assertEqual(summary["expected_matches"]["bt8192_off"], 1)
            self.assertIn("Base-model mistakes", markdown_path.read_text())

    def test_one_cell_output_change_fails_equivalence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = self._write_documents(root, changed_state="bt2048_on")

            _, _, passed = compare_e06_canary(
                result_root=root,
                output_dir=root / "reports",
                result_paths=paths,
            )

            self.assertFalse(passed)

    def test_runner_rejects_stale_server_config_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            marker = root / "active.json"
            marker.write_text(
                json.dumps(
                    {
                        "profile": "e06_bt8192_apc_off",
                        "server_config_sha256": "stale",
                        "pid": os.getpid(),
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ResultError):
                run_e06_canary(
                    state="bt8192_off",
                    dataset_path=ROOT / "datasets/e04_correctness_canary.json",
                    result_root=root / "results",
                    active_server_path=marker,
                )


if __name__ == "__main__":
    unittest.main()
