from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from qwen_serve_lab.e04_canary import compare_e04_canary, run_e04_canary


def _dataset() -> dict[str, object]:
    return {
        "schema_version": 1,
        "groups": [
            {
                "name": "lookup",
                "shared_prefix": "Return only the requested value. alpha=A beta=B",
                "cases": [
                    {"id": "one", "question": "Value for alpha?", "expected": "A"},
                    {"id": "two", "question": "Value for beta?", "expected": "B"},
                ],
            }
        ],
    }


def _completion(
    base_url: str,
    served_model_name: str,
    prompt: str,
    timeout_seconds: float,
) -> dict[str, object]:
    del base_url, served_model_name, timeout_seconds
    text = "A" if "alpha?" in prompt else "B"
    return {
        "choices": [{"text": text, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1},
    }


class E04CanaryTests(unittest.TestCase):
    def test_fixed_canary_passes_with_identical_correct_outputs_and_cache_hits(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_path = root / "dataset.json"
            dataset_path.write_text(json.dumps(_dataset()))
            marker_path = root / "active.json"
            result_root = root / "results"

            def run_state(state: str) -> Path:
                marker_path.write_text(
                    json.dumps(
                        {
                            "profile": f"e04_prefix_{state}_bf16",
                            "pid": os.getpid(),
                            "server_config_sha256": state * 32,
                        }
                    )
                )
                samples = iter(
                    (
                        "vllm:prefix_cache_queries 0\n"
                        "vllm:prefix_cache_hits 0\n",
                        "vllm:prefix_cache_queries 100\n"
                        f"vllm:prefix_cache_hits {80 if state == 'on' else 0}\n",
                    )
                )
                output_path, valid = run_e04_canary(
                    state=state,
                    dataset_path=dataset_path,
                    result_root=result_root,
                    active_server_path=marker_path,
                    request_completion=_completion,
                    metrics_fetcher=lambda base_url: next(samples),
                )
                self.assertTrue(valid)
                return output_path

            off_path = run_state("off")
            on_path = run_state("on")
            json_path, markdown_path, passed = compare_e04_canary(
                result_root=result_root,
                output_dir=root / "report",
                off_result=off_path,
                on_result=on_path,
            )

            self.assertTrue(passed)
            self.assertTrue(json_path.is_file())
            self.assertIn("Status: **PASS**", markdown_path.read_text())

    def test_output_mismatch_fails_canary_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            common = {
                "kind": "e04_correctness_canary",
                "dataset_sha256": "a" * 64,
                "valid": True,
            }
            off = {
                **common,
                "state": "off",
                "results": [
                    {
                        "id": "one",
                        "group": "lookup",
                        "prompt_sha256": "b" * 64,
                        "generated": "A",
                        "expected_match": True,
                    }
                ],
            }
            on = {
                **common,
                "state": "on",
                "prefix_metrics": {
                    "query_tokens": 100,
                    "hit_tokens": 80,
                    "hit_rate_percent": 80,
                },
                "results": [
                    {
                        "id": "one",
                        "group": "lookup",
                        "prompt_sha256": "b" * 64,
                        "generated": "B",
                        "expected_match": False,
                    }
                ],
            }
            off_path = root / "off.json"
            on_path = root / "on.json"
            off_path.write_text(json.dumps(off))
            on_path.write_text(json.dumps(on))

            _, _, passed = compare_e04_canary(
                result_root=root,
                output_dir=root / "report",
                off_result=off_path,
                on_result=on_path,
            )

            self.assertFalse(passed)


if __name__ == "__main__":
    unittest.main()
