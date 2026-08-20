from __future__ import annotations

import unittest

from qwen_serve_lab.prometheus import (
    MetricsError,
    parse_samples,
    prefix_delta,
    prefix_snapshot,
)


class PrometheusTests(unittest.TestCase):
    def test_prefix_counters_are_summed_and_differenced(self) -> None:
        before_text = """
# TYPE vllm:prefix_cache_queries counter
vllm:prefix_cache_queries{engine="0"} 100
vllm:prefix_cache_hits{engine="0"} 20
"""
        after_text = """
vllm:prefix_cache_queries{engine="0"} 500
vllm:prefix_cache_hits{engine="0"} 180
"""

        delta = prefix_delta(
            prefix_snapshot(before_text), prefix_snapshot(after_text)
        )

        self.assertEqual(delta["query_tokens"], 400)
        self.assertEqual(delta["hit_tokens"], 160)
        self.assertEqual(delta["hit_rate_percent"], 40)

    def test_samples_with_multiple_label_series_are_summed(self) -> None:
        samples = parse_samples(
            "vllm:prefix_cache_hits{engine=\"0\"} 2\n"
            "vllm:prefix_cache_hits{engine=\"1\"} 3\n"
        )
        self.assertEqual(samples["vllm:prefix_cache_hits"], 5)

    def test_counter_reset_is_rejected(self) -> None:
        before = {"query_tokens": 100.0, "hit_tokens": 50.0}
        after = {"query_tokens": 90.0, "hit_tokens": 40.0}
        with self.assertRaises(MetricsError):
            prefix_delta(before, after)


if __name__ == "__main__":
    unittest.main()
