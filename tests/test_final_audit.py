from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qwen_serve_lab.final_audit import audit_e01_e06, write_e01_e06_audit


ROOT = Path(__file__).resolve().parents[1]


class FinalAuditTests(unittest.TestCase):
    def test_committed_e01_e06_evidence_passes(self) -> None:
        audit = audit_e01_e06(ROOT)
        stages = {stage["id"]: stage for stage in audit["stages"]}

        self.assertEqual(audit["overall_status"], "PASS")
        self.assertEqual(audit["milestone_status"], "E01_E06_COMPLETE")
        self.assertEqual(audit["total_benchmark_runs"], 228)
        self.assertEqual(
            stages["E01"]["status"], "COMPLETE_WITH_PROTOCOL_DEVIATION"
        )
        self.assertEqual(stages["E03"]["status"], "COVERED_BY_E01")
        self.assertEqual(
            stages["E04"]["status"], "COMPLETE_WITH_LIMITATIONS"
        )
        self.assertEqual(
            stages["E05"]["status"],
            "COMPLETE_WITH_QUALITY_REGRESSION",
        )
        self.assertEqual(
            stages["E06"]["stacked_benefit_conditions"],
            ["capacity_reuse90_p1792", "reuse50_p1024"],
        )

    def test_audit_writer_emits_machine_and_human_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path, markdown_path, passed = write_e01_e06_audit(
                ROOT, Path(temp_dir)
            )
            data = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertTrue(passed)
        self.assertEqual(data["total_benchmark_runs"], 228)
        self.assertIn("E01_E06_COMPLETE", markdown)
        self.assertIn("Expected negative findings", markdown)
