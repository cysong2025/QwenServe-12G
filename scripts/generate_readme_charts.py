#!/usr/bin/env python3
"""Generate deterministic README SVGs from committed experiment reports."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "assets"

INK = "#0f172a"
MUTED = "#64748b"
GRID = "#e2e8f0"
PANEL = "#f8fafc"
BLUE = "#3b82f6"
GREEN = "#10b981"
AMBER = "#f59e0b"
RED = "#ef4444"
SLATE = "#64748b"


def _text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 14,
    weight: int = 400,
    fill: str = INK,
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:g}" y="{y:g}" font-family="Arial" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}">{html.escape(value)}</text>'
    )


def _rect(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str,
    radius: float = 0,
    stroke: str | None = None,
    stroke_width: float = 1,
) -> str:
    stroke_attr = (
        f' stroke="{stroke}" stroke-width="{stroke_width:g}"' if stroke else ""
    )
    return (
        f'<rect x="{x:g}" y="{y:g}" width="{width:g}" height="{height:g}" '
        f'rx="{radius:g}" fill="{fill}"{stroke_attr}/>'
    )


def _line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str = GRID,
    width: float = 1,
    dash: str | None = None,
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" '
        f'stroke="{stroke}" stroke-width="{width:g}"{dash_attr}/>'
    )


def _svg(width: int, height: int, title: str, body: list[str]) -> str:
    escaped_title = html.escape(title)
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
                f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
                f'aria-labelledby="title desc">'
            ),
            f"<title id=\"title\">{escaped_title}</title>",
            (
                '<desc id="desc">Generated from committed QwenServe-12G '
                "experiment reports.</desc>"
            ),
            "<style>",
            "text { font-family: Helvetica, Arial, sans-serif; }",
            "</style>",
            _rect(0, 0, width, height, fill="#ffffff", radius=16),
            *body,
            "</svg>",
            "",
        ]
    )


def _load_csv(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def render_architecture() -> str:
    width, height = 1040, 390
    body: list[str] = [
        _text(40, 42, "QwenServe-12G experiment and evidence flow", size=24, weight=700),
        _text(
            40,
            68,
            "Mac control plane + Windows/WSL2 GPU execution, connected by versioned evidence",
            size=13,
            fill=MUTED,
        ),
    ]

    boxes = [
        (40, 108, 210, 90, "Mac development", "Python · TOML · tests · Git", BLUE),
        (300, 108, 210, 90, "WSL2 runner", "matrix · traces · admission", AMBER),
        (560, 108, 210, 90, "vLLM service", "OpenAI API · BF16/FP8 · LoRA", GREEN),
        (820, 108, 180, 90, "RTX 5070", "12 GB · CUDA · telemetry", "#8b5cf6"),
    ]
    for x, y, w, h, title, subtitle, color in boxes:
        body.append(_rect(x, y, w, h, fill=PANEL, radius=12, stroke=GRID))
        body.append(_rect(x, y, 6, h, fill=color, radius=3))
        body.append(_text(x + 22, y + 35, title, size=17, weight=700))
        body.append(_text(x + 22, y + 62, subtitle, size=12, fill=MUTED))

    for start, end in [((250, 153), (300, 153)), ((510, 153), (560, 153)), ((770, 153), (820, 153))]:
        body.append(_line(*start, *end, stroke=MUTED, width=2))
        body.append(
            f'<path d="M {end[0]-8} {end[1]-5} L {end[0]} {end[1]} L {end[0]-8} {end[1]+5}" '
            f'fill="none" stroke="{MUTED}" stroke-width="2"/>'
        )

    lower = [
        (70, 260, 250, "Metrics", "TTFT · TPOT · goodput · VRAM"),
        (395, 260, 250, "Evidence", "CSV/JSON · hashes · exact traces"),
        (720, 260, 250, "Decision gates", "SLO · quality · fairness · limits"),
    ]
    for x, y, w, title, subtitle in lower:
        body.append(_rect(x, y, w, 82, fill="#f1f5f9", radius=12, stroke=GRID))
        body.append(_text(x + 20, y + 31, title, size=16, weight=700))
        body.append(_text(x + 20, y + 56, subtitle, size=12, fill=MUTED))
    body.append(_line(910, 198, 910, 230, stroke=MUTED, width=2))
    body.append(_line(910, 230, 195, 230, stroke=MUTED, width=2))
    body.append(_line(195, 230, 195, 260, stroke=MUTED, width=2))
    body.append(_line(320, 301, 395, 301, stroke=MUTED, width=2))
    body.append(_line(645, 301, 720, 301, stroke=MUTED, width=2))
    return _svg(width, height, "QwenServe-12G architecture", body)


def render_e09_goodput() -> str:
    report = _load_json("reports/e09_deadline_admission/final.json")
    profiles = report["profiles"]
    width, height = 980, 470
    left, top, chart_w, chart_h = 76, 96, 850, 270
    max_value = 2.2
    policies = [
        ("unbounded_goodput", "Unbounded", SLATE),
        ("fixed_goodput", "Fixed C4", BLUE),
        ("deadline_goodput", "Deadline-aware", GREEN),
    ]
    names = ["Nominal", "Periodic burst", "Shock burst"]
    body = [
        _text(40, 42, "E09 — SLO goodput under open-arrival traffic", size=24, weight=700),
        _text(
            40,
            68,
            "Median of 3 frozen holdout repetitions · higher is better",
            size=13,
            fill=MUTED,
        ),
    ]
    for tick in [0, 0.5, 1.0, 1.5, 2.0]:
        y = top + chart_h - tick / max_value * chart_h
        body.append(_line(left, y, left + chart_w, y, dash="4 5"))
        body.append(_text(left - 12, y + 5, f"{tick:.1f}", size=12, fill=MUTED, anchor="end"))
    body.append(_text(22, top + chart_h / 2, "req/s", size=12, fill=MUTED))

    group_w = chart_w / len(profiles)
    bar_w = 52
    gap = 12
    total_bars_w = len(policies) * bar_w + (len(policies) - 1) * gap
    for group_index, (profile, display_name) in enumerate(zip(profiles, names, strict=True)):
        group_x = left + group_index * group_w
        start_x = group_x + (group_w - total_bars_w) / 2
        for policy_index, (key, _, color) in enumerate(policies):
            value = float(profile[key])
            bar_h = value / max_value * chart_h
            x = start_x + policy_index * (bar_w + gap)
            y = top + chart_h - bar_h
            body.append(_rect(x, y, bar_w, bar_h, fill=color, radius=6))
            body.append(_text(x + bar_w / 2, y - 8, f"{value:.3f}", size=11, weight=600, anchor="middle"))
        body.append(_text(group_x + group_w / 2, top + chart_h + 28, display_name, size=13, weight=600, anchor="middle"))
        gain = float(profile["deadline_goodput_gain_percent"])
        if gain > 0:
            body.append(
                _text(
                    group_x + group_w / 2,
                    top + chart_h + 51,
                    f"+{gain:.2f}% vs best baseline",
                    size=11,
                    weight=700,
                    fill=GREEN,
                    anchor="middle",
                )
            )

    legend_x = 580
    for index, (_, label, color) in enumerate(policies):
        x = legend_x + index * 125
        body.append(_rect(x, 47, 12, 12, fill=color, radius=3))
        body.append(_text(x + 18, 58, label, size=11, fill=MUTED))
    body.append(_text(40, 448, "Frozen scope: Qwen2.5-3B · RTX 5070 · BF16 KV · batch-token 2048 · APC off", size=11, fill=MUTED))
    return _svg(width, height, "E09 SLO goodput comparison", body)


def render_ttft_improvements() -> str:
    e02 = _load_csv("reports/e02_batch_tokens/comparison.csv")
    e04 = _load_csv("reports/e04_prefix_cache/comparison.csv")
    e06 = _load_csv("reports/e06_combined/comparison.csv")

    def e02_value(concurrency: str) -> float:
        row = next(
            row
            for row in e02
            if row["input_len"] == "2048"
            and row["output_len"] == "256"
            and row["max_concurrency"] == concurrency
            and row["budget"] == "2048"
        )
        return -float(row["p95_ttft_delta_percent"])

    apc = next(row for row in e04 if row["condition"] == "capacity_reuse90_p1792")

    def e06_value(condition: str) -> float:
        row = next(row for row in e06 if row["condition"] == condition)
        return -float(row["combined_vs_best_single_ttft_percent"])

    rows = [
        ("Batch budget · Long C4", e02_value("4"), BLUE),
        ("Batch budget · Long C8", e02_value("8"), BLUE),
        ("APC · 79% hit · C8", -float(apc["p95_ttft_delta_percent"]), GREEN),
        ("Combined · reuse50 · C4", e06_value("reuse50_p1024"), AMBER),
        ("Combined · reuse90 · C8", e06_value("capacity_reuse90_p1792"), AMBER),
    ]
    width, height = 980, 430
    bar_x, bar_w = 295, 610
    max_value = 60.0
    body = [
        _text(40, 42, "Selected P95 TTFT reductions", size=24, weight=700),
        _text(40, 68, "Controlled comparisons only · lower latency is better", size=13, fill=MUTED),
    ]
    for tick in [0, 10, 20, 30, 40, 50, 60]:
        x = bar_x + tick / max_value * bar_w
        body.append(_line(x, 96, x, 358, dash="4 5"))
        body.append(_text(x, 382, f"{tick}%", size=11, fill=MUTED, anchor="middle"))

    for index, (label, value, color) in enumerate(rows):
        y = 112 + index * 52
        body.append(_text(bar_x - 18, y + 22, label, size=13, weight=600, anchor="end"))
        width_value = value / max_value * bar_w
        body.append(_rect(bar_x, y, width_value, 30, fill=color, radius=7))
        body.append(_text(bar_x + width_value + 10, y + 21, f"{value:.2f}%", size=12, weight=700))
    body.append(_text(40, 412, "E02 compares batch-token 2048 vs 8192; E06 compares the combined setting vs the best single factor.", size=11, fill=MUTED))
    return _svg(width, height, "Selected TTFT improvements", body)


def render_tradeoffs() -> str:
    capacity = _load_json("reports/e05_kv_cache/capacity.json")
    e05 = _load_json("reports/e05_kv_cache/quality.json")
    e07 = _load_json("reports/e07_lora/quality.json")
    e07_cells = _load_csv("reports/e07_lora/comparison.csv")
    width, height = 980, 430
    body = [
        _text(40, 42, "Quality and capacity trade-offs", size=24, weight=700),
        _text(40, 68, "A gain is deployable only when its frozen quality and latency gates also pass", size=13, fill=MUTED),
    ]

    panels = [(40, "E05 · FP8 KV cache", RED), (510, "E07 · QLoRA / LoRA", GREEN)]
    for x, title, color in panels:
        body.append(_rect(x, 96, 430, 285, fill=PANEL, radius=14, stroke=GRID))
        body.append(_rect(x, 96, 6, 285, fill=color, radius=3))
        body.append(_text(x + 24, 130, title, size=18, weight=700))

    cap_ratio = float(capacity["fp8_to_bf16_token_capacity_ratio"])
    body.extend(
        [
            _text(64, 169, f"{cap_ratio:.3f}×", size=32, weight=800, fill=BLUE),
            _text(64, 193, "KV token capacity", size=12, fill=MUTED),
        ]
    )
    e05_metrics = [
        ("Schema pass", e05["bf16"]["schema_pass_rate"], e05["fp8"]["schema_pass_rate"]),
        ("Root-cause F1", e05["bf16"]["root_cause_macro_f1"], e05["fp8"]["root_cause_macro_f1"]),
        ("Action F1", e05["bf16"]["action_micro_f1"], e05["fp8"]["action_micro_f1"]),
    ]
    body.append(_text(246, 211, "BF16", size=11, weight=700, fill=MUTED, anchor="end"))
    body.append(_text(294, 211, "FP8", size=11, weight=700, fill=RED))
    for index, (label, before, after) in enumerate(e05_metrics):
        y = 232 + index * 42
        body.append(_text(64, y, label, size=12, fill=MUTED))
        body.append(_text(246, y, f"{before*100:.1f}%", size=13, weight=650, anchor="end"))
        body.append(_text(270, y, "→", size=14, weight=700, fill=MUTED, anchor="middle"))
        body.append(_text(294, y, f"{after*100:.1f}%", size=13, weight=700, fill=RED))
    body.append(_rect(64, 343, 250, 25, fill="#fee2e2", radius=12))
    body.append(_text(189, 360, "Capacity PASS · quality FAIL", size=11, weight=700, fill="#b91c1c", anchor="middle"))

    e07_metrics = [
        ("Schema pass", e07["base"]["schema_pass_rate"], e07["lora"]["schema_pass_rate"]),
        ("Root-cause F1", e07["base"]["root_cause_macro_f1"], e07["lora"]["root_cause_macro_f1"]),
        ("Action F1", e07["base"]["action_micro_f1"], e07["lora"]["action_micro_f1"]),
    ]
    body.append(_text(716, 169, "Base", size=11, weight=700, fill=MUTED, anchor="end"))
    body.append(_text(764, 169, "LoRA", size=11, weight=700, fill=GREEN))
    for index, (label, before, after) in enumerate(e07_metrics):
        y = 210 + index * 42
        body.append(_text(534, y, label, size=12, fill=MUTED))
        body.append(_text(716, y, f"{before*100:.1f}%", size=13, weight=650, anchor="end"))
        body.append(_text(740, y, "→", size=14, weight=700, fill=MUTED, anchor="middle"))
        body.append(_text(764, y, f"{after*100:.1f}%", size=13, weight=700, fill=GREEN))
    failed_cells = sum(
        1 for cell in e07_cells if cell.get("online_cost") != "PASS"
    )
    total_cells = len(e07_cells)
    body.append(_rect(534, 343, 320, 25, fill="#fef3c7", radius=12))
    body.append(
        _text(
            694,
            360,
            f"Quality PASS · online latency FAIL ({failed_cells}/{total_cells} cells)",
            size=11,
            weight=700,
            fill="#92400e",
            anchor="middle",
        )
    )
    body.append(_text(40, 414, "Both negative deployment decisions are preserved; neither result is relabeled as a successful default.", size=11, fill=MUTED))
    return _svg(width, height, "Quality and capacity trade-offs", body)


def generated_assets() -> dict[Path, str]:
    return {
        ASSET_DIR / "system_architecture.svg": render_architecture(),
        ASSET_DIR / "e09_goodput.svg": render_e09_goodput(),
        ASSET_DIR / "ttft_improvements.svg": render_ttft_improvements(),
        ASSET_DIR / "quality_tradeoffs.svg": render_tradeoffs(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when committed SVGs differ from the report-derived output",
    )
    args = parser.parse_args()
    assets = generated_assets()
    if args.check:
        stale = [
            path
            for path, expected in assets.items()
            if not path.exists() or path.read_text(encoding="utf-8") != expected
        ]
        if stale:
            for path in stale:
                print(f"STALE: {path.relative_to(ROOT)}")
            print("Run: make charts")
            return 1
        print(f"README charts: PASS ({len(assets)} assets)")
        return 0

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for path, content in assets.items():
        path.write_text(content, encoding="utf-8")
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
