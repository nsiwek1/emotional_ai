#!/usr/bin/env python3
"""
Compare HF emotion predictions (joy / anger / neutral) to persona ground_truth emotion.

Reads JSON produced by emotion_score_hf.py (array of persona score records).

Prediction rules (configurable):
  - concat: argmax over three_way_renormalized on concatenated text (default)
  - seed: argmax on round-1 (seed) only

Ground truth: record["ground_truth_emotion"] ("joy" or "anger" in this dataset).

Strict match: predicted class equals ground truth when prediction is joy or anger.
If the model's argmax among {joy, anger, neutral} is neutral, that is counted as
neutral_predicted (not matching joy/anger labels).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def argmax_joy_anger_neutral(jan: dict[str, Any] | None) -> str | None:
    """Return 'joy', 'anger', or 'neutral' from joy_anger_neutral block."""
    if not jan:
        return None
    tw = jan.get("three_way_renormalized") or {}
    pairs = [
        ("joy", float(tw.get("joy", 0.0))),
        ("anger", float(tw.get("anger", 0.0))),
        ("neutral", float(tw.get("neutral", 0.0))),
    ]
    return max(pairs, key=lambda x: x[1])[0]


def extract_hf_block(record: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode == "concat":
        return record.get("concatenated", {}).get("hf") or {}
    if mode == "seed":
        rounds = record.get("per_round") or []
        if not rounds:
            return {}
        return rounds[0].get("hf") or {}
    raise ValueError(f"Unknown mode: {mode}")


def analyze_records(
    records: list[dict[str, Any]],
    mode: str,
) -> dict[str, Any]:
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    neutral_when_gt_joy = 0
    neutral_when_gt_anger = 0
    errors = 0
    skipped = 0
    total = 0
    matched = 0
    per_scenario: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "match": 0})

    rows: list[dict[str, Any]] = []

    for rec in records:
        gt = (rec.get("ground_truth_emotion") or "").strip().lower()
        if gt not in ("joy", "anger"):
            skipped += 1
            continue

        hf = extract_hf_block(rec, mode)
        if hf.get("error"):
            errors += 1
            rows.append(
                {
                    "persona_id": rec.get("persona_id"),
                    "scenario_id": rec.get("scenario_id"),
                    "ground_truth": gt,
                    "predicted": None,
                    "match": False,
                    "error": hf.get("error"),
                }
            )
            continue

        jan = hf.get("joy_anger_neutral") or {}
        pred = argmax_joy_anger_neutral(jan)
        if pred is None:
            errors += 1
            continue

        total += 1
        scenario = str(rec.get("scenario_id", ""))
        per_scenario[scenario]["n"] += 1

        if pred == "neutral":
            if gt == "joy":
                neutral_when_gt_joy += 1
            else:
                neutral_when_gt_anger += 1

        ok = pred == gt
        if ok:
            matched += 1
            per_scenario[scenario]["match"] += 1

        confusion[(gt, pred)] += 1

        rows.append(
            {
                "persona_id": rec.get("persona_id"),
                "scenario_id": scenario,
                "ground_truth": gt,
                "predicted": pred,
                "match": ok,
                "top_label_full": (hf.get("top") or {}).get("label"),
                "three_way": (jan.get("three_way_renormalized") or {}),
            }
        )

    accuracy = matched / total if total else 0.0

    return {
        "mode": mode,
        "summary": {
            "personas_with_joy_or_anger_gt": total,
            "skipped_non_binary_gt": skipped,
            "api_errors": errors,
            "accuracy_strict_joy_anger_neutral_argmax": accuracy,
            "matches": matched,
            "neutral_predicted_when_gt_joy": neutral_when_gt_joy,
            "neutral_predicted_when_gt_anger": neutral_when_gt_anger,
        },
        "confusion_ground_truth_vs_predicted": {
            f"{gt}_vs_{pred}": int(confusion.get((gt, pred), 0))
            for gt in ("joy", "anger")
            for pred in ("joy", "anger", "neutral")
        },
        "confusion_matrix_counts": {
            f"{a}->{b}": int(confusion.get((a, b), 0))
            for a in ("joy", "anger")
            for b in ("joy", "anger", "neutral")
        },
        "per_scenario_accuracy": {
            s: (v["match"] / v["n"] if v["n"] else 0.0) for s, v in per_scenario.items()
        },
        "per_scenario_counts": dict(per_scenario),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check joy/anger/neutral vs persona ground truth.")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="emotion_score_hf.py output JSON (array).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="",
        help="Optional JSON report path (default: <input_dir>/alignment_report_<mode>.json).",
    )
    parser.add_argument(
        "--mode",
        choices=("concat", "seed"),
        default="concat",
        help="Which prediction to compare (concatenated text vs seed turn only).",
    )
    args = parser.parse_args()

    out_path = args.output
    if not out_path:
        base = os.path.dirname(os.path.abspath(args.input)) or "."
        out_path = os.path.join(base, f"alignment_report_{args.mode}.json")

    with open(args.input, encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        print("Input must be a JSON array.", file=sys.stderr)
        sys.exit(1)

    report = analyze_records(records, args.mode)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    s = report["summary"]
    print(json.dumps(s, indent=2))
    print(f"Wrote report: {out_path}")


if __name__ == "__main__":
    main()
