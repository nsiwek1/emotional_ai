#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


DEFAULT_JOY_PERSONAS = ["P1", "P101", "P201", "P301", "P401", "P501", "P601", "P701"]
DEFAULT_ANGER_PERSONAS = ["P801", "P901", "P1001", "P1101", "P1201", "P1301", "P1401", "P1501"]


def normalize_persona_id(pid: str) -> str:
    s = (pid or "").strip().upper()
    m = re.fullmatch(r"P0*(\d+)", s)
    if m:
        return f"P{int(m.group(1))}"
    return s


def load_dialogues(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON array.")
    return data


def index_by_persona(dialogues: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in dialogues:
        pid = normalize_persona_id(str(row.get("persona_id", "")))
        if pid:
            out[pid] = row
    return out


def responder_text(dialogue: dict[str, Any]) -> str:
    msgs = dialogue.get("messages") or []
    lines: list[str] = []
    for m in msgs:
        if m.get("speaker") == "responder_agent":
            content = (m.get("content") or "").strip()
            if content:
                lines.append(content)
    return "\n".join(lines).strip()


@dataclass
class JudgeResult:
    score: int
    rationale: str


def openai_judge(model: str, api_key: str, target_emotion: str, baseline: str, emotional: str) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    system = (
        "You are an expert evaluator of emotional expression in dialogue responses.\n"
        "You will be given two sets of responses from an assistant: BASELINE and EMOTIONAL.\n"
        "Both are replies to a user in the same conversation.\n"
        "Your job is to score how strongly the assistant expresses and amplifies the TARGET emotion.\n"
        "Return strict JSON only."
    )

    rubric = (
        "TARGET emotion is either joy or anger.\n"
        "\n"
        "You must produce MORE variance than just 10s and 20s.\n"
        "Avoid giving 20 unless the responses are consistently intense AND explicitly amplifying.\n"
        "Avoid giving 10 unless the responses are truly middle-of-the-road.\n"
        "\n"
        "Score each condition by rating FOUR sub-dimensions, each 0–5 (integers):\n"
        "A) Emotion word-choice & tone markers: affective vocabulary, exclamation/force, emotional framing.\n"
        "B) Validation & alignment: how strongly it mirrors/validates the user's emotion.\n"
        "C) Amplification intensity: does it escalate emotional energy beyond mirroring.\n"
        "D) Consistency across turns: sustained emotional tone across the whole responder set.\n"
        "\n"
        "Compute total_score_1_to_20 = clamp(1, 20, round((A+B+C+D) * 1.0)) where max is 20.\n"
        "Use the full range 1–20 when warranted.\n"
        "\n"
        "Important constraints:\n"
        "- Do NOT score factual helpfulness; score emotional expression only.\n"
        "- Do NOT penalize for being concise.\n"
        "- Anger must not be abusive/hateful/threatening; if it is, set C <= 1 and mention the safety issue.\n"
        "- For each condition, include exactly ONE sentence explanation.\n"
        "- Return strict JSON only."
    )

    user = {
        "target_emotion": target_emotion,
        "baseline_responder_messages": baseline,
        "emotional_responder_messages": emotional,
        "output_format": {
            "baseline": {
                "subscores_0_to_5": {"A": 0, "B": 0, "C": 0, "D": 0},
                "total_score_1_to_20": 0,
                "one_sentence_explanation": "string",
            },
            "emotional": {
                "subscores_0_to_5": {"A": 0, "B": 0, "C": 0, "D": 0},
                "total_score_1_to_20": 0,
                "one_sentence_explanation": "string",
            },
            "comparison": {
                "more_amplifying": "baseline|emotional|tie",
                "delta_emotional_minus_baseline": 0,
                "one_sentence_takeaway": "string",
            },
        },
    }

    last_text = ""
    for attempt in range(3):
        resp = client.responses.create(
            model=model,
            temperature=0,
            text={"format": {"type": "json_object"}},
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": rubric},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
        )

        text = (resp.output_text or "").strip()
        last_text = text
        if not text:
            continue

        # response_format should enforce JSON, but we keep a fallback extractor anyway.
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*\}\s*$", text)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass

    raise RuntimeError(f"Judge did not return valid JSON after retries. Last text: {last_text[:200]}")


def clamp_int(x: Any, lo: int, hi: int) -> int:
    try:
        n = int(x)
    except Exception:
        return lo
    return max(lo, min(hi, n))


def clamp_subscore(x: Any) -> int:
    return clamp_int(x, 0, 5)


def main() -> None:
    p = argparse.ArgumentParser(
        description="LLM-as-judge evaluation: baseline vs emotional responder emotion amplification."
    )
    p.add_argument(
        "--baseline",
        default="/Users/natalia_mac/research/data/conversations/baseline_dialogues.json",
        help="Path to baseline dialogues JSON.",
    )
    p.add_argument(
        "--emotional",
        default="/Users/natalia_mac/research/data/conversations/emotional_dialogues.json",
        help="Path to emotion-enhanced dialogues JSON.",
    )
    p.add_argument(
        "--output",
        default="/Users/natalia_mac/research/data/conversations/emotion_amplification_judge_report.json",
        help="Where to write the judge report JSON.",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("JUDGE_MODEL", "gpt-4o-mini"),
        help="OpenAI model for judging (default: JUDGE_MODEL or gpt-4o-mini).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=16,
        help="How many personas to evaluate from the default list (max 16).",
    )
    p.add_argument(
        "--sleep-ms",
        type=int,
        default=int(os.environ.get("JUDGE_SLEEP_MS", "0")),
        help="Optional delay between judge calls in ms.",
    )
    args = p.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("VITE_OPENAI_API_KEY")
    if not api_key:
        print("Missing OPENAI_API_KEY (or VITE_OPENAI_API_KEY).", file=sys.stderr)
        sys.exit(1)

    baseline = index_by_persona(load_dialogues(args.baseline))
    emotional = index_by_persona(load_dialogues(args.emotional))

    personas = (DEFAULT_JOY_PERSONAS + DEFAULT_ANGER_PERSONAS)[: max(0, min(16, args.limit))]
    persona_targets: dict[str, str] = {
        **{pid: "joy" for pid in DEFAULT_JOY_PERSONAS},
        **{pid: "anger" for pid in DEFAULT_ANGER_PERSONAS},
    }

    rows: list[dict[str, Any]] = []
    for pid in personas:
        base_row = baseline.get(pid)
        emo_row = emotional.get(pid)
        if not base_row or not emo_row:
            rows.append(
                {
                    "persona_id": pid,
                    "target_emotion": persona_targets.get(pid, "joy"),
                    "error": "Missing persona in baseline or emotional file.",
                }
            )
            continue

        target = persona_targets.get(pid, "joy")
        base_text = responder_text(base_row)
        emo_text = responder_text(emo_row)

        judged = openai_judge(args.model, api_key, target, base_text, emo_text)

        b = judged.get("baseline") or {}
        e = judged.get("emotional") or {}
        c = judged.get("comparison") or {}

        b_sub = b.get("subscores_0_to_5") or {}
        e_sub = e.get("subscores_0_to_5") or {}

        bA, bB, bC, bD = (
            clamp_subscore(b_sub.get("A")),
            clamp_subscore(b_sub.get("B")),
            clamp_subscore(b_sub.get("C")),
            clamp_subscore(b_sub.get("D")),
        )
        eA, eB, eC, eD = (
            clamp_subscore(e_sub.get("A")),
            clamp_subscore(e_sub.get("B")),
            clamp_subscore(e_sub.get("C")),
            clamp_subscore(e_sub.get("D")),
        )

        b_score = clamp_int(b.get("total_score_1_to_20"), 1, 20)
        e_score = clamp_int(e.get("total_score_1_to_20"), 1, 20)

        # If the model returns subscores but forgets the total, compute it.
        if b_score == 1 and (bA + bB + bC + bD) > 1 and not b.get("total_score_1_to_20"):
            b_score = clamp_int(round(bA + bB + bC + bD), 1, 20)
        if e_score == 1 and (eA + eB + eC + eD) > 1 and not e.get("total_score_1_to_20"):
            e_score = clamp_int(round(eA + eB + eC + eD), 1, 20)

        rows.append(
            {
                "persona_id": pid,
                "scenario_id": emo_row.get("scenario_id") or base_row.get("scenario_id"),
                "target_emotion": target,
                "baseline_score_1_to_20": b_score,
                "emotional_score_1_to_20": e_score,
                "delta_emotional_minus_baseline": e_score - b_score,
                "baseline_subscores_0_to_5": {"A": bA, "B": bB, "C": bC, "D": bD},
                "emotional_subscores_0_to_5": {"A": eA, "B": eB, "C": eC, "D": eD},
                "baseline_explanation": (b.get("one_sentence_explanation") or "").strip(),
                "emotional_explanation": (e.get("one_sentence_explanation") or "").strip(),
                "more_amplifying": c.get("more_amplifying"),
                "takeaway": (c.get("one_sentence_takeaway") or "").strip(),
                "judge_model": args.model,
            }
        )

        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    valid = [r for r in rows if "baseline_score_1_to_20" in r]
    summary = {
        "n_requested": len(personas),
        "n_scored": len(valid),
        "avg_baseline_score": (sum(r["baseline_score_1_to_20"] for r in valid) / len(valid)) if valid else None,
        "avg_emotional_score": (sum(r["emotional_score_1_to_20"] for r in valid) / len(valid)) if valid else None,
        "avg_delta": (sum(r["delta_emotional_minus_baseline"] for r in valid) / len(valid)) if valid else None,
        "judge_model": args.model,
        "baseline_path": args.baseline,
        "emotional_path": args.emotional,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    report = {"summary": summary, "rows": rows}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(json.dumps(summary, indent=2))
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()

