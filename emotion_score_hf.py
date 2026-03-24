#!/usr/bin/env python3
"""
Hugging Face Inference API emotion scoring.

Recommended model (default): j-hartmann/emotion-english-distilroberta-base
  - English text-classification via Hugging Face Inference Providers (InferenceClient).
  - Do not use the legacy URL api-inference.huggingface.co (returns 410 Gone).
  - Labels include joy, anger, neutral (plus disgust, fear, sadness, surprise).
  - Each scored payload includes joy/anger/neutral marginals and a 3-way
    renormalization over those classes for analysis.

Scores each persona turn (seed = turns[0], follow-ups = turns[1:]) and the
concatenated full text. Uses the model's returned label scores.

Uncertainty: Shannon entropy over the normalized distribution of returned
labels. If the API returns only top-k labels, entropy is conditional on that
partial list (not the full softmax over all classes).

Usage:
  export HF_TOKEN=hf_...
  python emotion_score_hf.py --input data/personas/persona_final.json --output data/emotion_scores/out.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# See https://huggingface.co/j-hartmann/emotion-english-distilroberta-base (7 classes incl. joy, anger, neutral).
RECOMMENDED_EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
DEFAULT_MODEL = RECOMMENDED_EMOTION_MODEL


def load_personas(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return []
    # JSON array
    if raw.startswith("["):
        return json.loads(raw)
    # JSONL
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            lines.append(json.loads(line))
    return lines


def _normalize_probs(scores: list[float]) -> list[float]:
    s = sum(max(0.0, x) for x in scores)
    if s <= 0:
        n = len(scores)
        return [1.0 / n] * n if n else []
    return [max(0.0, x) / s for x in scores]


def joy_anger_neutral_from_labels(labels_out: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Extract joy / anger / neutral using normalized scores from the full returned label list.
    - seven_class_marginals: model-normalized mass on each of the three labels (subset of 7-way distribution).
    - three_way_renormalized: those three probabilities renormalized to sum to 1.
    """
    by_label: dict[str, float] = {}
    for x in labels_out:
        key = str(x.get("label", "")).strip().lower()
        by_label[key] = float(x.get("normalized", 0.0))

    j = by_label.get("joy", 0.0)
    a = by_label.get("anger", 0.0)
    n = by_label.get("neutral", 0.0)
    s = j + a + n
    if s > 0:
        three = {"joy": j / s, "anger": a / s, "neutral": n / s}
    else:
        three = {"joy": 0.0, "anger": 0.0, "neutral": 0.0}

    return {
        "seven_class_marginals": {"joy": j, "anger": a, "neutral": n},
        "three_way_renormalized": three,
    }


def shannon_entropy(probs: list[float]) -> float:
    """Entropy in nats (use log base e)."""
    h = 0.0
    for p in probs:
        if p > 0:
            h -= p * math.log(p)
    return h


def hf_inference(
    text: str,
    token: str,
    model_id: str,
    *,
    top_k: int = 7,
    max_retries: int = 5,
    base_sleep: float = 2.0,
) -> list[dict[str, Any]]:
    """
    Call Hugging Face Inference Providers via huggingface_hub.InferenceClient.
    Legacy POST to api-inference.huggingface.co returns 410 Gone — do not use.
    """
    from huggingface_hub import InferenceClient
    from huggingface_hub.errors import HfHubHTTPError

    client = InferenceClient(token=token)

    for attempt in range(max_retries):
        try:
            result = client.text_classification(text, model=model_id, top_k=top_k)
            return [{"label": str(x.label), "score": float(x.score)} for x in result]
        except HfHubHTTPError as e:
            code = getattr(e.response, "status_code", None)
            if code == 503 and attempt < max_retries - 1:
                time.sleep(min(base_sleep * (attempt + 1), 60.0))
                continue
            raise RuntimeError(f"HfHubHTTPError {code}: {e}") from e
    raise RuntimeError(f"Inference failed for {model_id} after retries.")


def score_text(
    text: str,
    token: str,
    model_id: str,
    *,
    top_k: int = 7,
) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {
            "labels": [],
            "entropy": 0.0,
            "top": {"label": None, "score": 0.0},
            "joy_anger_neutral": joy_anger_neutral_from_labels([]),
            "error": "empty_text",
        }

    try:
        labels_raw = hf_inference(text, token, model_id, top_k=top_k)
    except Exception as e:
        return {
            "labels": [],
            "entropy": 0.0,
            "top": {"label": None, "score": 0.0},
            "joy_anger_neutral": joy_anger_neutral_from_labels([]),
            "error": str(e),
        }

    scores = [float(x["score"]) for x in labels_raw]
    probs = _normalize_probs(scores)
    ent = shannon_entropy(probs)
    labels_out = [
        {"label": labels_raw[i]["label"], "score": labels_raw[i]["score"], "normalized": probs[i]}
        for i in range(len(labels_raw))
    ]
    if labels_out:
        top = max(labels_out, key=lambda x: x["score"])
        top_out = {"label": top["label"], "score": top["score"]}
    else:
        top_out = {"label": None, "score": 0.0}

    return {
        "labels": labels_out,
        "entropy": ent,
        "entropy_note": "Shannon entropy over normalized returned scores (partial if API returns top-k only).",
        "top": top_out,
        "joy_anger_neutral": joy_anger_neutral_from_labels(labels_out),
    }


def process_persona(
    persona: dict[str, Any],
    token: str,
    model_id: str,
    concat_sep: str,
    sleep_ms: int,
    top_k: int,
) -> dict[str, Any]:
    turns = persona.get("turns") or []
    seed_followup = {
        "seed": turns[0] if len(turns) > 0 else None,
        "followups": turns[1:] if len(turns) > 1 else [],
    }

    per_round: list[dict[str, Any]] = []
    for i, t in enumerate(turns):
        if sleep_ms > 0 and i > 0:
            time.sleep(sleep_ms / 1000.0)
        round_num = i + 1
        kind = "seed" if i == 0 else "followup"
        hf = score_text(str(t), token, model_id, top_k=top_k)
        per_round.append(
            {
                "round": round_num,
                "kind": kind,
                "text": str(t).strip(),
                "hf": hf,
            }
        )

    concat_text = concat_sep.join(str(t).strip() for t in turns if str(t).strip())
    if sleep_ms > 0 and turns:
        time.sleep(sleep_ms / 1000.0)
    concat_hf = score_text(concat_text, token, model_id, top_k=top_k)

    return {
        "persona_id": persona.get("persona_id"),
        "scenario_id": persona.get("scenario_id"),
        "ground_truth_emotion": persona.get("emotion"),
        "ground_truth_intensity": persona.get("intensity"),
        "topic": persona.get("topic"),
        "seed_and_followups": seed_followup,
        "per_round": per_round,
        "concatenated": {"text": concat_text, "hf": concat_hf},
        "hf_model": model_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score persona turns with Hugging Face Inference API.")
    parser.add_argument("--input", "-i", required=True, help="Path to persona JSON array or JSONL.")
    parser.add_argument("--output", "-o", required=True, help="Output JSON path.")
    parser.add_argument(
        "--model",
        default=os.environ.get("HF_EMOTION_MODEL", RECOMMENDED_EMOTION_MODEL),
        help=(
            f"HF model id (default: {RECOMMENDED_EMOTION_MODEL} — "
            "joy/anger/neutral-friendly English emotion classifier on Inference API)."
        ),
    )
    parser.add_argument(
        "--concat-sep",
        default=" ",
        help="Separator when joining turns for concatenated score (default: space).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max personas to process (0 = all).")
    parser.add_argument("--sleep-ms", type=int, default=0, help="Delay between API calls in ms.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(os.environ.get("HF_EMOTION_TOP_K", "7")),
        help="Number of classes to return from text-classification (default: 7 for Ekman+neutral).",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        print("Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN in the environment.", file=sys.stderr)
        sys.exit(1)

    personas = load_personas(args.input)
    if args.limit > 0:
        personas = personas[: args.limit]

    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(out_dir, exist_ok=True)

    jsonl_path = args.output.replace(".json", ".jsonl") if args.output.endswith(".json") else args.output + ".jsonl"

    results: list[dict[str, Any]] = []
    with open(jsonl_path, "w", encoding="utf-8") as jsonl_f:
        for idx, p in enumerate(personas):
            print(f"Scoring {p.get('persona_id', idx)} ({idx + 1}/{len(personas)})...", flush=True)
            rec = process_persona(
                p,
                token,
                args.model,
                args.concat_sep,
                args.sleep_ms,
                args.top_k,
            )
            results.append(rec)
            jsonl_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            jsonl_f.flush()

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote {len(results)} records to {args.output}")
    print(f"Checkpoint JSONL (one persona per line): {jsonl_path}")


if __name__ == "__main__":
    main()
