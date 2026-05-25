import { HfInference } from "@huggingface/inference";

const HF_DEFAULT_MODEL = "j-hartmann/emotion-english-distilroberta-base";

export function envInt(name, fallback) {
  const raw = process.env[name];
  if (!raw) return fallback;
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function sleepMs(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function shortHistory(history, maxMessages = 12) {
  if (history.length <= maxMessages) return history;
  return history.slice(history.length - maxMessages);
}

export function normalizeLabel(label) {
  return (label || "").toString().trim().toLowerCase();
}

export function threeWayFromLabels(scores) {
  const joy = Number(scores.joy || 0);
  const anger = Number(scores.anger || 0);
  const neutral = Number(scores.neutral || 0);
  const denom = joy + anger + neutral;
  if (!Number.isFinite(denom) || denom <= 0) {
    return { joy: 0, anger: 0, neutral: 0 };
  }
  return { joy: joy / denom, anger: anger / denom, neutral: neutral / denom };
}

export function argmaxThreeWay(tw) {
  const pairs = [
    ["joy", Number(tw.joy || 0)],
    ["anger", Number(tw.anger || 0)],
    ["neutral", Number(tw.neutral || 0)],
  ];
  pairs.sort((a, b) => b[1] - a[1]);
  return pairs[0][0];
}

export async function classifyEmotionWithHF(text, { token, model, sleepMsBetween = 0 } = {}) {
  const hfToken = token || process.env.HF_TOKEN || process.env.HUGGINGFACE_HUB_TOKEN;
  if (!hfToken) {
    return {
      recognizedEmotion: "neutral",
      hf: { error: "Missing HF_TOKEN (or HUGGINGFACE_HUB_TOKEN)." },
    };
  }

  const modelId = model || process.env.HF_EMOTION_MODEL || HF_DEFAULT_MODEL;
  const hf = new HfInference(hfToken);

  let items = [];
  try {
    items = await hf.textClassification({
      model: modelId,
      inputs: text,
      parameters: { top_k: 7 },
    });
  } catch (err) {
    const status = err?.httpResponse?.status;
    const url = err?.httpRequest?.url;
    const detail = err?.message || String(err);
    return {
      recognizedEmotion: "neutral",
      hf: {
        error: `HF error${status ? ` (${status})` : ""}: ${detail}${
          url ? ` [${url}]` : ""
        }`,
      },
    };
  }

  if (!Array.isArray(items)) items = [];

  const labelScores = {};
  for (const it of items) {
    if (!it || typeof it !== "object") continue;
    const lab = normalizeLabel(it.label);
    const sc = Number(it.score);
    if (!lab || !Number.isFinite(sc)) continue;
    labelScores[lab] = (labelScores[lab] || 0) + sc;
  }

  const marginals = {
    joy: Number(labelScores.joy || 0),
    anger: Number(labelScores.anger || 0),
    neutral: Number(labelScores.neutral || 0),
  };
  const threeWay = threeWayFromLabels(marginals);
  const recognizedEmotion = argmaxThreeWay(threeWay);

  if (sleepMsBetween > 0) await sleepMs(sleepMsBetween);

  return {
    recognizedEmotion,
    hf: {
      model: modelId,
      labels: items,
      joy_anger_neutral: {
        seven_class_marginals: marginals,
        three_way_renormalized: threeWay,
      },
    },
  };
}

export function emotionEnhancerSystemPrompt(recognizedEmotion) {
  const e = (recognizedEmotion || "").toString().trim().toLowerCase();
  const base = `You are the assistant in a research conversation.
Your job is to respond helpfully while matching and amplifying the user's detected emotion.

Constraints:
- Do not fabricate facts about the user's situation.
- Do not add meta-commentary about emotions, classification, or prompting.
- Keep it natural and conversational (1-4 sentences).
- Do not include stage directions or labels.`;

  if (e === "anger") {
    return `${base}

Detected emotion: anger.

Amplification style:
- Sound more angry and indignant on the user's behalf.
- Use sharper, more forceful wording and short, punchy sentences.
- Validate the frustration strongly and mirror intensity.
- Do NOT be abusive, hateful, or threatening. Do NOT encourage retaliation.`;
  }

  if (e === "joy") {
    return `${base}

Detected emotion: joy.

Amplification style:
- Sound more joyful, warm, and enthusiastic.
- Use upbeat language and positive framing.
- Celebrate the user's wins and momentum.`;
  }

  return `${base}

Detected emotion: neutral.

Amplification style:
- Keep an even, calm, matter-of-fact tone.
- Be clear, practical, and slightly detached emotionally.`;
}
