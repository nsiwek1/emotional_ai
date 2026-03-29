import fs from "node:fs/promises";
import path from "node:path";
import { HfInference } from "@huggingface/inference";
import OpenAI from "openai";
import dotenv from "dotenv";

// Explicitly load .env from the current working directory. This is more reliable
// than relying on implicit dotenv/config resolution across different Node modes.
dotenv.config({ path: path.resolve(process.cwd(), ".env") });

const HF_DEFAULT_MODEL = "j-hartmann/emotion-english-distilroberta-base";

const DEFAULT_PERSONA_IDS = [
  "P1",
  "P101",
  "P201",
  "P301",
  "P401",
  "P501",
  "P601",
  "P701",
  "P801",
  "P901",
  "P1001",
  "P1101",
  "P1201",
  "P1301",
  "P1401",
  "P1501",
];

function envInt(name, fallback) {
  const raw = process.env[name];
  if (!raw) return fallback;
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

async function ensureDir(dir) {
  await fs.mkdir(dir, { recursive: true });
}

async function readPersonas(inputPath) {
  const raw = await fs.readFile(inputPath, "utf8");
  return raw
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function pickPersonas(all, selectedIds) {
  const map = new Map(all.map((p) => [p.persona_id, p]));
  const missing = selectedIds.filter((id) => !map.has(id));
  if (missing.length) {
    throw new Error(`Missing persona IDs in input file: ${missing.join(", ")}`);
  }
  return selectedIds.map((id) => map.get(id));
}

function shortHistory(history, maxMessages = 12) {
  if (history.length <= maxMessages) return history;
  return history.slice(history.length - maxMessages);
}

function sleepMs(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function normalizeLabel(label) {
  return (label || "").toString().trim().toLowerCase();
}

function threeWayFromLabels(scores) {
  const joy = Number(scores.joy || 0);
  const anger = Number(scores.anger || 0);
  const neutral = Number(scores.neutral || 0);
  const denom = joy + anger + neutral;
  if (!Number.isFinite(denom) || denom <= 0) {
    return { joy: 0, anger: 0, neutral: 0 };
  }
  return { joy: joy / denom, anger: anger / denom, neutral: neutral / denom };
}

function argmaxThreeWay(tw) {
  const pairs = [
    ["joy", Number(tw.joy || 0)],
    ["anger", Number(tw.anger || 0)],
    ["neutral", Number(tw.neutral || 0)],
  ];
  pairs.sort((a, b) => b[1] - a[1]);
  return pairs[0][0];
}

async function classifyEmotionWithHF(text, { token, model, sleepMsBetween = 0 } = {}) {
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

  // @huggingface/inference returns [ {label, score}, ... ]
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

function emotionEnhancerSystemPrompt(recognizedEmotion) {
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

async function nextPersonaTurn(client, persona, history, round, model) {
  const system = `You are roleplaying one persona in a research conversation.
Stay in character and keep the same scenario.

Persona metadata:
- Persona ID: ${persona.persona_id}
- Emotion: ${persona.emotion}
- Intensity: ${persona.intensity}
- Topic: ${persona.topic}
- Background: ${persona.background}

Rules:
- Output one short message only (1-3 sentences).
- Keep emotional tone consistent.
- Keep discussing the same issue.
- No stage directions or labels.`;

  const seedHint = persona.turns?.[Math.min(round - 1, 4)] || "";

  const response = await client.responses.create({
    model,
    input: [
      { role: "system", content: system },
      {
        role: "user",
        content: `Round ${round}. Use this trajectory hint if helpful: "${seedHint}". Conversation so far:\n${JSON.stringify(
          shortHistory(history)
        )}\n\nWrite the persona's next message.`,
      },
    ],
  });

  const text = response.output_text?.trim();
  if (!text) throw new Error("Persona agent returned empty text.");
  return text;
}

async function nextResponderTurn(client, history, round, model, recognizedEmotion) {
  const system = emotionEnhancerSystemPrompt(recognizedEmotion);
  const response = await client.responses.create({
    model,
    input: [
      { role: "system", content: system },
      {
        role: "user",
        content: `Round ${round}. Continue this conversation naturally.\nConversation so far:\n${JSON.stringify(
          shortHistory(history)
        )}\n\nWrite the assistant's next reply.`,
      },
    ],
  });

  const text = response.output_text?.trim();
  if (!text) throw new Error("Responder agent returned empty text.");
  return text;
}

async function runOneConversation(client, persona, rounds, model) {
  const messages = [];
  for (let round = 1; round <= rounds; round += 1) {
    const personaTurn = await nextPersonaTurn(client, persona, messages, round, model);
    messages.push({ round, speaker: "persona_agent", content: personaTurn });

    const hfSleepMs = envInt("HF_SLEEP_MS", 0);
    const { recognizedEmotion, hf } = await classifyEmotionWithHF(personaTurn, {
      sleepMsBetween: hfSleepMs,
    });
    if (hf?.error) {
      console.warn(`HF classify failed (${persona.persona_id} r${round}): ${hf.error}`);
    }

    const responderTurn = await nextResponderTurn(
      client,
      messages,
      round,
      model,
      recognizedEmotion
    );
    messages.push({
      round,
      speaker: "responder_agent",
      content: responderTurn,
      recognized_emotion: recognizedEmotion,
      hf_emotion: hf,
    });
  }
  return messages;
}

async function main() {
  const inputPath =
    process.env.PERSONA_INPUT_FILE ||
    "/Users/natalia_mac/research/data/personas/personas_2026-03-23T20-14-30-670Z.jsonl";
  const outputDir = process.env.CONVERSATION_OUTPUT_DIR || "data/conversations";
  const model = process.env.CONVERSATION_MODEL || "gpt-4o-mini";
  const rounds = envInt("CONVERSATION_ROUNDS", 5);

  const key = process.env.OPENAI_API_KEY || process.env.VITE_OPENAI_API_KEY;
  if (!key) throw new Error("Missing OPENAI_API_KEY (or VITE_OPENAI_API_KEY).");

  const selectedIds = (process.env.SELECTED_PERSONA_IDS || DEFAULT_PERSONA_IDS.join(","))
    .split(",")
    .map((x) => x.trim().toUpperCase())
    .map((x) => (x === "P!" ? "P1" : x));

  await ensureDir(outputDir);
  const ts = new Date().toISOString().replace(/[:.]/g, "-");
  const jsonPath = path.join(outputDir, `agent_dialogues_emotion_enhanced_${ts}.json`);
  const jsonlPath = path.join(outputDir, `agent_dialogues_emotion_enhanced_${ts}.jsonl`);
  await fs.writeFile(jsonlPath, "", "utf8");

  const personas = await readPersonas(inputPath);
  const selected = pickPersonas(personas, selectedIds);
  const client = new OpenAI({ apiKey: key });

  const rows = [];
  for (let i = 0; i < selected.length; i += 1) {
    const persona = selected[i];
    console.log(`Running ${persona.persona_id} (${i + 1}/${selected.length})...`);
    const messages = await runOneConversation(client, persona, rounds, model);
    const row = {
      persona_id: persona.persona_id,
      scenario_id: persona.scenario_id,
      emotion: persona.emotion,
      intensity: persona.intensity,
      topic: persona.topic,
      rounds,
      messages,
    };
    rows.push(row);
    await fs.appendFile(jsonlPath, `${JSON.stringify(row)}\n`, "utf8");
  }

  await fs.writeFile(jsonPath, `${JSON.stringify(rows, null, 2)}\n`, "utf8");
  console.log(`Saved: ${jsonPath}`);
  console.log(`Saved: ${jsonlPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

