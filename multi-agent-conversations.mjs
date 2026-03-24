import fs from "node:fs/promises";
import path from "node:path";
import OpenAI from "openai";

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

async function nextResponderTurn(client, persona, history, round, model) {
  const response = await client.responses.create({
    model,
    input: [
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

    const responderTurn = await nextResponderTurn(client, persona, messages, round, model);
    messages.push({ round, speaker: "responder_agent", content: responderTurn });
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
  const jsonPath = path.join(outputDir, `agent_dialogues_${ts}.json`);
  const jsonlPath = path.join(outputDir, `agent_dialogues_${ts}.jsonl`);
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

