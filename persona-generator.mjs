import fs from "node:fs/promises";
import path from "node:path";
import OpenAI from "openai";

const EMOTIONS = ["joy", "anger"];
const INTENSITIES = ["low", "high"];
const TOPICS = [
  "work_performance",
  "relationship",
  "life_event",
  "public_event",
];

const TOPIC_LABELS = {
  work_performance: "work performance issue",
  relationship: "relationship issue",
  life_event: "personal life situation",
  public_event: "public policy issue",
};

const BACKGROUNDS = {
  joy: {
    low: {
      work_performance:
        "You received positive but modest feedback from your manager on a recent task.",
      relationship:
        "You had a pleasant conversation with a close friend that made you feel appreciated.",
      life_event:
        "You recently started a small hobby and are quietly enjoying it.",
      public_event:
        "You read about a local community program helping people in your city.",
    },
    high: {
      work_performance:
        "You got a major promotion after months of hard work and feel thrilled.",
      relationship:
        "Someone close surprised you with a deeply thoughtful gesture.",
      life_event:
        "You achieved a personal milestone you worked toward for years.",
      public_event:
        "You saw news about a major scientific breakthrough that could help many people.",
    },
  },
  anger: {
    low: {
      work_performance:
        "Some of your work was not acknowledged in a team meeting and it irritated you.",
      relationship:
        "A friend canceled plans at the last minute again, and it bothered you.",
      life_event:
        "Recurring apartment maintenance issues are starting to get on your nerves.",
      public_event:
        "You read about a local policy decision that feels unfair to many people.",
    },
    high: {
      work_performance:
        "A coworker presented your work as their own in a meeting, and you feel deeply disrespected.",
      relationship:
        "You had a serious argument with someone close and still feel furious.",
      life_event:
        "You were charged a large unexpected fee for something you never agreed to.",
      public_event:
        "You read about a government decision you view as deeply unfair and harmful.",
    },
  },
};

const TOPIC_KEYWORDS = {
  work_performance: [
    "work",
    "performance",
    "manager",
    "coworker",
    "meeting",
    "project",
    "feedback",
    "deadline",
    "team",
    "promotion",
  ],
  relationship: [
    "relationship",
    "friend",
    "partner",
    "family",
    "conversation",
    "argument",
    "support",
    "trust",
    "plans",
    "close",
  ],
  life_event: [
    "life",
    "home",
    "apartment",
    "hobby",
    "milestone",
    "fee",
    "health",
    "routine",
    "personal",
  ],
  public_event: [
    "public",
    "policy",
    "government",
    "community",
    "city",
    "news",
    "decision",
    "program",
    "political",
  ],
};

const EMOTION_KEYWORDS = {
  joy: ["happy", "glad", "excited", "hopeful", "grateful", "proud", "relieved", "joy"],
  anger: ["angry", "annoyed", "frustrated", "furious", "upset", "irritated", "outraged", "mad"],
};

const INTENSITY_KEYWORDS = {
  low: ["a bit", "a little", "somewhat", "slightly", "mildly", "quietly"],
  high: ["very", "extremely", "deeply", "really", "intensely", "absolutely"],
};

const STYLE_TRAITS = {
  ageBand: ["18-24", "25-34", "35-44", "45-60"],
  communicationStyle: ["direct", "reflective", "storytelling", "matter-of-fact", "casual"],
  verbosity: ["brief", "moderate"],
  formality: ["informal", "neutral", "formal"],
  pacing: ["calm", "urgent"],
};

const STOP_WORDS = new Set([
  "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "it", "is", "i", "me", "my", "you",
  "this", "that", "was", "were", "with", "as", "be", "am", "are", "at", "by", "from",
]);

function envInt(name, fallback) {
  const raw = process.env[name];
  if (!raw) return fallback;
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function envFloat(name, fallback) {
  const raw = process.env[name];
  if (!raw) return fallback;
  const parsed = Number.parseFloat(raw);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function scenarioLabel(emotion, intensity, topic) {
  return `${emotion}_${intensity}_${topic}`;
}

async function ensureDir(dir) {
  await fs.mkdir(dir, { recursive: true });
}

function chooseOne(items) {
  return items[Math.floor(Math.random() * items.length)];
}

function samplePersonaStyle() {
  return {
    ageBand: chooseOne(STYLE_TRAITS.ageBand),
    communicationStyle: chooseOne(STYLE_TRAITS.communicationStyle),
    verbosity: chooseOne(STYLE_TRAITS.verbosity),
    formality: chooseOne(STYLE_TRAITS.formality),
    pacing: chooseOne(STYLE_TRAITS.pacing),
  };
}

function normalizeTokens(text) {
  return new Set(
    String(text)
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter((w) => w && w.length > 2 && !STOP_WORDS.has(w))
  );
}

function jaccardSimilarity(aSet, bSet) {
  const a = [...aSet];
  const b = [...bSet];
  if (!a.length || !b.length) return 0;

  let intersection = 0;
  for (const token of a) {
    if (bSet.has(token)) intersection += 1;
  }

  const union = new Set([...a, ...b]).size;
  return union ? intersection / union : 0;
}

function trajectorySignature(turns) {
  return normalizeTokens((turns || []).join(" "));
}

function isNearDuplicateTrajectory(turns, existingSignatures, threshold) {
  const candidate = trajectorySignature(turns);
  for (const signature of existingSignatures) {
    if (jaccardSimilarity(candidate, signature) >= threshold) return true;
  }
  return false;
}

function hasAnyKeyword(text, keywords) {
  const lowered = String(text).toLowerCase();
  return keywords.some((word) => lowered.includes(word));
}

function countTopicTurns(turns, topic) {
  const keywords = TOPIC_KEYWORDS[topic] || [];
  return turns.filter((turn) => hasAnyKeyword(turn, keywords)).length;
}

function countEmotionTurns(turns, emotion) {
  const keywords = EMOTION_KEYWORDS[emotion] || [];
  return turns.filter((turn) => hasAnyKeyword(turn, keywords)).length;
}

function countIntensityTurns(turns, intensity) {
  const keywords = INTENSITY_KEYWORDS[intensity] || [];
  return turns.filter((turn) => hasAnyKeyword(turn, keywords)).length;
}

function uniqueTurnCount(turns) {
  return new Set(turns.map((turn) => turn.toLowerCase().trim())).size;
}

function buildPrompt({
  personaId,
  emotion,
  intensity,
  topic,
  background,
  personaStyle,
  avoidOpenings,
}) {
  const avoidBlock = avoidOpenings.length
    ? `Avoid reusing these opening styles from recent personas in this scenario:
${avoidOpenings
        .map((p, i) => `${i + 1}. ${p}`)
        .join("\n")}`
    : "";

  return `
You are generating synthetic user utterances for an emotion-steering study.

Persona ID: ${personaId}
Emotion: ${emotion}
Intensity: ${intensity}
Topic: ${topic}
Background: ${background}

Persona style profile:
- Age band: ${personaStyle.ageBand}
- Communication style: ${personaStyle.communicationStyle}
- Verbosity: ${personaStyle.verbosity}
- Formality: ${personaStyle.formality}
- Pacing: ${personaStyle.pacing}

Task:
Generate a 5-turn persona-only trajectory about the SAME underlying issue.
There is no assistant reply. The persona is continuing their own account over turns 1-5.

Turn progression:
- turn_1: initial issue statement
- turn_2: added context or timeline detail
- turn_3: personal impact
- turn_4: reflection, escalation/de-escalation consistent with intensity
- turn_5: next-step feeling or unresolved stance

Strict constraints:
- Output valid JSON only.
- Keep all 5 turns in first person.
- Each turn must be 1-2 sentences.
- Keep emotion and intensity consistent across all turns.
- Do not introduce a new unrelated event.
- All turns must stay anchored to the same topic and situation.
- Use lexical variety; avoid repetitive sentence frames.
- Avoid profanity, slurs, and explicit violence.
${avoidBlock}

Return this exact JSON shape:
{
  "turns": ["turn_1", "turn_2", "turn_3", "turn_4", "turn_5"]
}
`.trim();
}

async function generateOne(client, params, temperature) {
  const prompt = buildPrompt(params);
  const response = await client.responses.create({
    model: process.env.PERSONA_MODEL || "gpt-4o-mini",
    temperature,
    input: [
      {
        role: "system",
        content:
          "You create controlled, standardized synthetic user trajectories for research. Return strict JSON only.",
      },
      { role: "user", content: prompt },
    ],
    text: { format: { type: "json_object" } },
  });

  const text = response.output_text?.trim();
  if (!text) throw new Error("Empty model output.");

  const parsed = JSON.parse(text);
  if (!Array.isArray(parsed.turns)) throw new Error("Invalid payload: missing turns array.");

  return { turns: parsed.turns.map((turn) => String(turn).trim()) };
}

function dryRunSample(params, variant) {
  const toneMap = {
    joy: {
      low: ["a bit happy", "quietly encouraged", "calmly hopeful"],
      high: ["extremely excited", "deeply joyful", "really energized"],
    },
    anger: {
      low: ["a little annoyed", "mildly frustrated", "somewhat irritated"],
      high: ["extremely angry", "deeply furious", "really outraged"],
    },
  };

  const topicKeyword = TOPIC_KEYWORDS[params.topic][variant % TOPIC_KEYWORDS[params.topic].length];
  const tone = toneMap[params.emotion][params.intensity][variant % 3];
  const topicLabel = TOPIC_LABELS[params.topic] || params.topic.replaceAll("_", " ");

  return {
    turns: [
      `I'm dealing with this ${topicLabel} situation, and right now I feel ${tone}.`,
      `The ${topicKeyword} detail is what started this and made it more complicated than I expected.`,
      `It's affecting my day-to-day mindset because this ${topicKeyword} issue keeps coming back.`,
      `I keep replaying what happened, and that keeps my reaction at the same ${params.intensity} intensity.`,
      `At this point, I still feel ${tone} and I need this ${topicKeyword} situation to be addressed.`,
    ],
  };
}

function buildFallbackTrajectory(params, variant) {
  const topicKeyword = TOPIC_KEYWORDS[params.topic][variant % TOPIC_KEYWORDS[params.topic].length];
  const topicLabel = TOPIC_LABELS[params.topic] || params.topic.replaceAll("_", " ");
  const emotionWord = params.emotion === "joy" ? "happy" : "angry";
  const intensityWord = params.intensity === "high" ? "very" : "a bit";

  return {
    turns: [
      `I'm talking about this ${topicLabel} issue, and I feel ${intensityWord} ${emotionWord} about it.`,
      `The ${topicKeyword} part of the situation is the main reason this started.`,
      `This ${topicKeyword} issue is affecting how I feel and what I focus on each day.`,
      `Thinking through the same details keeps my emotional reaction at a similar level.`,
      `I still feel ${intensityWord} ${emotionWord}, and I want resolution on this same issue.`,
    ],
  };
}

function normalizeTurns(generated) {
  const { turns } = generated || {};
  if (!Array.isArray(turns)) return null;
  if (turns.length !== 5) return null;
  const cleaned = turns.map((turn) => String(turn).trim());
  if (cleaned.some((turn) => !turn)) return null;
  return cleaned;
}

async function main() {
  const personasPerScenario = envInt("PERSONAS_PER_SCENARIO", 100);
  const maxRetries = envInt("PERSONA_MAX_RETRIES", 1);
  const duplicateRetries = envInt("PERSONA_DUPLICATE_RETRIES", 0);
  const requestDelayMs = envInt("PERSONA_REQUEST_DELAY_MS", 0);
  const duplicateThreshold = envFloat("PERSONA_DUPLICATE_THRESHOLD", 0.88);
  const temperature = envFloat("PERSONA_TEMPERATURE", 0.9);
  const outDir = process.env.PERSONA_OUTPUT_DIR || "data/personas";
  const checkpointEvery = envInt("PERSONA_CHECKPOINT_EVERY", 25);
  const dryRun = process.argv.includes("--dry-run");

  if (!dryRun && !process.env.OPENAI_API_KEY && !process.env.VITE_OPENAI_API_KEY) {
    throw new Error("Missing API key. Set OPENAI_API_KEY (preferred) or VITE_OPENAI_API_KEY.");
  }

  const client = dryRun
    ? null
    : new OpenAI({
        apiKey: process.env.OPENAI_API_KEY || process.env.VITE_OPENAI_API_KEY,
      });

  await ensureDir(outDir);

  const ts = new Date().toISOString().replace(/[:.]/g, "-");
  const jsonPath = path.join(outDir, `personas_${ts}.json`);
  const jsonlPath = path.join(outDir, `personas_${ts}.jsonl`);

  await fs.writeFile(jsonlPath, "", "utf8");

  const rows = [];
  const scenarioStats = {};
  const totalExpected =
    EMOTIONS.length * INTENSITIES.length * TOPICS.length * personasPerScenario;

  let personaCounter = 1;
  let completed = 0;
  let retryCount = 0;
  let rejectedCount = 0;
  let fallbackCount = 0;
  let interrupted = false;

  process.on("SIGINT", () => {
    interrupted = true;
    console.log("SIGINT received. Finishing current item and checkpointing progress...");
  });

  outerLoop:
  for (const emotion of EMOTIONS) {
    for (const intensity of INTENSITIES) {
      for (const topic of TOPICS) {
        if (interrupted) break outerLoop;
        const scenario = scenarioLabel(emotion, intensity, topic);
        console.log(`Starting scenario: ${scenario}`);

        const background = BACKGROUNDS[emotion][intensity][topic];
        scenarioStats[scenario] = { accepted: 0, rejected: 0, fallbacks: 0 };

        for (let i = 0; i < personasPerScenario; i += 1) {
          if (interrupted) break outerLoop;
          const personaId = `P${personaCounter}`;
          const personaStyle = samplePersonaStyle();
          const avoidOpenings = [];

          const params = {
            personaId,
            emotion,
            intensity,
            topic,
            background,
            personaStyle,
            avoidOpenings,
          };

          let generated = null;
          let finalReason = null;
          const totalAttempts = Math.max(1, maxRetries + duplicateRetries);

          for (let attempt = 1; attempt <= totalAttempts; attempt += 1) {
            try {
              const candidate = dryRun
                ? dryRunSample(params, i + attempt)
                : await generateOne(client, params, temperature);

              const normalizedTurns = normalizeTurns(candidate);
              if (!normalizedTurns) {
                generated = null;
                finalReason = "Invalid turns payload.";
                rejectedCount += 1;
                retryCount += 1;
                scenarioStats[scenario].rejected += 1;
                if (attempt < totalAttempts) await wait(requestDelayMs * attempt);
                continue;
              }

              generated = { turns: normalizedTurns };
              finalReason = null;
              break;
            } catch (error) {
              generated = null;
              finalReason = error?.message || "unknown error";
              retryCount += 1;
              if (attempt < totalAttempts) await wait(requestDelayMs * attempt);
            }
          }

          let fallbackUsed = false;
          if (!generated) {
            generated = buildFallbackTrajectory(params, i);
            fallbackUsed = true;
            fallbackCount += 1;
            scenarioStats[scenario].fallbacks += 1;
            console.warn(`Fallback used for ${personaId} (${scenario}): ${finalReason || "unknown reason"}`);
          }

          rows.push({
            persona_id: personaId,
            scenario_id: scenario,
            emotion,
            intensity,
            topic,
            background,
            persona_style: personaStyle,
            turns: generated.turns,
            metadata: {
              fallback_used: fallbackUsed,
              final_reason: finalReason,
            },
          });
          await fs.appendFile(jsonlPath, `${JSON.stringify(rows[rows.length - 1])}\n`, "utf8");

          scenarioStats[scenario].accepted += 1;
          personaCounter += 1;
          completed += 1;

          if (completed % 10 === 0 || completed === totalExpected) {
            console.log(`Progress: ${completed}/${totalExpected}`);
          }
          if (checkpointEvery > 0 && completed % checkpointEvery === 0) {
            await fs.writeFile(jsonPath, `${JSON.stringify(rows, null, 2)}\n`, "utf8");
            console.log(`Checkpoint saved at ${completed} records.`);
          }

          if (!dryRun && requestDelayMs > 0) await wait(requestDelayMs);
        }
      }
    }
  }

  await fs.writeFile(jsonPath, `${JSON.stringify(rows, null, 2)}\n`, "utf8");

  const total = rows.length;
  const scenarios = EMOTIONS.length * INTENSITIES.length * TOPICS.length;

  if (interrupted) {
    console.log(`Interrupted run saved ${total} personas across ${scenarios} scenarios.`);
  } else {
    console.log(`Generated ${total} personas across ${scenarios} scenarios.`);
  }
  console.log(`Retries: ${retryCount}`);
  console.log(`Rejected outputs: ${rejectedCount}`);
  console.log(`Fallbacks used: ${fallbackCount}`);
  console.log("Per-scenario acceptance stats:");
  console.log(JSON.stringify(scenarioStats, null, 2));
  console.log(`Wrote JSON:  ${jsonPath}`);
  console.log(`Wrote JSONL: ${jsonlPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
