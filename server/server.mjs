import path from "node:path";
import Fastify from "fastify";
import cors from "@fastify/cors";
import rateLimit from "@fastify/rate-limit";
import OpenAI from "openai";
import dotenv from "dotenv";
import { classifyEmotionWithHF, envInt } from "../src/lib/emotion.mjs";
import {
  nextResponderTurnEmotion,
  nextResponderTurnBaseline,
  openerResponderTurnEmotion,
  openerResponderTurnBaseline,
} from "../src/lib/responder.mjs";
import { appendSessionTurn } from "../src/lib/log.mjs";

dotenv.config({ path: path.resolve(process.cwd(), ".env") });

const MAX_TURNS = 5;
const MAX_USER_MESSAGE = 2000;
const MAX_HISTORY = 12;
const MAX_EVENT_DESCRIPTION = 4000;

const OPENAI_KEY = process.env.OPENAI_API_KEY || process.env.VITE_OPENAI_API_KEY;
if (!OPENAI_KEY) {
  console.error("Missing OPENAI_API_KEY (or VITE_OPENAI_API_KEY) in environment.");
  process.exit(1);
}
const MODEL = process.env.CONVERSATION_MODEL || "gpt-4o-mini";
const openai = new OpenAI({ apiKey: OPENAI_KEY });

const allowedOrigins = (process.env.ALLOWED_ORIGINS || "http://localhost:5173")
  .split(",")
  .map((s) => s.trim())
  .filter(Boolean);

const fastify = Fastify({ logger: true });

await fastify.register(cors, {
  origin: (origin, cb) => {
    if (!origin) return cb(null, true);
    if (allowedOrigins.includes("*") || allowedOrigins.includes(origin)) {
      return cb(null, true);
    }
    return cb(new Error(`Origin ${origin} not allowed`), false);
  },
  methods: ["GET", "POST", "OPTIONS"],
});

await fastify.register(rateLimit, {
  max: 30,
  timeWindow: "1 minute",
  keyGenerator: (req) => {
    const pid =
      req.body && typeof req.body === "object" ? req.body.participantId : null;
    return pid || req.ip;
  },
});

const chatBodySchema = {
  type: "object",
  required: ["sessionId", "participantId", "condition", "turn", "history", "userMessage"],
  additionalProperties: false,
  properties: {
    sessionId: { type: "string", minLength: 1, maxLength: 64 },
    participantId: { type: "string", minLength: 1, maxLength: 64 },
    condition: { type: "string", enum: ["emotion_enhanced", "baseline"] },
    turn: { type: "integer", minimum: 1, maximum: MAX_TURNS },
    history: {
      type: "array",
      maxItems: MAX_HISTORY,
      items: {
        type: "object",
        required: ["role", "content"],
        additionalProperties: false,
        properties: {
          role: { type: "string", enum: ["user", "assistant"] },
          content: { type: "string", maxLength: MAX_USER_MESSAGE },
        },
      },
    },
    userMessage: { type: "string", minLength: 1, maxLength: MAX_USER_MESSAGE },
  },
};

const openBodySchema = {
  type: "object",
  required: ["sessionId", "participantId", "condition", "eventDescription"],
  additionalProperties: false,
  properties: {
    sessionId: { type: "string", minLength: 1, maxLength: 64 },
    participantId: { type: "string", minLength: 1, maxLength: 64 },
    condition: { type: "string", enum: ["emotion_enhanced", "baseline"] },
    eventDescription: { type: "string", minLength: 1, maxLength: MAX_EVENT_DESCRIPTION },
  },
};

fastify.get("/healthz", async () => ({ ok: true }));

fastify.post("/chat/open", { schema: { body: openBodySchema } }, async (request, reply) => {
  const { sessionId, participantId, condition, eventDescription } = request.body;

  // Classified for logging only (both arms); the label never steers the bot.
  const { recognizedEmotion, hf } = await classifyEmotionWithHF(eventDescription, {
    sleepMsBetween: envInt("HF_SLEEP_MS", 0),
  });
  if (hf?.error) {
    request.log.warn({ hfError: hf.error, sessionId }, "HF classify failed on opener; falling back to neutral");
  }

  let openingText;
  try {
    if (condition === "emotion_enhanced") {
      openingText = await openerResponderTurnEmotion(openai, eventDescription, MODEL);
    } else {
      openingText = await openerResponderTurnBaseline(openai, eventDescription, MODEL);
    }
  } catch (err) {
    request.log.error({ err: err?.message, sessionId }, "OpenAI opener call failed");
    return reply.code(502).send({
      error: err?.message || "OpenAI request failed",
      code: "openai_failed",
    });
  }

  try {
    await appendSessionTurn({
      sessionId,
      participantId,
      condition,
      turn: 0,
      user: null,
      assistant: openingText,
      is_opening: true,
      event_description: eventDescription,
      recognized_emotion: recognizedEmotion,
      hf_emotion: hf,
      model: MODEL,
      is_final: false,
    });
  } catch (err) {
    request.log.warn({ err: err?.message, sessionId }, "Failed to append opener log");
  }

  return { reply: openingText, turn: 0, isFinal: false };
});

fastify.post("/chat", { schema: { body: chatBodySchema } }, async (request, reply) => {
  const { sessionId, participantId, condition, turn, history, userMessage } = request.body;

  const transcript = [];
  let round = 1;
  for (let i = 0; i < history.length; i += 1) {
    const msg = history[i];
    const speaker = msg.role === "user" ? "persona_agent" : "responder_agent";
    transcript.push({ round, speaker, content: msg.content });
    if (msg.role === "assistant") round += 1;
  }
  transcript.push({ round: turn, speaker: "persona_agent", content: userMessage });

  // Classified for logging only (both arms); the label never steers the bot.
  const { recognizedEmotion, hf } = await classifyEmotionWithHF(userMessage, {
    sleepMsBetween: envInt("HF_SLEEP_MS", 0),
  });
  if (hf?.error) {
    request.log.warn({ hfError: hf.error, sessionId }, "HF classify failed; falling back to neutral");
  }

  let replyText;
  try {
    if (condition === "emotion_enhanced") {
      replyText = await nextResponderTurnEmotion(openai, transcript, turn, MODEL);
    } else {
      replyText = await nextResponderTurnBaseline(openai, transcript, turn, MODEL);
    }
  } catch (err) {
    request.log.error({ err: err?.message, sessionId }, "OpenAI call failed");
    return reply.code(502).send({
      error: err?.message || "OpenAI request failed",
      code: "openai_failed",
    });
  }

  const isFinal = turn >= MAX_TURNS;

  try {
    await appendSessionTurn({
      sessionId,
      participantId,
      condition,
      turn,
      user: userMessage,
      assistant: replyText,
      recognized_emotion: recognizedEmotion,
      hf_emotion: hf,
      model: MODEL,
      is_final: isFinal,
    });
  } catch (err) {
    request.log.warn({ err: err?.message, sessionId }, "Failed to append session log");
  }

  return { reply: replyText, turn, isFinal };
});

fastify.setErrorHandler((err, request, reply) => {
  if (err.validation) {
    return reply.code(400).send({ error: err.message, code: "bad_request" });
  }
  if (err.statusCode === 429) {
    return reply.code(429).send({ error: "Too many requests", code: "rate_limited" });
  }
  request.log.error({ err: err?.message }, "Unhandled server error");
  return reply.code(500).send({ error: "Server error", code: "server_error" });
});

const port = envInt("PORT", 3001);
try {
  await fastify.listen({ port, host: "0.0.0.0" });
} catch (err) {
  console.error(err);
  process.exit(1);
}
