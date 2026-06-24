import path from "node:path";
import { PassThrough } from "node:stream";
import Fastify from "fastify";
import cors from "@fastify/cors";
import rateLimit from "@fastify/rate-limit";
import OpenAI from "openai";
import dotenv from "dotenv";
import { classifyEmotionWithHF, envInt } from "../src/lib/emotion.mjs";
import { buildOpenerInput, buildNextInput } from "../src/lib/responder.mjs";
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
// Token streaming is on by default. Set STREAMING=0 to fall back to whole-reply
// JSON responses without a code change (the frontend detects which it got).
const STREAMING = (process.env.STREAMING ?? "1") !== "0";
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

// Awaits the (logging-only) HF classification and appends the turn. Never throws
// into the request path.
async function logTurn(request, hfPromise, logRow, assistant) {
  try {
    const { recognizedEmotion, hf } = await hfPromise;
    if (hf?.error) {
      request.log.warn({ hfError: hf.error, sessionId: logRow.sessionId }, "HF classify failed; neutral fallback");
    }
    await appendSessionTurn({
      ...logRow,
      assistant,
      recognized_emotion: recognizedEmotion,
      hf_emotion: hf,
      model: MODEL,
    });
  } catch (err) {
    request.log.warn({ err: err?.message, sessionId: logRow.sessionId }, "Failed to append session log");
  }
}

// Streams the model reply to the client as SSE while accumulating the full text
// for logging. HF classification runs concurrently (it never steers the reply).
async function streamReply(request, reply, { classifyText, input, logRow, turn, isFinal }) {
  const hfPromise = classifyEmotionWithHF(classifyText, { sleepMsBetween: envInt("HF_SLEEP_MS", 0) });

  let llmStream;
  try {
    llmStream = await openai.responses.create({ model: MODEL, input, stream: true });
  } catch (err) {
    request.log.error({ err: err?.message, sessionId: logRow.sessionId }, "OpenAI stream create failed");
    hfPromise.catch(() => {});
    return reply.code(502).send({ error: err?.message || "OpenAI request failed", code: "openai_failed" });
  }

  reply.header("Content-Type", "text/event-stream; charset=utf-8");
  reply.header("Cache-Control", "no-cache, no-transform");
  reply.header("X-Accel-Buffering", "no");

  const sse = new PassThrough();
  sse.on("error", () => {});
  let clientGone = false;
  const send = (obj) => {
    if (!clientGone && !sse.writableEnded) sse.write(`data: ${JSON.stringify(obj)}\n\n`);
  };
  request.raw.on("close", () => {
    clientGone = true;
    try { llmStream.controller?.abort?.(); } catch { /* noop */ }
  });

  (async () => {
    let full = "";
    let errored = false;
    try {
      for await (const event of llmStream) {
        if (clientGone) break;
        if (event.type === "response.output_text.delta" && event.delta) {
          full += event.delta;
          send({ type: "delta", text: event.delta });
        }
      }
    } catch (err) {
      errored = true;
      request.log.error({ err: err?.message, sessionId: logRow.sessionId }, "OpenAI stream errored mid-flight");
    }
    full = full.trim();
    if (!full) {
      send({ type: "error", error: errored ? "stream_failed" : "empty_response" });
    } else {
      send({ type: "done", turn, isFinal, reply: full });
    }
    if (!sse.writableEnded) sse.end();
    if (full && !clientGone) await logTurn(request, hfPromise, logRow, full);
    else hfPromise.catch(() => {});
  })();

  return reply.send(sse);
}

// Non-streaming fallback (STREAMING=0): a single JSON response with the whole reply.
async function jsonReply(request, reply, { classifyText, input, logRow, turn, isFinal }) {
  const hfPromise = classifyEmotionWithHF(classifyText, { sleepMsBetween: envInt("HF_SLEEP_MS", 0) });
  let full;
  try {
    const response = await openai.responses.create({ model: MODEL, input });
    full = response.output_text?.trim();
    if (!full) throw new Error("Responder returned empty text.");
  } catch (err) {
    request.log.error({ err: err?.message, sessionId: logRow.sessionId }, "OpenAI call failed");
    hfPromise.catch(() => {});
    return reply.code(502).send({ error: err?.message || "OpenAI request failed", code: "openai_failed" });
  }
  await logTurn(request, hfPromise, logRow, full);
  return { reply: full, turn, isFinal };
}

function handleTurn(request, reply, ctx) {
  return STREAMING ? streamReply(request, reply, ctx) : jsonReply(request, reply, ctx);
}

fastify.post("/chat/open", { schema: { body: openBodySchema } }, async (request, reply) => {
  const { sessionId, participantId, condition, eventDescription } = request.body;
  return handleTurn(request, reply, {
    classifyText: eventDescription,
    input: buildOpenerInput(condition, eventDescription),
    logRow: {
      sessionId,
      participantId,
      condition,
      turn: 0,
      user: null,
      is_opening: true,
      event_description: eventDescription,
      is_final: false,
    },
    turn: 0,
    isFinal: false,
  });
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

  const isFinal = turn >= MAX_TURNS;
  return handleTurn(request, reply, {
    classifyText: userMessage,
    input: buildNextInput(condition, transcript, turn),
    logRow: {
      sessionId,
      participantId,
      condition,
      turn,
      user: userMessage,
      is_final: isFinal,
    },
    turn,
    isFinal,
  });
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
