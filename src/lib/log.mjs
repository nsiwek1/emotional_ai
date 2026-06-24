import fs from "node:fs/promises";
import path from "node:path";
import { S3Client, PutObjectCommand } from "@aws-sdk/client-s3";

const DEFAULT_LOG_PATH = path.resolve(
  process.cwd(),
  "data/conversations/survey_sessions.jsonl"
);

// Default S3 key prefix, lab-convention style (lowercase, hyphenated, versionless).
// Override per deployment with the S3_PREFIX env var.
const DEFAULT_S3_PREFIX = "emotional-ai-anger-chatbot-natalia";

let dirEnsured = false;
let s3Client = null;

// Env is read lazily (not at module load): server.mjs runs dotenv.config() AFTER
// importing this module, so reading process.env at import time would miss .env values.
function s3Settings() {
  const bucket = process.env.S3_BUCKET;
  if (!bucket) return null;
  const prefix = (process.env.S3_PREFIX || DEFAULT_S3_PREFIX).replace(/^\/+|\/+$/g, "");
  const region = process.env.AWS_REGION || process.env.AWS_DEFAULT_REGION;
  return { bucket, prefix, region };
}

function getS3Client(region) {
  if (!s3Client) {
    s3Client = new S3Client(region ? { region } : {});
  }
  return s3Client;
}

// One immutable object per turn. Reconstruct a session by listing its prefix.
// participantId/sessionId are sanitized to a safe S3 key segment.
export function s3KeyForTurn(row, prefix = DEFAULT_S3_PREFIX) {
  const safe = (v) =>
    String(v ?? "unknown").replace(/[^A-Za-z0-9._-]/g, "-").slice(0, 128) || "unknown";
  const turn = Number.isFinite(row.turn) ? row.turn : 0;
  const padded = String(turn).padStart(2, "0");
  return `${prefix}/sessions/${safe(row.sessionId)}/turn-${padded}.json`;
}

async function ensureDir(filePath) {
  if (dirEnsured) return;
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  dirEnsured = true;
}

async function writeLocal(row, logPath) {
  await ensureDir(logPath);
  await fs.appendFile(logPath, `${JSON.stringify(row)}\n`, "utf8");
}

async function writeS3(row) {
  const settings = s3Settings();
  if (!settings) return; // S3 not configured: local-only (e.g. local dev)
  const client = getS3Client(settings.region);
  await client.send(
    new PutObjectCommand({
      Bucket: settings.bucket,
      Key: s3KeyForTurn(row, settings.prefix),
      Body: JSON.stringify(row, null, 2),
      ContentType: "application/json",
    })
  );
}

// Writes the turn to the local JSONL backup and (if S3_BUCKET is set) to S3.
// Both are attempted; throws only if a configured store fails, so the caller's
// existing try/catch logs a warning while the chat continues uninterrupted.
export async function appendSessionTurn(
  row,
  logPath = process.env.SESSION_LOG_PATH || DEFAULT_LOG_PATH
) {
  const stamped = { ts: new Date().toISOString(), ...row };
  const [local, s3] = await Promise.allSettled([
    writeLocal(stamped, logPath),
    writeS3(stamped),
  ]);
  const errors = [];
  if (local.status === "rejected") {
    errors.push(`local: ${local.reason?.message || local.reason}`);
  }
  if (s3.status === "rejected") {
    errors.push(`s3: ${s3.reason?.message || s3.reason}`);
  }
  if (errors.length) {
    throw new Error(`appendSessionTurn failed -> ${errors.join("; ")}`);
  }
}
