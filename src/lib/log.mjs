import fs from "node:fs/promises";
import path from "node:path";

const DEFAULT_LOG_PATH = path.resolve(
  process.cwd(),
  "data/conversations/survey_sessions.jsonl"
);

let dirEnsured = false;

async function ensureDir(filePath) {
  if (dirEnsured) return;
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  dirEnsured = true;
}

export async function appendSessionTurn(row, logPath = process.env.SESSION_LOG_PATH || DEFAULT_LOG_PATH) {
  await ensureDir(logPath);
  const line = JSON.stringify({ ts: new Date().toISOString(), ...row });
  await fs.appendFile(logPath, `${line}\n`, "utf8");
}
