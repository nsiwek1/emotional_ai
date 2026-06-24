import { emotionEnhancerSystemPrompt, shortHistory } from "./emotion.mjs";

const OPENER_INSTRUCTION =
  "The participant has just described an event in writing (below). Write a single opening message (1-3 sentences) that acknowledges the event and invites them to say more. Do not summarize it back at length. Do not include greetings like \"Hi\". Do not mention that you read a description. Speak directly to them.";

function openerUserContent(eventDescription) {
  return `${OPENER_INSTRUCTION}\n\nEvent description:\n${eventDescription}`;
}

function nextUserContent(history, round) {
  return `Round ${round}. Continue this conversation naturally.\nConversation so far:\n${JSON.stringify(
    shortHistory(history)
  )}\n\nWrite the assistant's next reply.`;
}

// Builds the Responses API `input`. emotion_enhanced gets the amplifying system
// prompt; baseline gets none (unchanged behavior). The server uses these for
// both the streaming and the (kill-switch) non-streaming paths.
export function buildOpenerInput(condition, eventDescription) {
  const user = { role: "user", content: openerUserContent(eventDescription) };
  return condition === "emotion_enhanced"
    ? [{ role: "system", content: emotionEnhancerSystemPrompt() }, user]
    : [user];
}

export function buildNextInput(condition, history, round) {
  const user = { role: "user", content: nextUserContent(history, round) };
  return condition === "emotion_enhanced"
    ? [{ role: "system", content: emotionEnhancerSystemPrompt() }, user]
    : [user];
}

// Non-streaming variants, kept for the offline batch scripts
// (multi-agent-conversations*.mjs). The live server streams instead.
async function completeOnce(client, model, input) {
  const response = await client.responses.create({ model, input });
  const text = response.output_text?.trim();
  if (!text) throw new Error("Responder returned empty text.");
  return text;
}

export function openerResponderTurnEmotion(client, eventDescription, model) {
  return completeOnce(client, model, buildOpenerInput("emotion_enhanced", eventDescription));
}

export function openerResponderTurnBaseline(client, eventDescription, model) {
  return completeOnce(client, model, buildOpenerInput("baseline", eventDescription));
}

export function nextResponderTurnEmotion(client, history, round, model) {
  return completeOnce(client, model, buildNextInput("emotion_enhanced", history, round));
}

export function nextResponderTurnBaseline(client, history, round, model) {
  return completeOnce(client, model, buildNextInput("baseline", history, round));
}
