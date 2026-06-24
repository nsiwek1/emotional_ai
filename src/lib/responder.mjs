import { emotionEnhancerSystemPrompt, shortHistory } from "./emotion.mjs";

const OPENER_INSTRUCTION =
  "The participant has just described an event in writing (below). Write a single opening message (1-3 sentences) that acknowledges the event and invites them to say more. Do not summarize it back at length. Do not include greetings like \"Hi\". Do not mention that you read a description. Speak directly to them.";

export async function openerResponderTurnEmotion(client, eventDescription, model) {
  const system = emotionEnhancerSystemPrompt();
  const response = await client.responses.create({
    model,
    input: [
      { role: "system", content: system },
      {
        role: "user",
        content: `${OPENER_INSTRUCTION}\n\nEvent description:\n${eventDescription}`,
      },
    ],
  });

  const text = response.output_text?.trim();
  if (!text) throw new Error("Opener agent returned empty text.");
  return text;
}

export async function openerResponderTurnBaseline(client, eventDescription, model) {
  const response = await client.responses.create({
    model,
    input: [
      {
        role: "user",
        content: `${OPENER_INSTRUCTION}\n\nEvent description:\n${eventDescription}`,
      },
    ],
  });

  const text = response.output_text?.trim();
  if (!text) throw new Error("Opener agent returned empty text.");
  return text;
}

export async function nextResponderTurnEmotion(client, history, round, model) {
  const system = emotionEnhancerSystemPrompt();
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

export async function nextResponderTurnBaseline(client, history, round, model) {
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
