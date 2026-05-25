import { emotionEnhancerSystemPrompt, shortHistory } from "./emotion.mjs";

export async function nextResponderTurnEmotion(client, history, round, model, recognizedEmotion) {
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
