import { assertRejects } from "@std/assert";
import type { OpenAICompletionsClient } from "../../src/adapters/openai_completions/adapter.ts";
import { sidModel } from "../../src/adapters/sid/adapter.ts";

function createUnusedClient(): OpenAICompletionsClient {
  return {
    chat: {
      completions: {
        stream() {
          throw new Error("client should not be called");
        },
      },
    },
  } as unknown as OpenAICompletionsClient;
}

Deno.test("SID rejects file inputs by default", async () => {
  const adapter = sidModel({
    model: "sid-1",
    client: createUnusedClient(),
  });

  const stream = adapter.stream({
    history: [{ type: "input_file", kind: "image/png", content: "https://example.com/cat.png" }],
    instructions: "test",
    tools: [],
    signal: AbortSignal.abort(),
  });

  await assertRejects(
    () => stream.next(),
    Error,
    "does not support media type",
  );
});
