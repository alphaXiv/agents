import { assertEquals } from "@std/assert";
import z from "zod";
import { Agent, type ChatItem, Tool } from "../../mod.ts";
import type { Adapter, AdapterStreamOptions } from "../../src/adapters/adapter.ts";
import { convertChatItemsToStream } from "../../src/client.ts";
import type { AdapterStreamIterator } from "../../src/types.ts";

/** Calls a tool on the first run only, recording every history and instructions it is given. */
function recordingToolModel(): Adapter<unknown, unknown> & {
  histories: ChatItem[][];
  instructions: string[];
} {
  const histories: ChatItem[][] = [];
  const instructions: string[] = [];
  let calls = 0;

  return {
    provider: "deterministic",
    model: "deterministic",
    histories,
    instructions,
    stream(options: AdapterStreamOptions<unknown, unknown>): AdapterStreamIterator {
      histories.push([...options.history]);
      instructions.push(options.instructions);
      calls += 1;

      if (calls === 1) {
        return convertChatItemsToStream({
          items: [{ type: "tool_use", tool_use_id: "id-1", kind: "lookup" }],
          inputTokens: 0,
          outputTokens: 0,
        });
      }

      return convertChatItemsToStream({
        items: [{ type: "output_text", content: "done" }],
        inputTokens: 0,
        outputTokens: 0,
      });
    },
  };
}

Deno.test("Lone surrogates are replaced before reaching the adapter", async () => {
  const model = recordingToolModel();
  const agent = new Agent({
    model,
    instructions: "Instructions \ud83d end",
    tools: [
      new Tool({
        name: "lookup",
        description: "Returns text with a lone surrogate",
        parameters: z.void(),
        execute: () => "result \ud83d end",
      }),
    ],
  });

  const input: ChatItem[] = [{ type: "input_text", content: "user \ud83d end" }];
  await agent.run(input);

  assertEquals(model.histories.length, 2);
  assertEquals(model.instructions, ["Instructions � end", "Instructions � end"]);

  const [first, second] = model.histories;
  assertEquals(first[0], { type: "input_text", content: "user � end" });
  assertEquals(second[0], { type: "input_text", content: "user � end" });

  const toolResult = second.find((item) => item.type === "tool_result_text");
  assertEquals(toolResult?.content, "result � end");

  assertEquals(input[0], { type: "input_text", content: "user \ud83d end" });
});
