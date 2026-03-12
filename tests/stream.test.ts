import { Agent, Tool } from "../mod.ts";
import { assertEquals } from "@std/assert";
import z from "zod";
import { delay } from "@std/async/delay";
import { enableDebugMode } from "../src/constants.ts";
import type { ChatItem, StreamItem } from "../src/types.ts";
import { addStreamItem } from "../src/client.ts";
import { testingAdapter } from "./utils/testing-adapter.ts";

enableDebugMode();

Deno.test("Basic streaming test", async () => {
  const agent = new Agent({
    adapter: testingAdapter,
    model: "deterministic",
    instructions: "Basic test",
  });
  const run = agent.stream("<nothing>");

  let count = 0;
  const output: ChatItem[] = [];
  for await (const part of run) {
    addStreamItem(output, part);
    count++;
  }

  assertEquals(count, 18);
  assertEquals(output, [{
    type: "output_text",
    content: "Basic test worked!",
  }]);
});

Deno.test("Parallel tool calls are streamed one by one in settlement order", async () => {
  const fastTool = new Tool({
    name: "fast_tool",
    description: "Resolves immediately",
    parameters: z.string(),
    execute: () => Promise.resolve("fast_result"),
  });
  const slowTool = new Tool({
    name: "slow_tool",
    description: "Resolves after a short delay",
    parameters: z.string(),
    execute: async () => {
      await delay(50);
      return "slow_result";
    },
  });

  const agent = new Agent({
    adapter: testingAdapter,
    model: "deterministic",
    instructions: "Parallel tool test",
    tools: [slowTool, fastTool],
  });

  const streamItems: StreamItem[] = [];
  const output: ChatItem[] = [];

  for await (const part of agent.stream("go")) {
    streamItems.push(part);
    addStreamItem(output, part);
  }

  // Two tool_use items must have been streamed
  const toolUseItems = streamItems.filter((s) => s.type === "tool_use");
  assertEquals(toolUseItems.length, 2);

  // Two tool_result_text items must have been streamed
  const toolResultItems = streamItems.filter((s) =>
    s.type === "tool_result_text"
  );
  assertEquals(toolResultItems.length, 2);

  // Each tool result must carry a unique stream index (streamed one-by-one,
  // not batched at the same position).
  const resultIndices = toolResultItems.map((r) => r.index);
  assertEquals(
    new Set(resultIndices).size,
    2,
    "tool results must have distinct stream indices",
  );

  // The fast tool result must appear before the slow tool result in the stream,
  // proving that iteratePromiseArray yields in settlement order, not submission order.
  const fastResultPos = streamItems.findIndex(
    (s) => s.type === "tool_result_text" && s.tool_use_id === "id-fast",
  );
  const slowResultPos = streamItems.findIndex(
    (s) => s.type === "tool_result_text" && s.tool_use_id === "id-slow",
  );
  assertEquals(
    fastResultPos < slowResultPos,
    true,
    "fast tool result must be streamed before slow tool result",
  );

  // The final assembled history must contain both tool results
  const toolResults = output.filter((c) => c.type === "tool_result_text");
  assertEquals(toolResults.length, 2);
});
