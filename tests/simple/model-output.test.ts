import { assert, assertEquals } from "@std/assert";
import { assertObjectMatch } from "@std/assert/object-match";
import z from "zod";
import { Agent, ModelOutput, Tool } from "../../mod.ts";
import { addStreamItem } from "../../src/client.ts";
import type { ChatItem, StreamItem, WithTraceId } from "../../src/types.ts";
import { deterministicTestModel } from "./testing-model.ts";

Deno.test("ModelOutput from a tool terminates agent run and returns the value", async () => {
  const outputTool = new Tool({
    name: "output_tool",
    description: "Returns model output",
    parameters: z.void(),
    execute: () => new ModelOutput({ ids: ["a", "b", "c"] }),
  });

  const agent = new Agent({
    model: deterministicTestModel(),
    instructions: "You are a friendly assistant.",
    tools: [outputTool],
  });

  const run = await agent.run("Call output tool");
  run.output satisfies { ids: string[] } | undefined;
  assertEquals(run.output, { ids: ["a", "b", "c"] });
});

Deno.test("ModelOutput outputText is JSON-stringified for non-string values", async () => {
  const outputTool = new Tool({
    name: "output_tool",
    description: "Returns model output",
    parameters: z.void(),
    execute: () => new ModelOutput(["x", "y"]),
  });

  const agent = new Agent({
    model: deterministicTestModel(),
    instructions: "You are a friendly assistant.",
    tools: [outputTool],
  });

  const run = await agent.run("Call output tool");
  run.output satisfies string[] | undefined;
  assertEquals(run.output, ["x", "y"]);
  assertEquals(run.outputText, JSON.stringify(["x", "y"]));
});

Deno.test("ModelOutput with a string value uses it directly as outputText", async () => {
  const outputTool = new Tool({
    name: "output_tool",
    description: "Returns model output",
    parameters: z.void(),
    execute: () => new ModelOutput("done"),
  });

  const agent = new Agent({
    model: deterministicTestModel(),
    instructions: "You are a friendly assistant.",
    tools: [outputTool],
  });

  const run = await agent.run("Call output tool");
  run.output satisfies string | undefined;
  assertEquals(run.output, "done");
  assertEquals(run.outputText, "done");
});

Deno.test("ModelOutput is not retried even if tool has retries configured", async () => {
  let callCount = 0;
  const outputTool = new Tool({
    name: "output_tool",
    description: "Returns model output",
    parameters: z.void(),
    retries: 3,
    execute: () => {
      callCount++;
      return new ModelOutput("done");
    },
  });

  const agent = new Agent({
    model: deterministicTestModel(),
    instructions: "You are a friendly assistant.",
    tools: [outputTool],
  });

  const run = await agent.run("Call output tool");
  run.output satisfies string | undefined;
  assertEquals(run.output, "done");
  assertEquals(callCount, 1);
});

Deno.test("ModelOutput terminates even when called alongside a regular tool", async () => {
  const searchTool = new Tool({
    name: "search",
    description: "Search tool",
    parameters: z.string(),
    execute: (param) => `results for ${param}`,
  });

  const outputTool = new Tool({
    name: "output_tool",
    description: "Returns model output",
    parameters: z.void(),
    execute: () => new ModelOutput(["id1", "id2"]),
  });

  const agent = new Agent({
    model: deterministicTestModel(),
    instructions: "You are a friendly assistant.",
    tools: [searchTool, outputTool],
  });

  const run = await agent.run("Call output tool and search");
  run.output satisfies string[] | undefined;
  assertEquals(run.output, ["id1", "id2"]);
});

Deno.test("ModelOutput terminates streaming", async () => {
  const outputTool = new Tool({
    name: "output_tool",
    description: "Returns model output",
    parameters: z.void(),
    execute: () => new ModelOutput({ result: true }),
  });

  const agent = new Agent({
    model: deterministicTestModel(),
    instructions: "Model output stream test",
    tools: [outputTool],
  });

  const items: StreamItem[] = [];
  for await (const part of agent.stream("go")) {
    items.push(part);
  }

  // Should have the tool_use but no further output after ModelOutput
  const toolUses = items.filter((s) => s.type === "tool_use");
  assertEquals(toolUses.length, 1);

  const textItems = items.filter((s) => s.type === "delta_output_text");
  assertEquals(textItems.length, 0);
});

Deno.test("Structured output retry streams apology text and rebuilds history", async () => {
  const agent = new Agent({
    model: deterministicTestModel(),
    instructions: "Structured output retry stream test",
    output: z.object({ name: z.string() }),
  });

  const stream = agent.stream("name a cat");
  const rebuiltHistory: WithTraceId<ChatItem>[] = [];

  let run:
    | {
      output: { name: string };
      history: WithTraceId<ChatItem>[];
      outputText: string;
    }
    | undefined;

  while (true) {
    const next = await stream.next();
    if (next.done) {
      run = next.value;
      break;
    }
    addStreamItem(rebuiltHistory, next.value);
  }

  if (!run) throw new Error("stream did not return a final run result");

  assertEquals(run.output, { name: "Bingus" });
  assertEquals(run.outputText, JSON.stringify({ name: "Bingus" }));
  assertEquals(rebuiltHistory.length, 3);
  assertObjectMatch(rebuiltHistory[0], {
    type: "output_text",
    content: JSON.stringify({ name: 123 }),
  });
  assertObjectMatch(rebuiltHistory[2], {
    type: "output_text",
    content: JSON.stringify({ name: "Bingus" }),
  });

  const retryMessage = rebuiltHistory[1];
  assertEquals(retryMessage?.type, "output_text");
  if (!retryMessage || retryMessage.type !== "output_text") {
    throw new Error("expected retry message to be an output_text item");
  }
  assert(
    retryMessage.content.includes(
      "I will try again to produce a JSON response that conforms to the expected schema.",
    ),
  );

  assertEquals(
    run.history,
    rebuiltHistory,
  );
});

Deno.test("ModelOutput output type is a union across multiple output tools", async () => {
  const toolA = new Tool({
    name: "output_tool",
    description: "Returns string array",
    parameters: z.void(),
    execute: () => new ModelOutput(["a", "b"]),
  });

  const toolB = new Tool({
    name: "other_output",
    description: "Returns object",
    parameters: z.void(),
    execute: () => new ModelOutput({ score: 42 }),
  });

  const regularTool = new Tool({
    name: "search",
    description: "Search tool",
    parameters: z.string(),
    execute: (param) => `results for ${param}`,
  });

  const agent = new Agent({
    model: deterministicTestModel(),
    instructions: "You are a friendly assistant.",
    tools: [regularTool, toolA, toolB],
  });

  const run = await agent.run("Call output tool");
  // regular tool contributes `never` which drops out, leaving the union
  run.output satisfies string[] | { score: number } | undefined;
  assertEquals(run.output, ["a", "b"]);
});

Deno.test("Structured output ORs with ModelOutput types", async () => {
  const outputTool = new Tool({
    name: "output_tool",
    description: "Returns model output",
    parameters: z.void(),
    execute: () => new ModelOutput(["id1", "id2"]),
  });

  const agent = new Agent({
    model: deterministicTestModel(),
    instructions: "You are a friendly assistant.",
    output: z.object({ temperature: z.number() }),
    tools: [outputTool],
  });

  const run = await agent.run("Call output tool");
  run.output satisfies { temperature: number } | string[];
  assertEquals(run.output, ["id1", "id2"]);
});

Deno.test("ModelOutput carries history up to the point of termination", async () => {
  const outputTool = new Tool({
    name: "output_tool",
    description: "Returns model output",
    parameters: z.void(),
    execute: () => new ModelOutput("final"),
  });

  const agent = new Agent({
    model: deterministicTestModel(),
    instructions: "You are a friendly assistant.",
    tools: [outputTool],
  });

  const run = await agent.run("Call output tool");
  // History should contain the tool_use from the adapter's response
  const toolUses = run.history.filter((h) => h.type === "tool_use");
  assert(toolUses.length > 0, "history should contain the tool_use item");
});

Deno.test("ModelOutput settling while the model is still streaming is not an unhandled rejection", async () => {
  const outputTool = new Tool({
    name: "output_tool",
    description: "Returns model output",
    parameters: z.void(),
    execute: () => new ModelOutput("done"),
  });

  const agent = new Agent({
    model: {
      provider: "slow",
      model: "slow",
      async *stream() {
        yield { type: "tool_use", index: 0, tool_use_id: "id-slow", kind: outputTool.name };
        await new Promise((resolve) => setTimeout(resolve, 10));
        return { inputTokens: 0, outputTokens: 0 };
      },
    },
    instructions: "Model output during stream test",
    tools: [outputTool],
  });

  const rejections: unknown[] = [];
  const onRejection = (event: PromiseRejectionEvent) => {
    event.preventDefault();
    rejections.push(event.reason);
  };
  globalThis.addEventListener("unhandledrejection", onRejection);
  try {
    const run = await agent.run("go");
    assertEquals(run.output, "done");
    assertEquals(rejections, []);
  } finally {
    globalThis.removeEventListener("unhandledrejection", onRejection);
  }
});
