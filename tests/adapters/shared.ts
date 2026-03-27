import { assert, assertEquals } from "@std/assert";
import z from "zod";
import { Agent, type AgentRunResult, Tool } from "../../mod.ts";
import type { Model } from "../../src/adapters/model.ts";
import type {
  AdapterStreamIterator,
  AgentStreamIterator,
  ChatItem,
  ProviderStreamMetadata,
  StreamItem,
  WithTraceId,
} from "../../src/types.ts";

export const INTEGRATION_TIMEOUT_MS = 120_000;

export function createToolFixtures() {
  const state = {
    voidCalls: 0,
    queryCalls: [] as string[],
  };

  const pingTool = new Tool({
    name: "ping_void",
    description: "Call this tool exactly once with no parameters. It returns pong.",
    parameters: z.void(),
    execute: () => {
      state.voidCalls += 1;
      return "pong";
    },
  });

  const echoTool = new Tool({
    name: "echo_query",
    description: 'Call this tool exactly once with {"query":"cats"}. It returns echo:cats.',
    parameters: z.object({ query: z.string() }).strict(),
    execute: ({ query }) => {
      state.queryCalls.push(query);
      return `echo:${query}`;
    },
  });

  return {
    state,
    pingTool,
    echoTool,
    query: "cats",
    pingResult: "pong",
    echoResult: "echo:cats",
    marker: "LIVE_TOOL_STREAM_COMPLETE",
  };
}

export function createStructuredOutputFixtures() {
  const output = z.object({
    animal: z.literal("cats"),
    count: z.literal(2),
    source: z.literal("live_test"),
  }).strict();

  return {
    output,
    expected: {
      animal: "cats",
      count: 2,
      source: "live_test",
    },
    instructions: "You are running a live structured output integration test. Use the provided schema only.",
    prompt: 'Return structured output with animal "cats", count 2, and source "live_test".',
  };
}

function createCalculatorFixtures() {
  const state = {
    calls: [] as {
      operation: "add" | "multiply" | "divide" | "subtract";
      left: number;
      right: number;
    }[],
  };

  const calculator = new Tool({
    name: "Calculating...",
    description: "A simple calculator to make math operations easier!",
    parameters: z.object({
      operation: z.enum(["add", "multiply", "divide", "subtract"]).describe(
        "The operator you want to calculate with",
      ),
      left: z.number(),
      right: z.number(),
    }),
    execute: ({ operation, left, right }) => {
      state.calls.push({ operation, left, right });

      if (operation === "add") {
        return (left + right).toString();
      } else if (operation === "multiply") {
        return (left * right).toString();
      } else if (operation === "divide") {
        return (left / right).toString();
      } else if (operation === "subtract") {
        return (left - right).toString();
      }

      operation satisfies never;
      return "";
    },
  });

  return { calculator, state };
}

export async function collectAdapterStream(stream: AdapterStreamIterator) {
  const items: StreamItem[] = [];

  while (true) {
    const next = await stream.next();
    if (next.done) {
      return { items, metadata: next.value };
    }
    items.push(next.value);
  }
}

export async function collectAgentStream(stream: AgentStreamIterator<unknown>) {
  const items: WithTraceId<StreamItem>[] = [];

  while (true) {
    const next = await stream.next();
    if (next.done) {
      return { items, result: next.value as AgentRunResult<unknown> };
    }
    items.push(next.value);
  }
}

export function assertHasUsage(metadata: ProviderStreamMetadata) {
  assert(metadata.inputTokens != null || metadata.outputTokens != null, "expected provider usage metadata");
}

function findToolUse<T extends StreamItem | WithTraceId<StreamItem>>(items: T[], toolName: string) {
  return items.find((item): item is Extract<T, { type: "tool_use" }> =>
    item.type === "tool_use" && item.kind === toolName
  );
}

function findToolResult<T extends StreamItem | WithTraceId<StreamItem>>(items: T[], toolUseId: string) {
  return items.find((item): item is Extract<T, { type: "tool_result_text" }> => {
    return item.type === "tool_result_text" && item.tool_use_id === toolUseId;
  });
}

export async function runAdapterToolStreamingTest(
  t: Deno.TestContext,
  options: {
    stream: AdapterStreamIterator;
    toolName: string;
    expectedContentSubstring?: string;
    expectVoid?: boolean;
  },
) {
  let items: StreamItem[] = [];
  let metadata: ProviderStreamMetadata | undefined;

  const collectedAdapterStream = await t.step("collect adapter stream", async () => {
    const collected = await collectAdapterStream(options.stream);
    items = collected.items;
    metadata = collected.metadata;
  });

  if (!collectedAdapterStream) return;

  await t.step("assert adapter streamed tool use", () => {
    assert(items.length > 0, "expected streamed items");
    assert(
      items.some((item) => item.type === "tool_use_start" && item.kind === options.toolName),
      `expected ${options.toolName} tool_use_start`,
    );

    const toolUse = findToolUse(items, options.toolName);
    assert(toolUse, `expected ${options.toolName} tool_use item`);

    if (options.expectVoid) {
      assertEquals(toolUse.content, undefined);
      return;
    }

    assert(
      toolUse.content?.includes(options.expectedContentSubstring ?? ""),
      `expected ${options.toolName} content to include ${options.expectedContentSubstring}`,
    );
  });

  await t.step("assert adapter usage metadata", () => {
    assert(metadata, "expected stream metadata");
    assertHasUsage(metadata);
  });
}

export async function runAgentToolStreamingTest(
  t: Deno.TestContext,
  options: {
    model: Model;
  },
) {
  const fixtures = createToolFixtures();

  const instructions = [
    "You are running a live SDK integration test.",
    `Call ${fixtures.pingTool.normalizedName} exactly once with no parameters.`,
    `Call ${fixtures.echoTool.normalizedName} exactly once with {\"query\":\"${fixtures.query}\"}.`,
    `After all tool calls, respond with exactly ${fixtures.marker} and nothing else.`,
  ].join(" ");

  const agent = new Agent({
    model: options.model,
    instructions,
    tools: [fixtures.pingTool, fixtures.echoTool],
  });

  let items: WithTraceId<StreamItem>[] = [];
  let result: AgentRunResult<unknown> | undefined;

  const collectedAgentStream = await t.step("collect agent stream", async () => {
    const collected = await collectAgentStream(agent.stream("Run the integration test now.", {
      signal: AbortSignal.timeout(INTEGRATION_TIMEOUT_MS),
    }));
    items = collected.items;
    result = collected.result;
  });

  if (!collectedAgentStream) return;

  await t.step("assert streamed void tool invocation", () => {
    assert(
      items.some((item) => item.type === "tool_use_start" && item.kind === fixtures.pingTool.normalizedName),
      `expected ${fixtures.pingTool.normalizedName} tool_use_start`,
    );

    const toolUse = findToolUse(items, fixtures.pingTool.normalizedName);
    assert(toolUse, `expected ${fixtures.pingTool.normalizedName} tool_use item`);
  });

  await t.step("assert streamed parameterized tool invocation", () => {
    assert(
      items.some((item) => item.type === "tool_use_start" && item.kind === fixtures.echoTool.normalizedName),
      `expected ${fixtures.echoTool.normalizedName} tool_use_start`,
    );

    const toolUse = findToolUse(items, fixtures.echoTool.normalizedName);
    assert(toolUse, `expected ${fixtures.echoTool.normalizedName} tool_use item`);
    assert(
      toolUse.content?.includes(fixtures.query),
      `expected ${fixtures.echoTool.normalizedName} content to include ${fixtures.query}`,
    );
  });

  await t.step("assert streamed void tool result", () => {
    const toolUse = findToolUse(items, fixtures.pingTool.normalizedName);
    assert(toolUse, `missing ${fixtures.pingTool.normalizedName} tool use for result assertion`);

    const toolResult = findToolResult(items, toolUse.tool_use_id);
    assert(toolResult, `expected ${fixtures.pingTool.normalizedName} tool_result_text`);
    assertEquals(toolResult.content, fixtures.pingResult);
  });

  await t.step("assert streamed parameterized tool result", () => {
    const toolUse = findToolUse(items, fixtures.echoTool.normalizedName);
    assert(toolUse, `missing ${fixtures.echoTool.normalizedName} tool use for result assertion`);

    const toolResult = findToolResult(items, toolUse.tool_use_id);
    assert(toolResult, `expected ${fixtures.echoTool.normalizedName} tool_result_text`);
    assertEquals(toolResult.content, fixtures.echoResult);
  });

  await t.step("assert final streamed output", () => {
    assert(result, "expected final agent result");
    assert(items.some((item) => item.type === "delta_output_text"), "expected streamed text output");
    assert(result.outputText.trim().endsWith(fixtures.marker), `expected final output to end with ${fixtures.marker}`);
  });

  await t.step("assert tool execution counts", () => {
    assertEquals(fixtures.state.voidCalls, 1);
    assertEquals(fixtures.state.queryCalls, [fixtures.query]);
  });
}

export async function runStructuredOutputStreamingTest(
  t: Deno.TestContext,
  options: {
    model: Model;
    expectPreview?: boolean;
  },
) {
  const fixtures = createStructuredOutputFixtures();
  const agent = new Agent({
    model: options.model,
    instructions: fixtures.instructions,
    output: fixtures.output,
  });

  let items: WithTraceId<StreamItem>[] = [];
  let result: AgentRunResult<unknown> | undefined;

  const collectedStructuredOutput = await t.step("collect structured output stream", async () => {
    const collected = await collectAgentStream(agent.stream(fixtures.prompt, {
      signal: AbortSignal.timeout(INTEGRATION_TIMEOUT_MS),
    }));
    items = collected.items;
    result = collected.result;
  });

  if (!collectedStructuredOutput) return;

  await t.step("assert structured output was streamed", () => {
    const hasPreview = items.some((item) => item.type === "delta_output_preview");
    const hasText = items.some((item) => item.type === "delta_output_text");

    if (options.expectPreview) {
      assert(hasPreview, "expected structured output preview chunks");
    }

    assert(hasPreview || hasText, "expected structured output stream items");
  });

  await t.step("assert structured output parsed", () => {
    assert(result, "expected final structured output result");
    assertEquals(result.output, fixtures.expected);
  });
}

export async function runBackAndForthCalculatorConversationTest(t: Deno.TestContext, options: { model: Model }) {
  const fixtures = createCalculatorFixtures();
  const agent = new Agent({
    model: options.model,
    instructions: `\
You are running a live multi-turn integration test.
Use ${fixtures.calculator.normalizedName} exactly once whenever a turn explicitly asks for arithmetic or validation.
When a turn explicitly says not to use a tool, answer from conversation memory only.
Keep answers terse and follow the exact requested format.`,
    tools: [fixtures.calculator],
  });

  const conversation: ChatItem[] = [];

  const assertTurnSucceeded = (result: AgentRunResult<unknown>, label: string) => {
    assert(result.history.length > 0, `expected ${label} to produce history`);
  };

  const turn1 = await t.step("turn 1 adds with calculator", async () => {
    const input: ChatItem = {
      type: "input_text",
      content: `Use ${fixtures.calculator.normalizedName} exactly once to add 7 and 5. Reply with only the number.`,
    };
    const result = await agent.run([
      ...conversation,
      input,
    ], {
      signal: AbortSignal.timeout(INTEGRATION_TIMEOUT_MS),
    });

    assertTurnSucceeded(result, "turn 1");
    conversation.push(input, { type: "output_text", content: result.outputText.trim() });
    assertEquals(fixtures.state.calls.length, 1);
    assertEquals(fixtures.state.calls[0]?.operation, "add");
    assertEquals(fixtures.state.calls[0]?.left, 7);
    assertEquals(fixtures.state.calls[0]?.right, 5);
  });

  if (!turn1) return;

  const turn2 = await t.step("turn 2 recalls prior result without tool use", async () => {
    const input: ChatItem = {
      type: "input_text",
      content: "Do not use any tool on this turn. What was the previous result? Reply with only the number.",
    };
    const result = await agent.run([
      ...conversation,
      input,
    ], {
      signal: AbortSignal.timeout(INTEGRATION_TIMEOUT_MS),
    });

    assertTurnSucceeded(result, "turn 2");
    conversation.push(input, { type: "output_text", content: result.outputText.trim() });
  });

  if (!turn2) return;

  const turn3 = await t.step("turn 3 validates prior result with calculator", async () => {
    const input: ChatItem = {
      type: "input_text",
      content:
        `Use ${fixtures.calculator.normalizedName} exactly once to subtract 5 from the previous result. If the calculator returns 7, reply with VALIDATED only.`,
    };
    const result = await agent.run([
      ...conversation,
      input,
    ], {
      signal: AbortSignal.timeout(INTEGRATION_TIMEOUT_MS),
    });

    assertTurnSucceeded(result, "turn 3");
    conversation.push(input, { type: "output_text", content: result.outputText.trim() });
  });

  if (!turn3) return;

  const turn4 = await t.step("turn 4 starts a new calculation", async () => {
    const input: ChatItem = {
      type: "input_text",
      content:
        `Use ${fixtures.calculator.normalizedName} exactly once to multiply 3 and 4. Reply with only the number.`,
    };
    const result = await agent.run([
      ...conversation,
      input,
    ], {
      signal: AbortSignal.timeout(INTEGRATION_TIMEOUT_MS),
    });

    assertTurnSucceeded(result, "turn 4");
    conversation.push(input, { type: "output_text", content: result.outputText.trim() });
  });

  if (!turn4) return;

  await t.step("turn 5 uses the latest result in a final calculation", async () => {
    const input: ChatItem = {
      type: "input_text",
      content:
        `Use ${fixtures.calculator.normalizedName} exactly once to divide the previous multiplication result by 2. Reply with only the number.`,
    };
    const result = await agent.run([
      ...conversation,
      input,
    ], {
      signal: AbortSignal.timeout(INTEGRATION_TIMEOUT_MS),
    });

    assertTurnSucceeded(result, "turn 5");
    conversation.push(input, { type: "output_text", content: result.outputText.trim() });
    assertEquals(conversation.filter((item) => item.type === "output_text").length, 5);
    assert(fixtures.state.calls.length >= 1, "expected calculator to be called at least once during the conversation");
  });
}
