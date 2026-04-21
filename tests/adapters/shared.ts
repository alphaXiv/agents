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

interface StructuredToolSimpleObjectInput {
  label: string;
  count: number;
}

interface StructuredToolMetadataValue {
  state: "ready";
  score: number;
}

interface StructuredToolNestedInput {
  header: {
    id: string;
    priority: number;
    tags: [string, string, string];
  };
  windows: [
    { day: "mon"; range: [number, number] },
    { day: "fri"; range: [number, number] },
  ];
  metadata: Record<string, StructuredToolMetadataValue>;
}

function createStructuredToolParameterFixtures() {
  const expected = {
    plainString: "lattice",
    constrainedString: "orbitals",
    decimalNumber: 12.5,
    constrainedInteger: 17,
    simpleObject: {
      label: "otter",
      count: 2,
    } satisfies StructuredToolSimpleObjectInput,
    nestedObject: {
      header: {
        id: "case_alpha",
        priority: 3,
        tags: ["red", "blue", "green"],
      },
      windows: [
        { day: "mon", range: [9, 11] },
        { day: "fri", range: [14, 16] },
      ],
      metadata: {
        alpha: { state: "ready", score: 0.5 },
        beta: { state: "ready", score: 0.5 },
      },
    } satisfies StructuredToolNestedInput,
  };

  const state = {
    plainStrings: [] as string[],
    constrainedStrings: [] as string[],
    decimalNumbers: [] as number[],
    constrainedIntegers: [] as number[],
    voidCalls: 0,
    simpleObjects: [] as StructuredToolSimpleObjectInput[],
    nestedObjects: [] as StructuredToolNestedInput[],
  };

  const plainStringTool = new Tool({
    name: "plain_string_input",
    description: `Call this tool exactly once with the bare string ${JSON.stringify(expected.plainString)}.`,
    parameters: z.string(),
    execute: (input) => {
      state.plainStrings.push(input);
      return "ok:plain-string";
    },
  });

  const constrainedStringTool = new Tool({
    name: "constrained_string_input",
    description: `Call this tool exactly once with the bare lowercase string ${
      JSON.stringify(expected.constrainedString)
    }.`,
    parameters: z.string().min(8).max(9).regex(/^[a-z]+$/),
    execute: (input) => {
      state.constrainedStrings.push(input);
      return "ok:constrained-string";
    },
  });

  const decimalNumberTool = new Tool({
    name: "decimal_number_input",
    description: `Call this tool exactly once with the bare number ${expected.decimalNumber}.`,
    parameters: z.number(),
    execute: (input) => {
      state.decimalNumbers.push(input);
      return "ok:decimal-number";
    },
  });

  const constrainedIntegerTool = new Tool({
    name: "constrained_integer_input",
    description: `Call this tool exactly once with the bare integer ${expected.constrainedInteger}.`,
    parameters: z.int().min(17).max(17),
    execute: (input) => {
      state.constrainedIntegers.push(input);
      return "ok:constrained-integer";
    },
  });

  const voidTool = new Tool({
    name: "void_input",
    description: "Call this tool exactly once with no parameters.",
    parameters: z.void(),
    execute: () => {
      state.voidCalls += 1;
      return "ok:void";
    },
  });

  const simpleObjectTool = new Tool({
    name: "simple_object_input",
    description: `Call this tool exactly once with ${JSON.stringify(expected.simpleObject)}.`,
    parameters: z.object({
      label: z.string().min(5).max(5),
      count: z.int().min(2).max(2),
    }).strict(),
    execute: (input) => {
      state.simpleObjects.push(input);
      return "ok:simple-object";
    },
  });

  const nestedObjectTool = new Tool({
    name: "nested_object_input",
    description: `Call this tool exactly once with ${JSON.stringify(expected.nestedObject)}.`,
    parameters: z.object({
      header: z.object({
        id: z.string().min(10).max(10).regex(/^case_[a-z]{5}$/),
        priority: z.int().min(3).max(3),
        tags: z.tuple([
          z.string().min(3).max(3),
          z.string().min(4).max(4),
          z.string().min(5).max(5),
        ]),
      }).strict(),
      windows: z.tuple([
        z.object({
          day: z.literal("mon"),
          range: z.tuple([z.int().min(9).max(9), z.int().min(11).max(11)]),
        }).strict(),
        z.object({
          day: z.literal("fri"),
          range: z.tuple([z.int().min(14).max(14), z.int().min(16).max(16)]),
        }).strict(),
      ]),
      metadata: z.record(
        z.string().regex(/^(alpha|beta)$/),
        z.object({
          state: z.literal("ready"),
          score: z.number().min(0.5).max(0.5),
        }).strict(),
      ),
    }).strict(),
    execute: (input) => {
      state.nestedObjects.push(input);
      return "ok:nested-object";
    },
  });

  const tools = {
    plainString: plainStringTool,
    constrainedString: constrainedStringTool,
    decimalNumber: decimalNumberTool,
    constrainedInteger: constrainedIntegerTool,
    void: voidTool,
    simpleObject: simpleObjectTool,
    nestedObject: nestedObjectTool,
  } as const;

  const marker = "LIVE_STRUCTURED_TOOL_PARAMETERS_COMPLETE";

  return {
    expected,
    state,
    tools,
    marker,
    instructions: [
      "You are running a live SDK integration test for structured tool parameters.",
      "Call every provided tool exactly once.",
      "Follow each tool description exactly and do not add extra fields.",
      `After all tool calls, respond with exactly ${marker} and nothing else.`,
    ].join(" "),
  };
}

function assertStructuredToolPayload(
  items: WithTraceId<StreamItem>[],
  toolName: string,
  expected: unknown,
) {
  const seenToolStarts = items
    .filter((item): item is Extract<WithTraceId<StreamItem>, { type: "tool_use_start" }> =>
      item.type === "tool_use_start"
    )
    .map((item) => item.kind);

  assert(
    items.some((item) => item.type === "tool_use_start" && item.kind === toolName),
    `expected ${toolName} tool_use_start, saw starts for: ${JSON.stringify(seenToolStarts)}`,
  );

  const toolUse = findToolUse(items, toolName);
  assert(toolUse, `expected ${toolName} tool_use item, saw starts for: ${JSON.stringify(seenToolStarts)}`);

  if (expected === undefined) {
    assertEquals(toolUse.content, undefined);
    return;
  }

  assert(toolUse.content, `expected ${toolName} content`);
  assertEquals(JSON.parse(toolUse.content), expected);
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
  assert(typeof metadata.inputTokens === "number", "expected inputTokens to exist");
  assert(typeof metadata.outputTokens === "number", "expected outputTokens to exist");
  assert(metadata.inputTokens > 0, "expected inputTokens to be > 0");
  assert(metadata.outputTokens > 0, "expected outputTokens to be > 0");
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
    `Call ${fixtures.pingTool.name} exactly once with no parameters.`,
    `Call ${fixtures.echoTool.name} exactly once with {\"query\":\"${fixtures.query}\"}.`,
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
      items.some((item) => item.type === "tool_use_start" && item.kind === fixtures.pingTool.name),
      `expected ${fixtures.pingTool.name} tool_use_start`,
    );

    const toolUse = findToolUse(items, fixtures.pingTool.name);
    assert(toolUse, `expected ${fixtures.pingTool.name} tool_use item`);
  });

  await t.step("assert streamed parameterized tool invocation", () => {
    assert(
      items.some((item) => item.type === "tool_use_start" && item.kind === fixtures.echoTool.name),
      `expected ${fixtures.echoTool.name} tool_use_start`,
    );

    const toolUse = findToolUse(items, fixtures.echoTool.name);
    assert(toolUse, `expected ${fixtures.echoTool.name} tool_use item`);
    assert(
      toolUse.content?.includes(fixtures.query),
      `expected ${fixtures.echoTool.name} content to include ${fixtures.query}`,
    );
  });

  await t.step("assert streamed void tool result", () => {
    const toolUse = findToolUse(items, fixtures.pingTool.name);
    assert(toolUse, `missing ${fixtures.pingTool.name} tool use for result assertion`);

    const toolResult = findToolResult(items, toolUse.tool_use_id);
    assert(toolResult, `expected ${fixtures.pingTool.name} tool_result_text`);
    assertEquals(toolResult.content, fixtures.pingResult);
  });

  await t.step("assert streamed parameterized tool result", () => {
    const toolUse = findToolUse(items, fixtures.echoTool.name);
    assert(toolUse, `missing ${fixtures.echoTool.name} tool use for result assertion`);

    const toolResult = findToolResult(items, toolUse.tool_use_id);
    assert(toolResult, `expected ${fixtures.echoTool.name} tool_result_text`);
    assertEquals(toolResult.content, fixtures.echoResult);
  });

  await t.step("assert final streamed output", () => {
    assert(result, "expected final agent result");
    assert(items.some((item) => item.type === "delta_output_text"), "expected streamed text output");
    assert(
      result.outputText.includes(fixtures.marker),
      `expected final output to include ${fixtures.marker} but instead outputted ${result.outputText}`,
    );
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
  },
) {
  const fixtures = createStructuredOutputFixtures();
  const agent = new Agent({
    model: options.model,
    instructions: fixtures.instructions,
    output: fixtures.output,
  });

  let result: AgentRunResult<unknown> | undefined;

  const collectedStructuredOutput = await t.step("collect structured output stream", async () => {
    const collected = await collectAgentStream(agent.stream(fixtures.prompt, {
      signal: AbortSignal.timeout(INTEGRATION_TIMEOUT_MS),
    }));
    result = collected.result;
  });

  if (!collectedStructuredOutput) return;

  await t.step("assert structured output parsed", () => {
    assert(result, "expected final structured output result");
    assertEquals(result.output, fixtures.expected);
  });
}

export async function runStructuredToolParameterStreamingTest(
  t: Deno.TestContext,
  options: {
    model: Model;
  },
) {
  const fixtures = createStructuredToolParameterFixtures();
  const agent = new Agent({
    model: options.model,
    instructions: fixtures.instructions,
    tools: [
      fixtures.tools.plainString,
      fixtures.tools.constrainedString,
      fixtures.tools.decimalNumber,
      fixtures.tools.constrainedInteger,
      fixtures.tools.void,
      fixtures.tools.simpleObject,
      fixtures.tools.nestedObject,
    ],
  });

  let items: WithTraceId<StreamItem>[] = [];
  let result: AgentRunResult<unknown> | undefined;

  const collectedAgentStream = await t.step("collect structured tool parameter stream", async () => {
    const collected = await collectAgentStream(agent.stream("Run the structured tool parameter test now.", {
      signal: AbortSignal.timeout(INTEGRATION_TIMEOUT_MS),
    }));
    items = collected.items;
    result = collected.result;
  });

  if (!collectedAgentStream) return;

  await t.step("plain string tool uses the expected payload and parsed input", () => {
    assertStructuredToolPayload(items, fixtures.tools.plainString.name, fixtures.expected.plainString);
    assertEquals(fixtures.state.plainStrings, [fixtures.expected.plainString]);
  });

  await t.step("constrained string tool uses the expected payload and parsed input", () => {
    assertStructuredToolPayload(items, fixtures.tools.constrainedString.name, fixtures.expected.constrainedString);
    assertEquals(fixtures.state.constrainedStrings, [fixtures.expected.constrainedString]);
  });

  await t.step("number tool uses the expected payload and parsed input", () => {
    assertStructuredToolPayload(items, fixtures.tools.decimalNumber.name, fixtures.expected.decimalNumber);
    assertEquals(fixtures.state.decimalNumbers, [fixtures.expected.decimalNumber]);
  });

  await t.step("integer tool uses the expected payload and parsed input", () => {
    assertStructuredToolPayload(items, fixtures.tools.constrainedInteger.name, fixtures.expected.constrainedInteger);
    assertEquals(fixtures.state.constrainedIntegers, [fixtures.expected.constrainedInteger]);
  });

  await t.step("void tool uses the expected payload and parsed input", () => {
    assertStructuredToolPayload(items, fixtures.tools.void.name, undefined);
    assertEquals(fixtures.state.voidCalls, 1);
  });

  await t.step("simple object tool uses the expected payload and parsed input", () => {
    assertStructuredToolPayload(items, fixtures.tools.simpleObject.name, fixtures.expected.simpleObject);
    assertEquals(fixtures.state.simpleObjects, [fixtures.expected.simpleObject]);
  });

  await t.step("nested object tool uses the expected payload and parsed input", () => {
    assertStructuredToolPayload(items, fixtures.tools.nestedObject.name, fixtures.expected.nestedObject);
    assertEquals(fixtures.state.nestedObjects, [fixtures.expected.nestedObject]);
  });

  await t.step("assert final streamed output for structured tool parameters", () => {
    assert(result, "expected final agent result");
    assert(
      result.outputText.includes(fixtures.marker),
      `expected final output to include ${fixtures.marker} but instead outputted ${result.outputText}`,
    );
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

export async function runToolLessHandoffResumeTest(t: Deno.TestContext, options: { model: Model }) {
  const loadProjectSnapshot = new Tool({
    name: "load_project_snapshot",
    description: "Load the current project snapshot. You must call this before answering.",
    parameters: z.void(),
    execute: () => {
      return JSON.stringify({
        repository: "alphaXiv/agents",
        activeModel: "gemini-3.1-flash-lite-preview",
        handoffCode: "FLASH-LITE-314",
      });
    },
  });

  const loadReleaseChecklist = new Tool({
    name: "load_release_checklist",
    description: "Load the release checklist. You must call this before answering.",
    parameters: z.void(),
    execute: () => {
      return JSON.stringify({
        nextTask: "verify history replay with a tool-less agent",
        owner: "SDK examples",
        status: "ready",
      });
    },
  });

  const firstAgent = new Agent({
    model: options.model,
    instructions: [
      "You are the first handoff agent.",
      "You must call load_project_snapshot exactly once.",
      "You must call load_release_checklist exactly once.",
      "Do not answer until both tool calls have completed.",
      "After both tools return, summarize the handoff in 3 short sentences.",
    ].join(" "),
    tools: [loadProjectSnapshot, loadReleaseChecklist],
  });

  const secondAgent = new Agent({
    model: options.model,
    instructions: [
      "You are the follow-up agent.",
      "You have no tools.",
      "Resume from the existing conversation history and answer naturally.",
      "Do not claim to have called any tools yourself.",
    ].join(" "),
  });

  const firstPrompt = [
    "Prepare a handoff for another agent.",
    "You must use your tools before answering.",
    "Include the repository, handoff code, next task, owner, and status.",
  ].join(" ");

  let firstRun: AgentRunResult<unknown> | undefined;
  let secondRun: AgentRunResult<unknown> | undefined;

  const completedFirstRun = await t.step("first agent creates a handoff with tools", async () => {
    firstRun = await firstAgent.run(firstPrompt, {
      signal: AbortSignal.timeout(INTEGRATION_TIMEOUT_MS),
    });
  });

  if (!completedFirstRun || !firstRun) return;

  const conversation: ChatItem[] = [
    { type: "input_text", content: firstPrompt },
    ...firstRun.history,
  ];

  const completedSecondRun = await t.step("second agent resumes the handoff without tools", async () => {
    secondRun = await secondAgent.run([
      ...conversation,
      {
        type: "input_text",
        content: "Continue from that handoff. What handoff code and next task were already established?",
      },
    ], {
      signal: AbortSignal.timeout(INTEGRATION_TIMEOUT_MS),
    });
  });

  if (!completedSecondRun || !secondRun) return;

  const completedFirstRunResult = firstRun;
  const completedSecondRunResult = secondRun;

  await t.step("handoff details survive tool-less replay", () => {
    const resumedOutput = completedSecondRunResult.outputText.toLowerCase();

    assert(
      completedFirstRunResult.outputText.includes("FLASH-LITE-314"),
      "expected first handoff to mention the handoff code",
    );
    assert(
      completedSecondRunResult.outputText.includes("FLASH-LITE-314"),
      "expected resumed handoff to preserve the handoff code",
    );
    assert(
      /history replay|replay/.test(resumedOutput),
      "expected resumed handoff to preserve the replay task",
    );
    assert(
      /tool-less|tool less|without tools/.test(resumedOutput),
      "expected resumed handoff to preserve the next task",
    );
  });
}
