import type Anthropic from "@anthropic-ai/sdk";
import { assert, assertEquals, assertRejects } from "@std/assert";
import z from "zod";
import { addStreamItem, Agent, type AnyTool, type ChatItem, type StreamItem, Tool } from "../../mod.ts";
import { anthropicModel } from "../../src/adapters/anthropic/adapter.ts";
import { applyAnthropicCacheBreakpoint, getAnthropicHistory } from "../../src/adapters/anthropic/history.ts";
import { getAnthropicMessagesStreamConfig } from "../../src/adapters/anthropic/models.ts";
import { createAnthropicCompatibleSchema, normalizeAnthropicTools } from "../../src/adapters/anthropic/utils.ts";
import {
  collectAdapterStream,
  createToolFixtures,
  INTEGRATION_TIMEOUT_MS,
  runAdapterToolStreamingTest,
  runAgentToolStreamingTest,
  runBackAndForthCalculatorConversationTest,
  runStructuredOutputStreamingTest,
  runStructuredToolParameterStreamingTest,
} from "./shared.ts";
import type { Adapter } from "../../src/adapters/adapter.ts";

const HAS_ANTHROPIC_KEY = Boolean(Deno.env.get("ANTHROPIC_API_KEY"));

Deno.test({
  name: "AnthropicAdapter streams a parameterized tool call (claude-opus-5, adaptive)",
  ignore: !HAS_ANTHROPIC_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    const fixtures = createToolFixtures();
    const adapter = anthropicModel({
      model: "claude-opus-5",
      effort: "high",
    });

    await runAdapterToolStreamingTest(t, {
      stream: adapter.stream({
        history: [{
          type: "input_text",
          content: `Call ${fixtures.echoTool.normalizedName} exactly once with {\"query\":\"${fixtures.query}\"}.`,
        }],
        instructions: "You are a compliant live integration test assistant.",
        tools: [fixtures.echoTool],
        signal: AbortSignal.timeout(INTEGRATION_TIMEOUT_MS),
      }),
      toolName: fixtures.echoTool.normalizedName,
      expectedContentSubstring: fixtures.query,
    });
  },
});

Deno.test({
  name: "AnthropicAdapter streams a parameterized tool call (claude-opus-4-8, adaptive)",
  ignore: !HAS_ANTHROPIC_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    const fixtures = createToolFixtures();
    const adapter = anthropicModel({
      model: "claude-opus-4-8",
      effort: "high",
    });

    await runAdapterToolStreamingTest(t, {
      stream: adapter.stream({
        history: [{
          type: "input_text",
          content: `Call ${fixtures.echoTool.normalizedName} exactly once with {\"query\":\"${fixtures.query}\"}.`,
        }],
        instructions: "You are a compliant live integration test assistant.",
        tools: [fixtures.echoTool],
        signal: AbortSignal.timeout(INTEGRATION_TIMEOUT_MS),
      }),
      toolName: fixtures.echoTool.normalizedName,
      expectedContentSubstring: fixtures.query,
    });
  },
});

Deno.test({
  name: "AnthropicAdapter streams a parameterized tool call (claude-opus-4-7, adaptive)",
  ignore: !HAS_ANTHROPIC_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    const fixtures = createToolFixtures();
    const adapter = anthropicModel({
      model: "claude-opus-4-7",
      effort: "high",
    });

    await runAdapterToolStreamingTest(t, {
      stream: adapter.stream({
        history: [{
          type: "input_text",
          content: `Call ${fixtures.echoTool.normalizedName} exactly once with {\"query\":\"${fixtures.query}\"}.`,
        }],
        instructions: "You are a compliant live integration test assistant.",
        tools: [fixtures.echoTool],
        signal: AbortSignal.timeout(INTEGRATION_TIMEOUT_MS),
      }),
      toolName: fixtures.echoTool.normalizedName,
      expectedContentSubstring: fixtures.query,
    });
  },
});

Deno.test({
  name: "AnthropicAdapter streams a parameterized tool call (claude-opus-4-6, adaptive)",
  ignore: !HAS_ANTHROPIC_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    const fixtures = createToolFixtures();
    const adapter = anthropicModel({
      model: "claude-opus-4-6",
      effort: "low",
    });

    await runAdapterToolStreamingTest(t, {
      stream: adapter.stream({
        history: [{
          type: "input_text",
          content: `Call ${fixtures.echoTool.normalizedName} exactly once with {\"query\":\"${fixtures.query}\"}.`,
        }],
        instructions: "You are a compliant live integration test assistant.",
        tools: [fixtures.echoTool],
        signal: AbortSignal.timeout(INTEGRATION_TIMEOUT_MS),
      }),
      toolName: fixtures.echoTool.normalizedName,
      expectedContentSubstring: fixtures.query,
    });
  },
});

Deno.test({
  name: "AnthropicModel streams tools and results (claude-opus-5, adaptive-only)",
  ignore: !HAS_ANTHROPIC_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runAgentToolStreamingTest(t, {
      model: anthropicModel({ model: "claude-opus-5", effort: "low" }),
    });
  },
});

Deno.test({
  name: "AnthropicModel streams tools and results (claude-opus-4-8, adaptive-only)",
  ignore: !HAS_ANTHROPIC_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runAgentToolStreamingTest(t, {
      model: anthropicModel({ model: "claude-opus-4-8", effort: "low" }),
    });
  },
});

Deno.test({
  name: "AnthropicModel streams tools and results (claude-opus-4-7, adaptive-only)",
  ignore: !HAS_ANTHROPIC_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runAgentToolStreamingTest(t, {
      model: anthropicModel({ model: "claude-opus-4-7", effort: "low" }),
    });
  },
});

Deno.test({
  name: "AnthropicModel streams tools and results (claude-opus-4-6, adaptive-only)",
  ignore: !HAS_ANTHROPIC_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runAgentToolStreamingTest(t, {
      model: anthropicModel({ model: "claude-opus-4-6", effort: "low" }),
    });
  },
});

Deno.test({
  name: "claude-opus-4-8 works with",
  fn() {
    assertEquals(
      getAnthropicMessagesStreamConfig({ model: "claude-opus-4-8", effort: "xhigh" }),
      {
        thinking: { type: "adaptive", display: "summarized" },
        output_config: { effort: "xhigh" },
        betas: undefined,
      },
    );
  },
});

Deno.test({
  name: "claude-opus-4-7 works with",
  fn() {
    assertEquals(
      getAnthropicMessagesStreamConfig({ model: "claude-opus-4-7", effort: "xhigh" }),
      {
        thinking: { type: "adaptive", display: "summarized" },
        output_config: { effort: "xhigh" },
        betas: undefined,
      },
    );
  },
});

Deno.test({
  name: "claude-sonnet-5 works with",
  fn() {
    assertEquals(
      getAnthropicMessagesStreamConfig({ model: "claude-sonnet-5", effort: "xhigh" }),
      {
        thinking: { type: "adaptive", display: "summarized" },
        output_config: { effort: "xhigh" },
        betas: undefined,
      },
    );
  },
});

Deno.test({
  name: "claude-opus-4-7 supports omitted thinking display",
  fn() {
    assertEquals(
      getAnthropicMessagesStreamConfig({
        model: "claude-opus-4-7",
        effort: "xhigh",
        thinkingDisplay: "omitted",
      }),
      {
        thinking: { type: "adaptive", display: "omitted" },
        output_config: { effort: "xhigh" },
        betas: undefined,
      },
    );
  },
});

Deno.test({
  name: "AnthropicModel streams tools and results (claude-sonnet-4-5, extended-only)",
  ignore: !HAS_ANTHROPIC_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runAgentToolStreamingTest(t, {
      model: anthropicModel({ model: "claude-sonnet-4-5", thinkingLevel: "low", interleaved: true }),
    });
  },
});

Deno.test({
  name: "AnthropicModel streams tools and results (claude-haiku-4-5, extended-only)",
  ignore: !HAS_ANTHROPIC_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runAgentToolStreamingTest(t, {
      model: anthropicModel({ model: "claude-haiku-4-5", thinkingLevel: "low", interleaved: true }),
    });
  },
});

Deno.test({
  name: "AnthropicModel streams tools and results (claude-opus-4-5, extended + effort)",
  ignore: !HAS_ANTHROPIC_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runAgentToolStreamingTest(t, {
      model: anthropicModel({ model: "claude-opus-4-5", thinkingLevel: "low", effort: "low", interleaved: true }),
    });
  },
});

Deno.test({
  name: "AnthropicModel streams tools and results (claude-sonnet-4-6, adaptive mode)",
  ignore: !HAS_ANTHROPIC_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runAgentToolStreamingTest(t, {
      model: anthropicModel({ model: "claude-sonnet-4-6", effort: "low" }),
    });
  },
});

Deno.test({
  name: "AnthropicModel streams tools and results (claude-sonnet-4-6, extended mode)",
  ignore: !HAS_ANTHROPIC_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runAgentToolStreamingTest(t, {
      model: anthropicModel({ model: "claude-sonnet-4-6", thinkingLevel: "low", interleaved: true }),
    });
  },
});

Deno.test({
  name: "AnthropicModel executes structured tool parameters (claude-sonnet-4-6, adaptive mode)",
  ignore: !HAS_ANTHROPIC_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runStructuredToolParameterStreamingTest(t, {
      model: anthropicModel({ model: "claude-sonnet-4-6", effort: "low" }),
    });
  },
});

Deno.test({
  name: "AnthropicModel streams native structured output (claude-sonnet-4-6)",
  ignore: !HAS_ANTHROPIC_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runStructuredOutputStreamingTest(t, {
      model: anthropicModel({ model: "claude-sonnet-4-6", effort: "low" }),
    });
  },
});

Deno.test({
  name: "AnthropicModel streams non-native structured output (claude-opus-4-1)",
  ignore: !HAS_ANTHROPIC_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runStructuredOutputStreamingTest(t, {
      model: anthropicModel({ model: "claude-opus-4-1", thinkingLevel: "low", interleaved: true }),
    });
  },
});

Deno.test({
  name: "AnthropicModel keeps a 5-turn calculator conversation (claude-haiku-4-5)",
  ignore: !HAS_ANTHROPIC_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runBackAndForthCalculatorConversationTest(t, {
      model: anthropicModel({ model: "claude-haiku-4-5", thinkingLevel: "minimal", interleaved: true }),
    });
  },
});

Deno.test("dynamic records and tuples round-trip through Anthropic compatibility", () => {
  const schema = z.object({
    tone: z.tuple([z.enum(["formal", "casual"]), z.enum(["warm", "cold"])]).describe("Tone tuple"),
    metadata: z.record(z.string(), z.object({ score: z.number().int().min(0) })),
  });

  const compatibility = createAnthropicCompatibleSchema(schema, {
    kind: "output",
    rootPath: "output",
  });

  const original = {
    tone: ["formal", "warm"],
    metadata: {
      alpha: { score: 1 },
      beta: { score: 2 },
    },
  };

  const anthropic = compatibility.toProvider(original);
  assertEquals(anthropic, {
    tone: { item0: "formal", item1: "warm" },
    metadata: [
      { key: "alpha", value: { score: 1 } },
      { key: "beta", value: { score: 2 } },
    ],
  });

  assertEquals(compatibility.fromProvider(anthropic), original);
  assert(compatibility.instructions.includes("Tuple"));
  assert(compatibility.instructions.includes("Record"));
});

Deno.test("finite-key records stay object-shaped", () => {
  const schema = z.record(z.enum(["id", "name"]), z.string().min(1));

  const compatibility = createAnthropicCompatibleSchema(schema, {
    kind: "output",
    rootPath: "output",
  });

  const anthropicJsonSchema = compatibility.jsonSchema;
  assertEquals(anthropicJsonSchema.type, "object");
  assertEquals(Array.isArray(anthropicJsonSchema.required), true);
  assertEquals(compatibility.toProvider({ id: "1", name: "Bingus" }), {
    id: "1",
    name: "Bingus",
  });
  assertEquals(compatibility.fromProvider({ id: "1", name: "Bingus" }), {
    id: "1",
    name: "Bingus",
  });
});

Deno.test("tool schemas can be wrapped to satisfy Anthropic top-level object requirement", () => {
  const schema = z.string().min(2).describe("Search query");

  const compatibility = createAnthropicCompatibleSchema(schema, {
    kind: "tool",
    requireTopLevelObject: true,
    rootPath: "input",
  });

  assertEquals(compatibility.toProvider("cats"), { content: "cats" });
  assertEquals(compatibility.fromProvider({ content: "cats" }), "cats");

  const anthropicJsonSchema = compatibility.jsonSchema;
  assertEquals(anthropicJsonSchema.type, "object");
  if (!("properties" in anthropicJsonSchema)) {
    throw new Error("expected an object JSON schema");
  }
  assertEquals(
    (anthropicJsonSchema.properties as Record<string, { description?: string }>).content?.description,
    "Search query",
  );
  assert(compatibility.instructions.includes("at least 2 characters"));
});

Deno.test("string formats are described in instructions instead of the schema", () => {
  const compatibility = createAnthropicCompatibleSchema(
    z.object({ url: z.url(), day: z.iso.date(), slug: z.string().regex(/^[a-z]+$/) }),
    {
      kind: "tool",
      requireTopLevelObject: true,
      rootPath: "input",
    },
  );

  assertEquals(compatibility.jsonSchema.properties, {
    url: { type: "string" },
    day: { type: "string" },
    slug: { type: "string" },
  });
  assert(compatibility.instructions.includes("`input.url` must be a valid uri"));
  assert(compatibility.instructions.includes("`input.day` must be a valid date"));
  assert(compatibility.instructions.includes("`input.slug` must match `^[a-z]+$`"));
});

Deno.test("tools are strict unless one opts out", () => {
  const options = {
    description: "A tool",
    parameters: z.object({ query: z.string() }),
    execute: () => "ok",
  };
  const plain = new Tool({ ...options, name: "plain" });
  const optOut = new Tool({ ...options, name: "opt_out", anthropicStrict: false });

  assertEquals(normalizeAnthropicTools([plain, optOut]).map(({ anthropic }) => anthropic.strict), [true, false]);
});

Deno.test("tool input is not streamed eagerly, so the API validates it", () => {
  const tool = new Tool({
    name: "search",
    description: "A tool",
    parameters: z.object({ query: z.string() }),
    execute: () => "ok",
  });

  const voidTool = new Tool({
    name: "ping",
    description: "A tool",
    parameters: z.void(),
    execute: () => "ok",
  });

  for (const normalized of normalizeAnthropicTools([tool, voidTool])) {
    assertEquals("eager_input_streaming" in normalized.anthropic, false);
  }
});

Deno.test("a malformed tool call still replays as a legal tool_use block", async () => {
  const tool = new Tool({
    name: "search",
    description: "A tool",
    parameters: z.object({ query: z.string() }),
    execute: () => "ok",
  });
  const malformed = '{"query": what is a cat}';

  const history = await getAnthropicHistory({
    history: [
      { type: "tool_use", tool_use_id: "call_1", kind: "search", content: malformed },
      { type: "tool_result_text", tool_use_id: "call_1", content: "Error: Invalid parameters for tool" },
    ],
    normalizedTools: normalizeAnthropicTools([tool]),
    signal: AbortSignal.abort(),
  });

  assertEquals(history[0], {
    role: "assistant",
    content: [{ type: "tool_use", id: "call_1", name: "search", input: { content: malformed } }],
  });
});

Deno.test("malformed tool arguments are handed on rather than ending the run", async () => {
  const tool = new Tool({
    name: "search",
    description: "A tool",
    parameters: z.object({ query: z.string() }),
    execute: () => "ok",
  });
  const malformed = '{"query": what is a cat}';
  const events = [
    { type: "content_block_start", index: 0, content_block: { type: "tool_use", id: "call_1", name: "search" } },
    { type: "content_block_delta", index: 0, delta: { type: "input_json_delta", partial_json: malformed } },
    { type: "content_block_stop", index: 0 },
  ];
  const adapter = anthropicModel({
    model: "claude-opus-5",
    apiKey: "unused",
    client: {
      beta: {
        messages: {
          stream: () =>
            Object.assign(
              async function* () {
                yield* events;
              }(),
              {
                finalMessage: () => Promise.resolve({ usage: { input_tokens: 1, output_tokens: 1 } }),
              },
            ),
        },
      },
      // deno-lint-ignore no-explicit-any
    } as any,
  });

  const { items } = await collectAdapterStream(adapter.stream({
    history: [{ type: "input_text", content: "hi" }],
    instructions: "",
    tools: [tool],
    signal: AbortSignal.timeout(1000),
  }));

  assertEquals(items.filter((item) => item.type === "tool_use"), [{
    type: "tool_use",
    index: 0,
    kind: tool.name,
    tool_use_id: "call_1",
    content: malformed,
  }]);
});

Deno.test("Anthropic retry feedback is replayed as a user message", async () => {
  const history = await getAnthropicHistory({
    history: [
      { type: "output_text", content: '{"broken": true}' },
      {
        type: "output_text",
        content:
          "Sorry, my output has an error:\nboom\nI will try again to produce a JSON response that conforms to the expected schema.",
      },
    ],
    normalizedTools: [],
    signal: AbortSignal.abort(),
  });

  assertEquals(history, [
    { role: "assistant", content: [{ type: "text", text: '{"broken": true}' }] },
    {
      role: "user",
      content: [{
        type: "text",
        text:
          "Sorry, my output has an error:\nboom\nI will try again to produce a JSON response that conforms to the expected schema.",
      }],
    },
  ]);
});

Deno.test("Anthropic tool history re-wraps normalized string tool inputs as objects", async () => {
  const searchTool = new Tool({
    name: "Searching the internet...",
    description: "Use when you want to search the internet",
    parameters: z.string(),
    execute: () => "unused",
  });

  const history = await getAnthropicHistory({
    history: [{
      type: "tool_use",
      tool_use_id: "call_1",
      kind: searchTool.name,
      content: '"cats"',
    }],
    normalizedTools: normalizeAnthropicTools([searchTool]),
    signal: AbortSignal.abort(),
  });

  assertEquals(history, [{
    role: "assistant",
    content: [{
      type: "tool_use",
      id: "call_1",
      name: searchTool.normalizedName,
      input: { content: "cats" },
    }],
  }]);
});

Deno.test("Anthropic wraps a scalar tool input whose tool is no longer registered", async () => {
  // A wrapper-object tool call authored by another provider stores its inner scalar.
  // Replayed once the tool is gone (nothing to re-wrap it), the raw scalar would make
  // Anthropic reject the request with "tool_use.input: Input should be an object".
  const history = await getAnthropicHistory({
    history: [{
      type: "tool_use",
      tool_use_id: "call_1",
      kind: "search_web",
      content: '"habitat challenge dataset"',
    }],
    normalizedTools: [],
    signal: AbortSignal.abort(),
  });

  assertEquals(history, [{
    role: "assistant",
    content: [{
      type: "tool_use",
      id: "call_1",
      name: "search_web",
      input: { content: "habitat challenge dataset" },
    }],
  }]);
});

Deno.test("Anthropic replays missing tool definitions with normalized names", async () => {
  const history = await getAnthropicHistory({
    history: [
      { type: "tool_use", tool_use_id: "call_1", kind: "Load Project Snapshot", content: undefined },
      { type: "tool_result_text", tool_use_id: "call_1", content: "ready" },
    ],
    normalizedTools: [],
    signal: AbortSignal.abort(),
  });

  assertEquals(history, [
    {
      role: "assistant",
      content: [{
        type: "tool_use",
        id: "call_1",
        name: "load_project_snapshot",
        input: {},
      }],
    },
    {
      role: "user",
      content: [{
        type: "tool_result",
        tool_use_id: "call_1",
        content: "ready",
        is_error: false,
      }],
    },
  ]);
});

Deno.test("Anthropic structured output streamed as text is restored before emission", async () => {
  const chunks = [
    '{"memorySnapshot":{"longTermAssociations":[{"key":"a","value":"1"}]},',
    '"responseStrategy":{"tone":{"item0":"casual","item1":"warm"}}}',
  ];

  const client = {
    beta: {
      messages: {
        stream() {
          return {
            async *[Symbol.asyncIterator]() {
              for (const chunk of chunks) {
                yield { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: chunk } };
              }
              yield { type: "content_block_stop", index: 0 };
            },
            finalMessage() {
              return { usage: { input_tokens: 0, output_tokens: 0 } };
            },
          };
        },
      },
    },
  } as unknown as Anthropic;

  const adapter = anthropicModel({
    model: "claude-sonnet-4-5",
    client,
  });

  const output = z.object({
    memorySnapshot: z.object({
      longTermAssociations: z.record(z.string(), z.string()),
    }),
    responseStrategy: z.object({
      tone: z.tuple([z.string(), z.string()]),
    }),
  });

  const stream = adapter.stream({
    history: [],
    instructions: "test",
    tools: [],
    signal: AbortSignal.abort(),
    output,
  });

  const items = [];
  const rebuiltHistory: ChatItem[] = [];
  while (true) {
    const next = await stream.next();
    if (next.done) break;
    items.push(next.value);
    addStreamItem(rebuiltHistory, next.value);
  }

  assertEquals(items, [
    {
      type: "delta_output_text",
      index: 0,
      delta: JSON.stringify({
        memorySnapshot: { longTermAssociations: { a: "1" } },
        responseStrategy: { tone: ["casual", "warm"] },
      }),
    },
  ]);
  assertEquals(rebuiltHistory, [{
    type: "output_text",
    content: JSON.stringify({
      memorySnapshot: { longTermAssociations: { a: "1" } },
      responseStrategy: { tone: ["casual", "warm"] },
    }),
  }]);
});

Deno.test("Anthropic ignores signature deltas when thinking display omits reasoning text", async () => {
  const client = {
    beta: {
      messages: {
        stream() {
          return {
            async *[Symbol.asyncIterator]() {
              yield { type: "content_block_delta", index: 0, delta: { type: "signature_delta", signature: "sig_1" } };
              yield { type: "content_block_delta", index: 1, delta: { type: "text_delta", text: "done" } };
              yield { type: "content_block_stop", index: 1 };
            },
            finalMessage() {
              return { usage: { input_tokens: 0, output_tokens: 0 } };
            },
          };
        },
      },
    },
  } as unknown as Anthropic;

  const adapter = anthropicModel({
    model: "claude-opus-4-7",
    client,
    effort: "high",
    thinkingDisplay: "omitted",
  });

  const { items, metadata } = await collectAdapterStream(adapter.stream({
    history: [],
    instructions: "test",
    tools: [],
    signal: AbortSignal.abort(),
  }));

  assertEquals(items, [{
    type: "delta_output_text",
    delta: "done",
    index: 1,
  }]);
  assertEquals(metadata, { inputTokens: 0, outputTokens: 0, cacheReadTokens: null, cacheWriteTokens: null });
});

Deno.test("Anthropic attaches signatures after reasoning block completion", async () => {
  const client = {
    beta: {
      messages: {
        stream() {
          return {
            async *[Symbol.asyncIterator]() {
              yield {
                type: "content_block_start",
                index: 0,
                content_block: { type: "thinking", thinking: "", signature: "" },
              };
              yield { type: "content_block_delta", index: 0, delta: { type: "signature_delta", signature: "sig_123" } };
              yield {
                type: "content_block_delta",
                index: 0,
                delta: { type: "thinking_delta", thinking: "Let me think." },
              };
              yield { type: "content_block_stop", index: 0 };
            },
            finalMessage() {
              return { usage: { input_tokens: 0, output_tokens: 0 } };
            },
          };
        },
      },
    },
  } as unknown as Anthropic;

  const adapter = anthropicModel({
    model: "claude-opus-4-7",
    client,
    effort: "high",
    thinkingDisplay: "summarized",
  });

  const stream = adapter.stream({
    history: [],
    instructions: "test",
    tools: [],
    signal: AbortSignal.abort(),
  });

  const items: StreamItem[] = [];
  const rebuiltHistory: ChatItem[] = [];
  while (true) {
    const next = await stream.next();
    if (next.done) break;
    items.push(next.value);
    addStreamItem(rebuiltHistory, next.value);
  }

  assertEquals(items, [{
    type: "delta_output_reasoning",
    index: 0,
    delta: "Let me think.",
  }]);
  assertEquals(rebuiltHistory, [{
    type: "output_reasoning",
    content: "Let me think.",
  }]);

  const history = await getAnthropicHistory({
    history: rebuiltHistory,
    normalizedTools: [],
    signal: AbortSignal.abort(),
  });
  assertEquals(history, [{
    role: "assistant",
    content: [{
      type: "thinking",
      thinking: "Let me think.",
      signature: "sig_123",
    }],
  }]);
});

async function checkReasoning(model: Adapter<unknown, unknown>, options: { shouldStreamReasoning: boolean }) {
  const agent = new Agent({
    model,
    instructions: "You are a puzzle solver.",
  });

  const collectedItems: StreamItem[] = [];
  for await (
    const item of agent.stream(`\
5 pirates of different ages have a treasure of 100 gold coins. On their ship, they decide to split the coins using this scheme:
The oldest pirate proposes how to share the coins, and ALL pirates (including the oldest) vote for or against it.
If 50% or more of the pirates vote for it, then the coins will be shared that way. Otherwise, the pirate proposing the scheme will be thrown overboard, and the process is repeated with the pirates that remain.
As pirates tend to be a bloodthirsty bunch, if a pirate would get the same number of coins if he voted for or against a proposal, he will vote against so that the pirate who proposed the plan will be thrown overboard.
Assuming that all 5 pirates are intelligent, rational, greedy, and do not wish to die, (and are rather good at math for pirates) what will happen?`)
  ) {
    collectedItems.push(item);
  }

  const reasoningItem = collectedItems.find((item) => item.type === "delta_output_reasoning");
  assertEquals(Boolean(reasoningItem), options.shouldStreamReasoning);
  assert(collectedItems.find((item) => item.type === "delta_output_text"));
}

Deno.test({
  name: "Anthropic properly streams reasoning for claude-sonnet-4-5",
  ignore: !HAS_ANTHROPIC_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
}, async () => {
  await checkReasoning(anthropicModel({ model: "claude-sonnet-4-5", thinkingLevel: "medium" }), {
    shouldStreamReasoning: true,
  });
});

Deno.test({
  name: "Anthropic properly streams reasoning for claude-sonnet-4-6",
  ignore: !HAS_ANTHROPIC_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
}, async () => {
  await checkReasoning(anthropicModel({ model: "claude-sonnet-4-6", effort: "medium" }), {
    shouldStreamReasoning: true,
  });
});

Deno.test({
  name: "Anthropic properly streams reasoning for claude-opus-4-5",
  ignore: !HAS_ANTHROPIC_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
}, async () => {
  await checkReasoning(anthropicModel({ model: "claude-opus-4-5", effort: "medium" }), {
    shouldStreamReasoning: true,
  });
});

Deno.test({
  name: "Anthropic properly streams reasoning for claude-opus-4-6",
  ignore: !HAS_ANTHROPIC_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
}, async () => {
  await checkReasoning(anthropicModel({ model: "claude-opus-4-6", effort: "medium" }), {
    shouldStreamReasoning: true,
  });
});
Deno.test({
  name: "Anthropic properly streams summarized reasoning for claude-opus-4-7",
  ignore: !HAS_ANTHROPIC_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
}, async () => {
  await checkReasoning(anthropicModel({ model: "claude-opus-4-7", effort: "max" }), {
    shouldStreamReasoning: true,
  });
});

Deno.test({
  name: "Anthropic supports omitted reasoning display for claude-opus-4-7",
  ignore: !HAS_ANTHROPIC_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
}, async () => {
  await checkReasoning(anthropicModel({ model: "claude-opus-4-7", effort: "max", thinkingDisplay: "omitted" }), {
    shouldStreamReasoning: false,
  });
});

/**
 * Captures the request body the adapter builds, and replays a fixed usage block,
 * so caching can be asserted without spending a live call.
 */
function createCapturingAnthropicClient(usage: Partial<Anthropic.Beta.BetaUsage> = {}) {
  const requests: Anthropic.Beta.Messages.MessageCreateParams[] = [];
  const client = {
    beta: {
      messages: {
        stream(params: Anthropic.Beta.Messages.MessageCreateParams) {
          requests.push(params);
          return Object.assign(
            (async function* () {})(),
            {
              finalMessage: () => ({
                usage: {
                  input_tokens: 10,
                  output_tokens: 20,
                  cache_read_input_tokens: null,
                  cache_creation_input_tokens: null,
                  ...usage,
                },
              }),
            },
          );
        },
      },
    },
  } as unknown as Anthropic;
  return { client, requests };
}

/** The breakpoint the adapter placed on the tail of the conversation, if any. */
function tailCacheControl(request: Anthropic.Beta.Messages.MessageCreateParams) {
  const content = request.messages.at(-1)?.content;
  const block = typeof content === "string" ? undefined : content?.at(-1);
  if (!block || block.type === "thinking" || block.type === "redacted_thinking") return undefined;
  return block.cache_control;
}

function streamOnce(
  adapter: Adapter<unknown, unknown>,
  { instructions = "You are a test assistant.", tools = [], cache }: {
    instructions?: string;
    tools?: AnyTool[];
    cache?: boolean;
  } = {},
) {
  return collectAdapterStream(adapter.stream({
    history: [
      { type: "input_text", content: "hi" },
      { type: "tool_use", kind: "echo", tool_use_id: "toolu_1", content: '{"query":"x"}' },
      { type: "tool_result_text", tool_use_id: "toolu_1", content: "x" },
    ],
    instructions,
    tools,
    cache,
    signal: AbortSignal.timeout(INTEGRATION_TIMEOUT_MS),
  }));
}

Deno.test("Anthropic caching breakpoints cover the system prefix and the conversation tail", async () => {
  const { client, requests } = createCapturingAnthropicClient();
  await streamOnce(anthropicModel({ model: "claude-opus-4-8", cache: { ttl: "1h" }, client }));

  const [request] = requests;
  assertEquals(request.system, [{
    type: "text",
    text: "You are a test assistant.",
    cache_control: { type: "ephemeral", ttl: "1h" },
  }]);

  // Tail breakpoint rides the last block so the next turn reads this turn's prefix.
  assertEquals(tailCacheControl(request), { type: "ephemeral", ttl: "1h" });
});

Deno.test("Anthropic caching defaults to the 5 minute cache", async () => {
  const { client, requests } = createCapturingAnthropicClient();
  await streamOnce(anthropicModel({ model: "claude-opus-4-8", cache: true, client }));

  assertEquals(requests[0].system, [{
    type: "text",
    text: "You are a test assistant.",
    cache_control: { type: "ephemeral", ttl: undefined },
  }]);
});

Deno.test("Anthropic caching is on by default for a tool-using agent and off without tools", async () => {
  const { echoTool } = createToolFixtures();

  const withTools = createCapturingAnthropicClient();
  await streamOnce(anthropicModel({ model: "claude-opus-4-8", client: withTools.client }), { tools: [echoTool] });
  assertEquals(tailCacheControl(withTools.requests[0]), { type: "ephemeral", ttl: undefined });

  const withoutTools = createCapturingAnthropicClient();
  await streamOnce(anthropicModel({ model: "claude-opus-4-8", client: withoutTools.client }));
  const [request] = withoutTools.requests;
  assertEquals(request.system, "You are a test assistant.");
  assertEquals(tailCacheControl(request), undefined);
});

Deno.test("Anthropic caching honours the caller's default over the tools heuristic", async () => {
  const { echoTool } = createToolFixtures();

  // Agent-level `cache: false` is the opt out of what tools would otherwise switch on.
  const off = createCapturingAnthropicClient();
  await streamOnce(anthropicModel({ model: "claude-opus-4-8", client: off.client }), {
    tools: [echoTool],
    cache: false,
  });
  assertEquals(tailCacheControl(off.requests[0]), undefined);

  // And switches caching on for a toolless agent that would otherwise skip it.
  const on = createCapturingAnthropicClient();
  await streamOnce(anthropicModel({ model: "claude-opus-4-8", client: on.client }), { cache: true });
  assertEquals(tailCacheControl(on.requests[0]), { type: "ephemeral", ttl: undefined });
});

Deno.test("Anthropic caching keeps a cache configured on the model over the caller's default", async () => {
  // Someone who configured a ttl asked for caching explicitly; a blanket default must not undo it.
  const { client, requests } = createCapturingAnthropicClient();
  await streamOnce(anthropicModel({ model: "claude-opus-4-8", cache: { ttl: "1h" }, client }), { cache: false });

  assertEquals(tailCacheControl(requests[0]), { type: "ephemeral", ttl: "1h" });
});

Deno.test("Anthropic caching stays off when explicitly disabled on a tool-using agent", async () => {
  const { echoTool } = createToolFixtures();
  const { client, requests } = createCapturingAnthropicClient();
  await streamOnce(anthropicModel({ model: "claude-opus-4-8", cache: false, client }), { tools: [echoTool] });

  const [request] = requests;
  assertEquals(request.system, "You are a test assistant.");
  assertEquals(tailCacheControl(request), undefined);
});

Deno.test("Anthropic sends blank instructions as a bare string, which caching cannot turn into an empty block", async () => {
  const { client, requests } = createCapturingAnthropicClient();
  await streamOnce(anthropicModel({ model: "claude-opus-4-8", cache: true, client }), { instructions: "" });

  assertEquals(requests[0].system, "");
});

Deno.test("Anthropic reports cache token usage separately from uncached input", async () => {
  const { client } = createCapturingAnthropicClient({
    cache_read_input_tokens: 4096,
    cache_creation_input_tokens: 512,
  });
  const { metadata } = await streamOnce(anthropicModel({ model: "claude-opus-4-8", cache: true, client }));

  assertEquals(metadata, {
    inputTokens: 10,
    outputTokens: 20,
    cacheReadTokens: 4096,
    cacheWriteTokens: 512,
  });
});

Deno.test("Anthropic caching skips a thinking block, which cannot carry a breakpoint", () => {
  const history: Anthropic.Messages.MessageParam[] = [{
    role: "assistant",
    content: [{ type: "thinking", thinking: "hmm", signature: "sig" }],
  }];
  applyAnthropicCacheBreakpoint(history, { type: "ephemeral" });

  assertEquals(history, [{
    role: "assistant",
    content: [{ type: "thinking", thinking: "hmm", signature: "sig" }],
  }]);
});

Deno.test({
  name: "AnthropicModel reads its own cached prefix on a second call (claude-haiku-4-5)",
  ignore: !HAS_ANTHROPIC_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn() {
    // Anthropic silently skips caching below the model minimum (4096 tokens here),
    // so the instructions have to clear it for the assertion to mean anything.
    const instructions = `You are a test assistant.\n${
      "Answer questions truthfully and cite the relevant policy section.\n".repeat(1200)
    }`;
    const adapter = anthropicModel({ model: "claude-haiku-4-5", cache: true });
    const call = () =>
      collectAdapterStream(adapter.stream({
        history: [{ type: "input_text", content: "Reply with the single word: ok" }],
        instructions,
        tools: [],
        signal: AbortSignal.timeout(INTEGRATION_TIMEOUT_MS),
      }));

    const first = await call();
    assert(
      (first.metadata.cacheWriteTokens ?? 0) > 0,
      `Expected the first call to write a cache entry, got ${JSON.stringify(first.metadata)}`,
    );

    const second = await call();
    assert(
      (second.metadata.cacheReadTokens ?? 0) > 0,
      `Expected the second call to read the cached prefix, got ${JSON.stringify(second.metadata)}`,
    );
    // Cached tokens are billed separately, so they must not double count as input.
    assert(
      (second.metadata.inputTokens ?? 0) < (second.metadata.cacheReadTokens ?? 0),
      `Expected cached tokens to be excluded from inputTokens, got ${JSON.stringify(second.metadata)}`,
    );
  },
});

Deno.test("Agent cache option reaches the Anthropic adapter", async () => {
  const { echoTool } = createToolFixtures();

  async function runAgent(cache: boolean | undefined) {
    const { client, requests } = createCapturingAnthropicClient();
    const agent = new Agent({
      model: anthropicModel({ model: "claude-opus-4-8", client }),
      instructions: "You are a test assistant.",
      tools: [echoTool],
      cache,
    });
    await agent.run("hi");
    return requests[0];
  }

  assertEquals(tailCacheControl(await runAgent(undefined)), { type: "ephemeral", ttl: undefined });
  assertEquals(tailCacheControl(await runAgent(false)), undefined);
});

Deno.test("textlike application/* files are inlined instead of rejected", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (() => Promise.resolve(new Response('{"cats": 2}'))) as typeof fetch;

  try {
    const history = await getAnthropicHistory({
      history: [{ type: "input_file", kind: "application/json", content: "https://example.com/cats.json" }],
      normalizedTools: [],
      signal: AbortSignal.abort(),
    });

    assertEquals(history, [{
      role: "user",
      content: [{ type: "text", text: '<ant-file>{"cats": 2}</ant-file>' }],
    }]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

Deno.test("non-textlike media types are still rejected", async () => {
  await assertRejects(
    () =>
      getAnthropicHistory({
        history: [{ type: "input_file", kind: "video/mp2t", content: "https://example.com/cats.ts" }],
        normalizedTools: [],
        signal: AbortSignal.abort(),
      }),
    Error,
    "video/mp2t",
  );
});
