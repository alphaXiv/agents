import Anthropic from "@anthropic-ai/sdk";
import { assert, assertEquals } from "@std/assert";
import z from "zod";
import { addStreamItem, Agent, type ChatItem, type StreamItem, Tool } from "../../mod.ts";
import { anthropicModel } from "../../src/adapters/anthropic/adapter.ts";
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
import { Adapter } from "../../src/adapters/adapter.ts";

const HAS_ANTHROPIC_KEY = Boolean(Deno.env.get("ANTHROPIC_API_KEY"));

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

Deno.test("Anthropic retry feedback is replayed as a user message", async () => {
  const adapter = new AnthropicAdapter({
    model: "claude-sonnet-4-5",
    client: {} as Anthropic,
    streamConfig: {},
  });

  const history = await adapter.getHistory(
    [
      { type: "output_text", content: '{"broken": true}' },
      {
        type: "output_text",
        content:
          "Sorry, my output has an error:\nboom\nI will try again to produce a JSON response that conforms to the expected schema.",
      },
    ],
    [],
    AbortSignal.abort(),
  );

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
  const adapter = new AnthropicAdapter({
    model: "claude-sonnet-4-5",
    client: {} as Anthropic,
    streamConfig: {},
  });
  const searchTool = new Tool({
    name: "Searching the internet...",
    description: "Use when you want to search the internet",
    parameters: z.string(),
    execute: () => "unused",
  });

  const history = await adapter.getHistory(
    [{
      type: "tool_use",
      tool_use_id: "call_1",
      kind: searchTool.name,
      content: '"cats"',
    }],
    normalizeAnthropicTools([searchTool]),
    AbortSignal.abort(),
  );

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

Deno.test("Anthropic replays missing tool definitions with normalized names", async () => {
  const adapter = new AnthropicAdapter({
    model: "claude-sonnet-4-5",
    client: {} as Anthropic,
    streamConfig: {},
  });

  const history = await adapter.getHistory(
    [
      { type: "tool_use", tool_use_id: "call_1", kind: "Load Project Snapshot", content: undefined },
      { type: "tool_result_text", tool_use_id: "call_1", content: "ready" },
    ],
    [],
    AbortSignal.abort(),
  );

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

  const adapter = new AnthropicAdapter({
    model: "claude-sonnet-4-5",
    client,
    streamConfig: {},
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

  const adapter = new AnthropicAdapter({
    model: "claude-opus-4-7",
    client,
    streamConfig: getAnthropicMessagesStreamConfig({
      model: "claude-opus-4-7",
      effort: "high",
      thinkingDisplay: "omitted",
    }),
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
  assertEquals(metadata, { inputTokens: 0, outputTokens: 0 });
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

  const adapter = new AnthropicAdapter({
    model: "claude-opus-4-7",
    client,
    streamConfig: getAnthropicMessagesStreamConfig({
      model: "claude-opus-4-7",
      effort: "high",
      thinkingDisplay: "summarized",
    }),
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

  const history = await adapter.getHistory(rebuiltHistory, [], AbortSignal.abort());
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
