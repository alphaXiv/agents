import { assert, assertEquals, assertThrows } from "@std/assert";
import OpenAI from "openai";
import type { ChatCompletionMessageParam } from "openai/resources/chat/completions";
import z from "zod";
import { openAICompletionsModel } from "../../src/adapters/openai_completions/adapter.ts";
import { getOpenAICompletionsHistory } from "../../src/adapters/openai_completions/history.ts";
import { normalizeOpenAICompletionsTools } from "../../src/adapters/openai_completions/tools.ts";
import { RETRY_RESUMABILITY_PROMPT } from "../../src/constants.ts";
import { Tool } from "../../src/tool.ts";

/**
 * Replays raw stream events without going through the SDK. Fine for asserting how the
 * adapter maps deltas; not usable for usage assertions, which depend on which fields the
 * SDK itself preserves — use `createFakeTransportClient` for those.
 */
function createMockClient(
  events: unknown[],
  usage: { prompt_tokens: number; completion_tokens: number } | undefined,
  captureRequest?: (request: unknown) => void,
) {
  return {
    chat: {
      completions: {
        stream(request: unknown) {
          captureRequest?.(request);

          return {
            async *[Symbol.asyncIterator]() {
              for (const event of events) {
                yield event;
              }
            },
            // deno-lint-ignore require-await
            async finalChatCompletion() {
              return { usage };
            },
          };
        },
      },
    },
  } as unknown as Pick<OpenAI, "chat">;
}

Deno.test("OpenAI Completions history normalizes assistant, retry, tools, and files", async () => {
  const searchTool = new Tool({
    name: "Search",
    description: "Search for documents",
    parameters: z.string(),
    execute: () => "unused",
  });

  const originalFetch = globalThis.fetch;
  globalThis.fetch = ((input: string | URL | Request, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
    if (init?.method === "HEAD") {
      return Promise.resolve(new Response(null, { headers: { "Content-Length": "6" } }));
    }

    return Promise.resolve(new Response(url.includes("cats.csv") ? "name,count\ncat,2" : "unused"));
  }) as typeof fetch;

  try {
    const history = await getOpenAICompletionsHistory({
      model: "test-model",
      pdfSupport: { mode: "native" },
      history: [
        { type: "input_text", content: "hello" },
        { type: "output_text", content: "hi there" },
        { type: "output_reasoning", content: "I should search first." },
        { type: "tool_use", tool_use_id: "call_1", kind: searchTool.name, content: '"cats"' },
        { type: "tool_result_text", tool_use_id: "call_1", content: "found 2 results" },
        { type: "tool_result_file", tool_use_id: "call_1", kind: "text/csv", content: "https://example.com/cats.csv" },
        { type: "input_file", kind: "application/pdf", content: "https://example.com/cats.pdf" },
      ],
      instructions: "Be useful",
      normalizedTools: normalizeOpenAICompletionsTools([searchTool]),
      signal: AbortSignal.abort(),
    });

    assertEquals(history, [
      { role: "system", content: "Be useful" },
      { role: "user", content: "hello" },
      { role: "assistant", content: "hi there" },
      {
        role: "assistant",
        content: null,
        reasoning_content: "I should search first.",
        tool_calls: [{
          id: "call_1",
          type: "function",
          function: {
            name: "search",
            arguments: '{"content":"cats"}',
          },
        }],
      } as ChatCompletionMessageParam,
      { role: "tool", tool_call_id: "call_1", content: "found 2 results" },
      {
        role: "user",
        content: [{
          type: "text",
          text: '<file mime-type="text/csv">name,count\ncat,2</file>',
        }],
      },
      {
        role: "user",
        content: [{
          type: "file",
          file: {
            file_data: "data:application/pdf;base64,dW51c2Vk",
            filename: "cats.pdf",
          },
        }],
      },
    ]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

Deno.test("OpenAI Completions retry feedback is replayed as user content and resumability uses system role", async () => {
  const retryHistory = await getOpenAICompletionsHistory({
    model: "test-model",
    history: [
      { type: "output_text", content: '{"broken": true}' },
      {
        type: "output_text",
        content:
          "Sorry, my output has an error:\nboom\nI will try again to produce a JSON response that conforms to the expected schema.",
      },
    ],
    instructions: "Be useful",
    normalizedTools: [],
    signal: AbortSignal.abort(),
  });

  assertEquals(retryHistory, [
    { role: "system", content: "Be useful" },
    { role: "assistant", content: '{"broken": true}' },
    {
      role: "user",
      content:
        "Sorry, my output has an error:\nboom\nI will try again to produce a JSON response that conforms to the expected schema.",
    },
  ]);

  const resumableHistory = await getOpenAICompletionsHistory({
    model: "test-model",
    history: [{ type: "output_text", content: "partial response" }],
    instructions: "Be useful",
    normalizedTools: [],
    signal: AbortSignal.abort(),
  });

  assertEquals(resumableHistory, [
    { role: "system", content: "Be useful" },
    { role: "assistant", content: "partial response" },
    { role: "system", content: RETRY_RESUMABILITY_PROMPT },
  ]);
});

Deno.test("OpenAI Completions replays missing tool definitions with normalized names", async () => {
  const history = await getOpenAICompletionsHistory({
    model: "test-model",
    history: [
      { type: "tool_use", tool_use_id: "call_1", kind: "Load Project Snapshot", content: undefined },
      { type: "tool_result_text", tool_use_id: "call_1", content: "ready" },
    ],
    instructions: "Be useful",
    normalizedTools: [],
    signal: AbortSignal.abort(),
  });

  assertEquals(history, [
    { role: "system", content: "Be useful" },
    {
      role: "assistant",
      content: null,
      reasoning_content: "",
      tool_calls: [{
        id: "call_1",
        type: "function",
        function: {
          name: "load_project_snapshot",
          arguments: "{}",
        },
      }],
    } as ChatCompletionMessageParam,
    { role: "tool", tool_call_id: "call_1", content: "ready" },
  ]);
});

Deno.test("OpenAI Completions uses reversible OpenAI compatibility for tuple and record tool schemas", () => {
  const tupleTool = new Tool({
    name: "Tuple Tool",
    description: "Uses nested tuple and record input",
    parameters: z.object({
      header: z.object({
        tags: z.tuple([z.string(), z.number()]),
      }).strict(),
      metadata: z.record(z.string(), z.object({ score: z.number() }).strict()),
    }).strict(),
    execute: () => "unused",
  });

  const [normalizedTool] = normalizeOpenAICompletionsTools([tupleTool]);
  const parameters = normalizedTool?.openAI.function.parameters as {
    properties?: Record<string, unknown>;
  };
  const header = parameters.properties?.header as { properties?: Record<string, unknown> } | undefined;
  const metadata = parameters.properties?.metadata as { items?: Record<string, unknown> } | undefined;
  const tags = header?.properties?.tags as {
    type?: unknown;
    properties?: Record<string, unknown>;
    required?: unknown;
    prefixItems?: unknown;
  } | undefined;

  assertEquals(
    normalizedTool?.compatibility?.toProvider({
      header: { tags: ["cats", 2] },
      metadata: { alpha: { score: 0.5 } },
    }),
    {
      header: { tags: { item0: "cats", item1: 2 } },
      metadata: [{ key: "alpha", value: { score: 0.5 } }],
    },
  );
  assertEquals(
    normalizedTool?.compatibility?.fromProvider({
      header: { tags: { item0: "cats", item1: 2 } },
      metadata: [{ key: "alpha", value: { score: 0.5 } }],
    }),
    {
      header: { tags: ["cats", 2] },
      metadata: { alpha: { score: 0.5 } },
    },
  );

  assertEquals(tags?.prefixItems, undefined);
  assertEquals(tags?.type, "object");
  assertEquals(tags?.properties, {
    item0: { type: "string" },
    item1: { type: "number" },
  });
  assertEquals(tags?.required, ["item0", "item1"]);
  assertEquals(metadata?.items, {
    type: "object",
    properties: {
      key: { type: "string" },
      value: {
        type: "object",
        properties: { score: { type: "number" } },
        required: ["score"],
        additionalProperties: false,
      },
    },
    required: ["key", "value"],
    additionalProperties: false,
  });
  assert(normalizedTool?.openAI.function.description?.includes("<input_requirements>"));
});

Deno.test("OpenAI Completions restores structured output from OpenAI-compatible surrogate shapes", async () => {
  let capturedRequest: unknown;

  const adapter = openAICompletionsModel({
    model: "test-model",
    provider: "Test Provider",
    client: createMockClient(
      [
        {
          choices: [{
            delta: {
              content: '{"tags":{"item0":"red","item1":2},"metadata":[{"key":"alpha","value":"1"}]}',
            },
          }],
        },
      ],
      { prompt_tokens: 3, completion_tokens: 2 },
      (request) => {
        capturedRequest = request;
      },
    ),
  });

  const stream = adapter.stream({
    history: [],
    instructions: "Be useful",
    tools: [],
    signal: AbortSignal.abort(),
    output: z.object({
      tags: z.tuple([z.string(), z.number()]),
      metadata: z.record(z.string(), z.string()),
    }),
  });

  const items = [];
  while (true) {
    const next = await stream.next();
    if (next.done) break;
    items.push(next.value);
  }

  assertEquals(items, [{
    type: "delta_output_text",
    index: 0,
    delta: '{"tags":["red",2],"metadata":{"alpha":"1"}}',
  }]);

  assert(
    ((capturedRequest as { messages: Array<{ content: string }> }).messages[0]?.content).includes(
      "<output_requirements>",
    ),
  );
});

Deno.test("OpenAI Completions stream maps text, reasoning, and tool calls", async () => {
  let capturedRequest: unknown;

  const searchTool = new Tool({
    name: "Search",
    description: "Search for documents",
    parameters: z.string(),
    execute: () => "unused",
  });

  const adapter = openAICompletionsModel({
    model: "test-model",
    provider: "Test Provider",
    client: createMockClient([
      {
        choices: [{
          delta: {
            content: "Hello",
          },
        }],
      },
      {
        choices: [{
          delta: {
            reasoning: "Need tool",
          },
        }],
      },
      {
        choices: [{
          delta: {
            tool_calls: [{
              index: 0,
              id: "call_1",
              function: {
                name: "search",
                arguments: '{"content":',
              },
            }],
          },
        }],
      },
      {
        choices: [{
          delta: {
            tool_calls: [{
              index: 0,
              function: {
                arguments: '"cats"}',
              },
            }],
          },
        }],
      },
      {
        choices: [{
          delta: {
            content: "Done",
          },
        }],
      },
    ], {
      prompt_tokens: 11,
      completion_tokens: 7,
    }, (request) => {
      capturedRequest = request;
    }),
    parallelToolCalls: false,
    extraRequestBody: () => ({
      reasoning: { enabled: true },
    }),
  });

  const stream = adapter.stream({
    history: [],
    instructions: "Be useful",
    tools: [searchTool],
    signal: AbortSignal.abort(),
    output: z.object({ answer: z.string() }),
  });

  const items = [];
  while (true) {
    const next = await stream.next();
    if (next.done) {
      assertEquals(next.value, { inputTokens: 11, outputTokens: 7, cacheReadTokens: null, cacheWriteTokens: 0 });
      break;
    }
    items.push(next.value);
  }

  assertEquals(items, [
    { type: "delta_output_text", delta: "Hello", index: 0 },
    { type: "delta_output_reasoning", delta: "Need tool", index: 1 },
    { type: "tool_use_start", tool_use_id: "call_1", kind: "Search", index: 2 },
    { type: "delta_output_text", delta: "Done", index: 3 },
    { type: "tool_use", tool_use_id: "call_1", kind: "Search", content: '"cats"', index: 2 },
  ]);

  assertEquals(capturedRequest, {
    model: "test-model",
    messages: [{ role: "system", content: "Be useful" }],
    parallel_tool_calls: false,
    tools: [{
      type: "function",
      function: {
        name: "search",
        description: "Search for documents",
        strict: true,
        parameters: {
          type: "object",
          properties: {
            content: {
              $schema: "https://json-schema.org/draft/2020-12/schema",
              type: "string",
            },
          },
          required: ["content"],
          additionalProperties: false,
        },
      },
    }],
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "output",
        strict: true,
        schema: {
          type: "object",
          properties: {
            answer: { type: "string" },
          },
          required: ["answer"],
          additionalProperties: false,
        },
      },
    },
    stream: true,
    reasoning: { enabled: true },
    stream_options: { include_usage: true },
  });
});

Deno.test("OpenAI Completions validates pdf support against supported mime types", () => {
  assertThrows(
    () =>
      openAICompletionsModel({
        model: "test-model",
        provider: "Test Provider",
        client: createMockClient([], undefined),
        supportedMimeTypes: ["text/plain"],
        pdfSupport: { mode: "native" },
      }),
    Error,
    "pdfSupport requires application/pdf to be included in supportedMimeTypes",
  );
});

/**
 * Drives the real OpenAI client over a fake transport. A hand-rolled `totalUsage()` /
 * `finalChatCompletion()` cannot show which usage fields the SDK actually preserves —
 * `totalUsage()` silently drops `prompt_tokens_details` — so usage assertions have to
 * replay chunks through the SDK rather than mock its accessors.
 */
function createFakeTransportClient(chunks: unknown[]) {
  const sse = chunks.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`).join("") + "data: [DONE]\n\n";
  return new OpenAI({
    apiKey: "test",
    fetch: () => Promise.resolve(new Response(sse, { status: 200, headers: { "content-type": "text/event-stream" } })),
  });
}

function usageChunks(usage: Record<string, unknown>) {
  const base = { id: "c", object: "chat.completion.chunk", created: 1, model: "gpt-4.1-mini" };
  return [
    { ...base, choices: [{ index: 0, delta: { role: "assistant", content: "hi" }, finish_reason: null }] },
    { ...base, choices: [{ index: 0, delta: {}, finish_reason: "stop" }] },
    { ...base, choices: [], usage },
  ];
}

Deno.test("OpenAI Completions reports cached tokens as their own bucket, outside inputTokens", async () => {
  const client = createFakeTransportClient(usageChunks({
    prompt_tokens: 1500,
    completion_tokens: 7,
    total_tokens: 1507,
    prompt_tokens_details: { cached_tokens: 1024 },
  }));
  const adapter = openAICompletionsModel({ model: "gpt-4.1-mini", client });

  const stream = adapter.stream({
    history: [],
    instructions: "Be useful",
    tools: [],
    signal: AbortSignal.timeout(5000),
  });
  while (true) {
    const next = await stream.next();
    if (!next.done) continue;
    // OpenAI counts cached tokens inside prompt_tokens, so 1500 - 1024 is the uncached remainder.
    assertEquals(next.value, { inputTokens: 476, outputTokens: 7, cacheReadTokens: 1024, cacheWriteTokens: 0 });
    break;
  }
});

Deno.test("OpenAI Completions splits the cache write premium out of inputTokens (GPT-5.6+)", async () => {
  // From GPT-5.6 OpenAI bills cache writes at 1.25x and counts them inside prompt_tokens,
  // so leaving them in inputTokens would price the premium tokens at the full rate.
  const client = createFakeTransportClient(usageChunks({
    prompt_tokens: 1500,
    completion_tokens: 7,
    total_tokens: 1507,
    prompt_tokens_details: { cached_tokens: 1024, cache_write_tokens: 400 },
  }));
  const adapter = openAICompletionsModel({ model: "gpt-5.6", client });

  const stream = adapter.stream({
    history: [],
    instructions: "Be useful",
    tools: [],
    signal: AbortSignal.timeout(5000),
  });
  while (true) {
    const next = await stream.next();
    if (!next.done) continue;
    assertEquals(next.value, { inputTokens: 76, outputTokens: 7, cacheReadTokens: 1024, cacheWriteTokens: 400 });
    break;
  }
});

Deno.test("OpenAI Completions reports an uncached call with no cache bucket", async () => {
  const client = createFakeTransportClient(usageChunks({
    prompt_tokens: 1500,
    completion_tokens: 7,
    total_tokens: 1507,
    prompt_tokens_details: { cached_tokens: 0 },
  }));
  const adapter = openAICompletionsModel({ model: "gpt-4.1-mini", client });

  const stream = adapter.stream({
    history: [],
    instructions: "Be useful",
    tools: [],
    signal: AbortSignal.timeout(5000),
  });
  while (true) {
    const next = await stream.next();
    if (!next.done) continue;
    assertEquals(next.value, { inputTokens: 1500, outputTokens: 7, cacheReadTokens: 0, cacheWriteTokens: 0 });
    break;
  }
});
