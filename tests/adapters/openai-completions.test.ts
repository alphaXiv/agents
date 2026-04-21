import { assert, assertEquals, assertThrows } from "@std/assert";
import type OpenAI from "openai";
import z from "zod";
import { OpenAICompletionsAdapter } from "../../src/adapters/openai_completions/adapter.ts";
import { normalizeOpenAICompletionsTools } from "../../src/adapters/openai_completions/tools.ts";
import { RETRY_RESUMABILITY_PROMPT } from "../../src/constants.ts";
import { Tool } from "../../src/tool.ts";

function createMockClient(
  events: unknown[],
  totalUsage: { prompt_tokens: number; completion_tokens: number } | undefined,
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
            totalUsage() {
              return totalUsage;
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

  const adapter = new OpenAICompletionsAdapter({
    model: "test-model",
    name: "Test Provider",
    client: createMockClient([], undefined),
    pdfSupport: { mode: "native" },
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
    const history = await adapter.getHistory(
      [
        { type: "input_text", content: "hello" },
        { type: "output_text", content: "hi there" },
        { type: "output_reasoning", content: "I should search first." },
        { type: "tool_use", tool_use_id: "call_1", kind: searchTool.name, content: '"cats"' },
        { type: "tool_result_text", tool_use_id: "call_1", content: "found 2 results" },
        { type: "tool_result_file", tool_use_id: "call_1", kind: "text/csv", content: "https://example.com/cats.csv" },
        { type: "input_file", kind: "application/pdf", content: "https://example.com/cats.pdf" },
      ],
      "Be useful",
      normalizeOpenAICompletionsTools([searchTool]),
      AbortSignal.abort(),
    );

    assertEquals(history, [
      { role: "system", content: "Be useful" },
      { role: "user", content: "hello" },
      { role: "assistant", content: "hi there" },
      {
        role: "assistant",
        content: null,
        tool_calls: [{
          id: "call_1",
          type: "function",
          function: {
            name: "search",
            arguments: '{"content":"cats"}',
          },
        }],
      },
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
  const adapter = new OpenAICompletionsAdapter({
    model: "test-model",
    name: "Test Provider",
    client: createMockClient([], undefined),
  });

  const retryHistory = await adapter.getHistory(
    [
      { type: "output_text", content: '{"broken": true}' },
      {
        type: "output_text",
        content:
          "Sorry, my output has an error:\nboom\nI will try again to produce a JSON response that conforms to the expected schema.",
      },
    ],
    "Be useful",
    [],
    AbortSignal.abort(),
  );

  assertEquals(retryHistory, [
    { role: "system", content: "Be useful" },
    { role: "assistant", content: '{"broken": true}' },
    {
      role: "user",
      content:
        "Sorry, my output has an error:\nboom\nI will try again to produce a JSON response that conforms to the expected schema.",
    },
  ]);

  const resumableHistory = await adapter.getHistory(
    [{ type: "output_text", content: "partial response" }],
    "Be useful",
    [],
    AbortSignal.abort(),
  );

  assertEquals(resumableHistory, [
    { role: "system", content: "Be useful" },
    { role: "assistant", content: "partial response" },
    { role: "system", content: RETRY_RESUMABILITY_PROMPT },
  ]);
});

Deno.test("OpenAI Completions replays missing tool definitions with normalized names", async () => {
  const adapter = new OpenAICompletionsAdapter({
    model: "test-model",
    name: "Test Provider",
    client: createMockClient([], undefined),
  });

  const history = await adapter.getHistory(
    [
      { type: "tool_use", tool_use_id: "call_1", kind: "Load Project Snapshot", content: undefined },
      { type: "tool_result_text", tool_use_id: "call_1", content: "ready" },
    ],
    "Be useful",
    [],
    AbortSignal.abort(),
  );

  assertEquals(history, [
    { role: "system", content: "Be useful" },
    {
      role: "assistant",
      content: null,
      tool_calls: [{
        id: "call_1",
        type: "function",
        function: {
          name: "load_project_snapshot",
          arguments: "{}",
        },
      }],
    },
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

  const adapter = new OpenAICompletionsAdapter({
    model: "test-model",
    name: "Test Provider",
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

  const adapter = new OpenAICompletionsAdapter({
    model: "test-model",
    name: "Test Provider",
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
      assertEquals(next.value, { inputTokens: 11, outputTokens: 7 });
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
      new OpenAICompletionsAdapter({
        model: "test-model",
        name: "Test Provider",
        client: createMockClient([], undefined),
        supportedMimeTypes: ["text/plain"],
        pdfSupport: { mode: "native" },
      }),
    Error,
    "pdfSupport requires application/pdf to be included in supportedMimeTypes",
  );
});
