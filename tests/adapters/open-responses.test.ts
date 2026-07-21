import { assert, assertEquals, assertRejects } from "@std/assert";
import type OpenAI from "openai";
import z from "zod";
import { openResponsesModel } from "../../src/adapters/open_responses/adapter.ts";
import { getOpenResponsesHistory } from "../../src/adapters/open_responses/history.ts";
import { normalizeOpenResponsesTools } from "../../src/adapters/open_responses/tools.ts";
import { openAIModel } from "../../src/adapters/openai/adapter.ts";
import { getOpenAISupportedMimeTypes } from "../../src/adapters/openai/mimes.ts";
import { getModelModalities } from "../../src/adapters/openai/models.ts";
import { Tool } from "../../src/tool.ts";

function createMockClient(
  events: unknown[],
  finalResponse: {
    usage: {
      input_tokens: number;
      output_tokens: number;
      input_tokens_details?: { cached_tokens?: number; cache_write_tokens?: number };
    };
  },
  captureRequest?: (request: unknown) => void,
) {
  return {
    responses: {
      stream(request: unknown) {
        captureRequest?.(request);

        return {
          async *[Symbol.asyncIterator]() {
            for (const event of events) {
              yield event;
            }
          },
          finalResponse() {
            return finalResponse;
          },
        };
      },
    },
  } as unknown as Pick<OpenAI, "responses">;
}

function getItemId(item: unknown): string {
  return (item as { id: string }).id;
}

Deno.test("Open Responses history normalizes assistant, reasoning, tool, and file items", async () => {
  const searchTool = new Tool({
    name: "Search",
    description: "Search for documents",
    parameters: z.string(),
    execute: () => "unused",
  });

  const originalFetch = globalThis.fetch;
  globalThis.fetch = ((input: string | URL | Request) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
    return Promise.resolve(new Response(url.includes("cats.csv") ? "name,count\ncat,2" : "unused"));
  }) as typeof fetch;

  try {
    const history = await getOpenResponsesHistory({
      model: "test-model",
      history: [
        { type: "input_text", content: "hello" },
        { type: "output_text", content: "hi there" },
        { type: "output_reasoning", content: "I should search first." },
        { type: "tool_use", tool_use_id: "call_1", kind: searchTool.name, content: '"cats"' },
        { type: "tool_result_text", tool_use_id: "call_1", content: "found 2 results" },
        { type: "tool_result_file", tool_use_id: "call_1", kind: "text/csv", content: "https://example.com/cats.csv" },
        { type: "input_file", kind: "application/pdf", content: "https://example.com/cats.pdf" },
      ],
      normalizedTools: normalizeOpenResponsesTools([searchTool]),
      signal: AbortSignal.abort(),
    });

    assertEquals(history, [
      {
        type: "message",
        role: "user",
        status: "completed",
        content: [{ type: "input_text", text: "hello" }],
      },
      {
        id: getItemId(history[1]),
        type: "message",
        role: "assistant",
        status: "completed",
        content: [{ type: "output_text", text: "hi there", annotations: [] }],
      },
      {
        id: getItemId(history[2]),
        type: "function_call",
        call_id: "call_1",
        name: "search",
        arguments: '{"content":"cats"}',
        status: "completed",
      },
      {
        id: getItemId(history[3]),
        type: "function_call_output",
        status: "completed",
        call_id: "call_1",
        output: [
          { type: "input_text", text: "found 2 results" },
          { type: "input_text", text: '<file mime-type="text/csv">name,count\ncat,2</file>' },
        ],
      },
      {
        type: "message",
        role: "user",
        status: "completed",
        content: [{
          type: "input_file",
          file_data: "data:application/pdf;base64,dW51c2Vk",
          filename: "cats.pdf",
        }],
      },
    ]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

Deno.test("Open Responses retry feedback is replayed as a user message", async () => {
  const history = await getOpenResponsesHistory({
    model: "test-model",
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
    {
      id: getItemId(history[0]),
      type: "message",
      role: "assistant",
      status: "completed",
      content: [{ type: "output_text", text: '{"broken": true}', annotations: [] }],
    },
    {
      type: "message",
      role: "user",
      status: "completed",
      content: [{
        type: "input_text",
        text:
          "Sorry, my output has an error:\nboom\nI will try again to produce a JSON response that conforms to the expected schema.",
      }],
    },
  ]);
});

Deno.test("Open Responses replays missing tool definitions with normalized names", async () => {
  const history = await getOpenResponsesHistory({
    model: "test-model",
    history: [
      { type: "tool_use", tool_use_id: "call_1", kind: "Load Project Snapshot", content: undefined },
      { type: "tool_result_text", tool_use_id: "call_1", content: "ready" },
    ],
    normalizedTools: [],
    signal: AbortSignal.abort(),
  });

  assertEquals(history, [
    {
      id: getItemId(history[0]),
      type: "function_call",
      call_id: "call_1",
      name: "load_project_snapshot",
      arguments: "{}",
      status: "completed",
    },
    {
      id: getItemId(history[1]),
      type: "function_call_output",
      status: "completed",
      call_id: "call_1",
      output: [{ type: "input_text", text: "ready" }],
    },
  ]);
});

Deno.test("Open Responses uses reversible OpenAI compatibility for tuple and record tool schemas", () => {
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

  const [normalizedTool] = normalizeOpenResponsesTools([tupleTool]);
  const parameters = normalizedTool?.openResponses.parameters as {
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
  assert(normalizedTool?.openResponses.description?.includes("<input_requirements>"));
});

Deno.test("Open Responses restores structured output from OpenAI-compatible surrogate shapes", async () => {
  let capturedRequest: unknown;

  const adapter = openResponsesModel({
    model: "test-model",
    provider: "Test Provider",
    client: createMockClient(
      [
        {
          type: "response.output_text.delta",
          sequence_number: 1,
          output_index: 0,
          content_index: 0,
          item_id: "msg_1",
          delta: '{"tags":{"item0":"red","item1":2},"metadata":[{"key":"alpha","value":"1"}]}',
          logprobs: [],
        },
      ],
      { usage: { input_tokens: 3, output_tokens: 2 } },
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

  assert((capturedRequest as { instructions: string }).instructions.includes("<output_requirements>"));
});

Deno.test("Open Responses stream unwraps primitive tool arguments when the done event omits the tool name", async () => {
  const searchTool = new Tool({
    name: "Search",
    description: "Search for documents",
    parameters: z.string(),
    execute: () => "unused",
  });

  const adapter = openResponsesModel({
    model: "test-model",
    provider: "Test Provider",
    client: createMockClient([
      {
        type: "response.output_item.added",
        sequence_number: 1,
        output_index: 0,
        item: {
          id: "fc_1",
          type: "function_call",
          status: "in_progress",
          call_id: "call_1",
          name: "search",
          arguments: "",
        },
      },
      {
        type: "response.function_call_arguments.done",
        sequence_number: 2,
        output_index: 0,
        item_id: "fc_1",
        arguments: '{"content":"cats"}',
      },
    ], {
      usage: { input_tokens: 0, output_tokens: 0 },
    }),
  });

  const stream = adapter.stream({
    history: [],
    instructions: "Be useful",
    tools: [searchTool],
    signal: AbortSignal.abort(),
  });

  const items = [];
  while (true) {
    const next = await stream.next();
    if (next.done) break;
    items.push(next.value);
  }

  assertEquals(items, [
    { type: "tool_use_start", tool_use_id: "call_1", kind: "Search", index: 0 },
    { type: "tool_use", tool_use_id: "call_1", kind: "Search", content: '"cats"', index: 0 },
  ]);
});

Deno.test("Open Responses synthesizes tool_use_start when arguments.done arrives without output_item.added", async () => {
  const searchTool = new Tool({
    name: "Search",
    description: "Search for documents",
    parameters: z.string(),
    execute: () => "unused",
  });

  const adapter = openResponsesModel({
    model: "test-model",
    provider: "Test Provider",
    client: createMockClient([
      {
        type: "response.function_call_arguments.done",
        sequence_number: 1,
        output_index: 0,
        item_id: "fc_1",
        name: "search",
        arguments: '{"content":"cats"}',
      },
    ], {
      usage: { input_tokens: 0, output_tokens: 0 },
    }),
  });

  const stream = adapter.stream({
    history: [],
    instructions: "Be useful",
    tools: [searchTool],
    signal: AbortSignal.abort(),
  });

  const items = [];
  while (true) {
    const next = await stream.next();
    if (next.done) break;
    items.push(next.value);
  }

  assertEquals(items, [
    { type: "tool_use_start", tool_use_id: "fc_1", kind: "Search", index: 0 },
    { type: "tool_use", tool_use_id: "fc_1", kind: "Search", content: '"cats"', index: 0 },
  ]);
});

Deno.test("Open Responses stream maps text, reasoning, refusal, and function calls", async () => {
  let capturedRequest: unknown;

  const searchTool = new Tool({
    name: "Search",
    description: "Search for documents",
    parameters: z.string(),
    execute: () => "unused",
  });

  const adapter = openResponsesModel({
    model: "test-model",
    provider: "Test Provider",
    client: createMockClient([
      {
        type: "response.output_item.added",
        sequence_number: 1,
        output_index: 0,
        item: {
          id: "msg_1",
          type: "message",
          role: "assistant",
          status: "in_progress",
          content: [],
        },
      },
      {
        type: "response.output_text.delta",
        sequence_number: 2,
        output_index: 0,
        content_index: 0,
        item_id: "msg_1",
        delta: "Hello",
        logprobs: [],
      },
      {
        type: "response.output_item.added",
        sequence_number: 3,
        output_index: 1,
        item: {
          id: "rs_1",
          type: "reasoning",
          status: "in_progress",
          summary: [],
        },
      },
      {
        type: "response.reasoning_summary_text.delta",
        sequence_number: 4,
        output_index: 1,
        item_id: "rs_1",
        summary_index: 0,
        delta: "Need tool",
      },
      {
        type: "response.output_item.added",
        sequence_number: 5,
        output_index: 2,
        item: {
          id: "fc_1",
          type: "function_call",
          status: "in_progress",
          call_id: "call_1",
          name: "search",
          arguments: "",
        },
      },
      {
        type: "response.function_call_arguments.done",
        sequence_number: 6,
        output_index: 2,
        item_id: "fc_1",
        name: "search",
        arguments: '{"content":"cats"}',
      },
      {
        type: "response.refusal.delta",
        sequence_number: 7,
        output_index: 3,
        content_index: 0,
        item_id: "msg_2",
        delta: "Nope",
      },
    ], {
      usage: { input_tokens: 11, output_tokens: 7 },
    }, (request) => {
      capturedRequest = request;
    }),
    reasoning: { effort: "medium", summary: "auto" },
    serviceTier: "priority",
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
    { type: "tool_use", tool_use_id: "call_1", kind: "Search", content: '"cats"', index: 2 },
    { type: "delta_output_text", delta: "Nope", index: 3 },
  ]);

  assertEquals(capturedRequest, {
    model: "test-model",
    input: [],
    instructions: "Be useful",
    parallel_tool_calls: true,
    service_tier: "priority",
    tools: [{
      type: "function",
      name: "search",
      description: "Search for documents",
      strict: false,
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
    }],
    text: {
      format: {
        type: "json_schema",
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
    reasoning: { effort: "medium", summary: "auto" },
    stream: true,
  });
});

Deno.test("OpenAIModel defaults effort for reasoning models", async () => {
  let capturedRequest: unknown;
  const model = openAIModel({
    model: "gpt-5.4",
    apiKey: "test-key",
    client: createMockClient([], { usage: { input_tokens: 0, output_tokens: 0 } }, (request) => {
      capturedRequest = request;
    }),
  });

  const stream = model.stream({
    history: [],
    instructions: "test",
    tools: [],
    signal: AbortSignal.abort(),
  });

  assertEquals(await stream.next(), {
    done: true,
    value: { inputTokens: 0, outputTokens: 0, cacheReadTokens: null, cacheWriteTokens: 0 },
  });
  assertEquals((capturedRequest as { reasoning?: unknown }).reasoning, { effort: "medium", summary: "auto" });
});

Deno.test("OpenAIModel omits reasoning for non-reasoning models", async () => {
  let capturedRequest: unknown;
  const model = openAIModel({
    model: "gpt-4.1-mini",
    apiKey: "test-key",
    client: createMockClient([], { usage: { input_tokens: 0, output_tokens: 0 } }, (request) => {
      capturedRequest = request;
    }),
  });

  const stream = model.stream({
    history: [],
    instructions: "test",
    tools: [],
    signal: AbortSignal.abort(),
  });

  assertEquals(await stream.next(), {
    done: true,
    value: { inputTokens: 0, outputTokens: 0, cacheReadTokens: null, cacheWriteTokens: 0 },
  });
  assertEquals((capturedRequest as { reasoning?: unknown }).reasoning, undefined);
});

Deno.test("OpenAIModel stores configured service tier", async () => {
  let capturedRequest: unknown;
  const model = openAIModel({
    model: "gpt-5.4-nano",
    apiKey: "test-key",
    serviceTier: "flex",
    client: createMockClient([], { usage: { input_tokens: 0, output_tokens: 0 } }, (request) => {
      capturedRequest = request;
    }),
  });

  const stream = model.stream({
    history: [],
    instructions: "test",
    tools: [],
    signal: AbortSignal.abort(),
  });

  assertEquals(await stream.next(), {
    done: true,
    value: { inputTokens: 0, outputTokens: 0, cacheReadTokens: null, cacheWriteTokens: 0 },
  });
  assertEquals((capturedRequest as { service_tier?: unknown }).service_tier, "flex");
});

Deno.test("Open Responses respects explicitly configured supported mime types", async () => {
  await assertRejects(
    () =>
      getOpenResponsesHistory({
        model: "test-model",
        supportedMimeTypes: ["text/plain"],
        history: [{ type: "input_file", kind: "image/png", content: "https://example.com/cat.png" }],
        normalizedTools: [],
        signal: AbortSignal.abort(),
      }),
    Error,
    "does not support media type",
  );
});

Deno.test("OpenAIModel infers supported mime types from model modalities", async () => {
  const multimodalTypes = getOpenAISupportedMimeTypes(getModelModalities("gpt-5.4-nano"));
  const textOnlyTypes = getOpenAISupportedMimeTypes(getModelModalities("gpt-oss-20b"));

  const multimodalHistory = await getOpenResponsesHistory({
    model: "gpt-5.4-nano",
    supportedMimeTypes: multimodalTypes,
    history: [{ type: "input_file", kind: "image/png", content: "https://example.com/cat.png" }],
    normalizedTools: [],
    signal: AbortSignal.abort(),
  });

  assertEquals(multimodalHistory, [{
    type: "message",
    role: "user",
    status: "completed",
    content: [{
      type: "input_image",
      image_url: "https://example.com/cat.png",
      detail: "auto",
    }],
  }]);

  await assertRejects(
    () =>
      getOpenResponsesHistory({
        model: "gpt-oss-20b",
        supportedMimeTypes: textOnlyTypes,
        history: [{ type: "input_file", kind: "image/png", content: "https://example.com/cat.png" }],
        normalizedTools: [],
        signal: AbortSignal.abort(),
      }),
    Error,
    "does not support media type",
  );

  const documentHistory = await getOpenResponsesHistory({
    model: "gpt-5.4-nano",
    supportedMimeTypes: multimodalTypes,
    history: [{ type: "input_file", kind: "application/msword", content: "https://example.com/doc.doc" }],
    normalizedTools: [],
    signal: AbortSignal.abort(),
  });

  assertEquals(documentHistory, [{
    type: "message",
    role: "user",
    status: "completed",
    content: [{
      type: "input_file",
      file_url: "https://example.com/doc.doc",
      filename: "doc.doc",
    }],
  }]);
});

Deno.test("Open Responses respects parallelToolCalls option", async () => {
  let capturedRequest: unknown;

  const searchTool = new Tool({
    name: "Search",
    description: "Search for documents",
    parameters: z.string(),
    execute: () => "unused",
  });

  const adapter = openResponsesModel({
    model: "test-model",
    provider: "Test Provider",
    client: createMockClient([], { usage: { input_tokens: 0, output_tokens: 0 } }, (request) => {
      capturedRequest = request;
    }),
    parallelToolCalls: false,
  });

  const stream = adapter.stream({
    history: [],
    instructions: "Be useful",
    tools: [searchTool],
    signal: AbortSignal.abort(),
    output: undefined,
  });

  assertEquals(await stream.next(), {
    done: true,
    value: { inputTokens: 0, outputTokens: 0, cacheReadTokens: null, cacheWriteTokens: 0 },
  });

  assertEquals((capturedRequest as { parallel_tool_calls?: boolean }).parallel_tool_calls, false);
});

Deno.test("Open Responses reports cached tokens as their own bucket, outside inputTokens", async () => {
  const client = createMockClient([], {
    usage: { input_tokens: 1500, output_tokens: 7, input_tokens_details: { cached_tokens: 1024 } },
  });
  const adapter = openResponsesModel({ model: "gpt-4.1-mini", client });

  const stream = adapter.stream({
    history: [],
    instructions: "Be useful",
    tools: [],
    signal: AbortSignal.abort(),
  });

  // OpenAI counts cached tokens inside input_tokens, so 1500 - 1024 is the uncached remainder.
  assertEquals(await stream.next(), {
    done: true,
    value: { inputTokens: 476, outputTokens: 7, cacheReadTokens: 1024, cacheWriteTokens: 0 },
  });
});

Deno.test("Open Responses splits the cache write premium out of inputTokens (GPT-5.6+)", async () => {
  // From GPT-5.6 OpenAI bills cache writes at 1.25x and counts them inside input_tokens.
  const client = createMockClient([], {
    usage: {
      input_tokens: 1500,
      output_tokens: 7,
      input_tokens_details: { cached_tokens: 1024, cache_write_tokens: 400 },
    },
  });
  const adapter = openResponsesModel({ model: "gpt-5.6", client });

  const stream = adapter.stream({ history: [], instructions: "Be useful", tools: [], signal: AbortSignal.abort() });
  assertEquals(await stream.next(), {
    done: true,
    value: { inputTokens: 76, outputTokens: 7, cacheReadTokens: 1024, cacheWriteTokens: 400 },
  });
});
