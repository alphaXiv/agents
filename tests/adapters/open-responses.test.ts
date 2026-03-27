import { assertEquals, assertRejects } from "@std/assert";
import type OpenAI from "openai";
import z from "zod";
import { OpenResponsesAdapter } from "../../src/adapters/open_responses/adapter.ts";
import { normalizeOpenResponsesTools } from "../../src/adapters/open_responses/tools.ts";
import { OpenAIModel } from "../../src/adapters/openai/model.ts";
import { Tool } from "../../src/tool.ts";

function createMockClient(
  events: unknown[],
  finalResponse: { usage: { input_tokens: number; output_tokens: number } },
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

  const adapter = new OpenResponsesAdapter({
    model: "test-model",
    name: "Test Provider",
    client: createMockClient([], { usage: { input_tokens: 0, output_tokens: 0 } }),
  });

  const originalFetch = globalThis.fetch;
  globalThis.fetch = ((input: string | URL | Request) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
    return Promise.resolve(new Response(url.includes("cats.csv") ? "name,count\ncat,2" : "unused"));
  }) as typeof fetch;

  try {
    const history = await adapter.getHistory(
      [
        { type: "input_text", content: "hello" },
        { type: "output_text", content: "hi there" },
        { type: "output_reasoning", content: "I should search first." },
        { type: "tool_use", tool_use_id: "call_1", kind: searchTool.normalizedName, content: '"cats"' },
        { type: "tool_result_text", tool_use_id: "call_1", content: "found 2 results" },
        { type: "tool_result_file", tool_use_id: "call_1", kind: "text/csv", content: "https://example.com/cats.csv" },
        { type: "input_file", kind: "application/pdf", content: "https://example.com/cats.pdf" },
      ],
      normalizeOpenResponsesTools([searchTool]),
      AbortSignal.abort(),
    );

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
  const adapter = new OpenResponsesAdapter({
    model: "test-model",
    name: "Test Provider",
    client: createMockClient([], { usage: { input_tokens: 0, output_tokens: 0 } }),
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

Deno.test("Open Responses stream maps text, reasoning, refusal, and function calls", async () => {
  let capturedRequest: unknown;

  const searchTool = new Tool({
    name: "Search",
    description: "Search for documents",
    parameters: z.string(),
    execute: () => "unused",
  });

  const adapter = new OpenResponsesAdapter({
    model: "test-model",
    name: "Test Provider",
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
      assertEquals(next.value, { inputTokens: 11, outputTokens: 7 });
      break;
    }
    items.push(next.value);
  }

  assertEquals(items, [
    { type: "delta_output_text", delta: "Hello", index: 0 },
    { type: "delta_output_reasoning", delta: "Need tool", index: 1 },
    { type: "tool_use_start", tool_use_id: "call_1", kind: "search", index: 2 },
    { type: "tool_use", tool_use_id: "call_1", kind: "search", content: '"cats"', index: 2 },
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
        $schema: "https://json-schema.org/draft/2020-12/schema",
        type: "object",
        properties: {
          content: { type: "string" },
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
          $schema: "https://json-schema.org/draft/2020-12/schema",
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

Deno.test("OpenAIModel defaults effort for reasoning models", () => {
  const model = new OpenAIModel({
    model: "gpt-5",
    apiKey: "test-key",
  });

  assertEquals(model.effort, "medium");
});

Deno.test("OpenAIModel omits reasoning for non-reasoning models", () => {
  const model = new OpenAIModel({
    model: "gpt-4.1-mini",
    apiKey: "test-key",
  });

  assertEquals(model.effort, undefined);
});

Deno.test("OpenAIModel stores configured service tier", () => {
  const model = new OpenAIModel({
    model: "gpt-4.1-mini",
    apiKey: "test-key",
    serviceTier: "flex",
  });

  assertEquals(model.serviceTier, "flex");
});

Deno.test("Open Responses respects explicitly configured supported mime types", async () => {
  const adapter = new OpenResponsesAdapter({
    model: "test-model",
    name: "Test Provider",
    client: createMockClient([], { usage: { input_tokens: 0, output_tokens: 0 } }),
    supportedMimeTypes: ["text/plain"],
  });

  await assertRejects(
    () =>
      adapter.getHistory(
        [{ type: "input_file", kind: "image/png", content: "https://example.com/cat.png" }],
        [],
        AbortSignal.abort(),
      ),
    Error,
    "does not support media type",
  );
});

Deno.test("OpenAIModel infers supported mime types from model modalities", async () => {
  const multimodalModel = new OpenAIModel({
    model: "gpt-4.1-mini",
    apiKey: "test-key",
  });
  const textOnlyModel = new OpenAIModel({
    model: "gpt-oss-20b",
    apiKey: "test-key",
  });

  const multimodalHistory = await multimodalModel.adapter.getHistory(
    [{ type: "input_file", kind: "image/png", content: "https://example.com/cat.png" }],
    [],
    AbortSignal.abort(),
  );

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
      textOnlyModel.adapter.getHistory(
        [{ type: "input_file", kind: "image/png", content: "https://example.com/cat.png" }],
        [],
        AbortSignal.abort(),
      ),
    Error,
    "does not support media type",
  );

  const documentHistory = await multimodalModel.adapter.getHistory(
    [{ type: "input_file", kind: "application/msword", content: "https://example.com/doc.doc" }],
    [],
    AbortSignal.abort(),
  );

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

  const adapter = new OpenResponsesAdapter({
    model: "test-model",
    name: "Test Provider",
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
    value: { inputTokens: 0, outputTokens: 0 },
  });

  assertEquals((capturedRequest as { parallel_tool_calls?: boolean }).parallel_tool_calls, false);
});
