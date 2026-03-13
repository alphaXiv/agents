import { assertEquals } from "@std/assert";
import { getAnthropicHistory } from "../../src/adapters/anthropic/utils.ts";
import type { ChatItem } from "../../src/types.ts";

const signal = AbortSignal.abort();

Deno.test("basic text input and output", async () => {
  const history: ChatItem[] = [
    { type: "input_text", content: "hello" },
    { type: "output_text", content: "hi there" },
  ];

  const result = await getAnthropicHistory(history, [], signal);
  assertEquals(result, [
    { role: "user", content: [{ type: "text", text: "hello" }] },
    { role: "assistant", content: [{ type: "text", text: "hi there" }] },
  ]);
});

Deno.test("single tool use and result", async () => {
  const history: ChatItem[] = [
    { type: "input_text", content: "search for cats" },
    {
      type: "tool_use",
      tool_use_id: "t1",
      kind: "search",
      content: '{"q":"cats"}',
    },
    {
      type: "tool_result_text",
      tool_use_id: "t1",
      content: "found cats",
    },
  ];

  const result = await getAnthropicHistory(history, [], signal);
  assertEquals(result, [
    { role: "user", content: [{ type: "text", text: "search for cats" }] },
    {
      role: "assistant",
      content: [{
        type: "tool_use",
        id: "t1",
        name: "search",
        input: { q: "cats" },
      }],
    },
    {
      role: "user",
      content: [{
        type: "tool_result",
        tool_use_id: "t1",
        content: "found cats",
        is_error: false,
      }],
    },
  ]);
});

Deno.test("tool_result_file image is buffered after tool_results", async () => {
  const history: ChatItem[] = [
    { type: "input_text", content: "show me" },
    { type: "tool_use", tool_use_id: "t1", kind: "search", content: "{}" },
    { type: "tool_use", tool_use_id: "t2", kind: "image_tool" },
    {
      type: "tool_result_text",
      tool_use_id: "t1",
      content: "search result",
    },
    {
      type: "tool_result_text",
      tool_use_id: "t2",
      content: "here is the image",
    },
    {
      type: "tool_result_file",
      tool_use_id: "t2",
      kind: "image/png",
      content: "https://example.com/img.png",
    },
    { type: "output_text", content: "done" },
  ];

  const result = await getAnthropicHistory(history, [], signal);

  // tool_result_file image should appear after both tool_results,
  // flushed when output_text is encountered
  assertEquals(result, [
    { role: "user", content: [{ type: "text", text: "show me" }] },
    {
      role: "assistant",
      content: [{
        type: "tool_use",
        id: "t1",
        name: "search",
        input: {},
      }],
    },
    {
      role: "assistant",
      content: [{
        type: "tool_use",
        id: "t2",
        name: "image_tool",
        input: {},
      }],
    },
    {
      role: "user",
      content: [{
        type: "tool_result",
        tool_use_id: "t1",
        content: "search result",
        is_error: false,
      }],
    },
    {
      role: "user",
      content: [{
        type: "tool_result",
        tool_use_id: "t2",
        content: "here is the image",
        is_error: false,
      }],
    },
    {
      role: "user",
      content: [{
        type: "image",
        source: { type: "url", url: "https://example.com/img.png" },
      }],
    },
    {
      role: "assistant",
      content: [{ type: "text", text: "done" }],
    },
  ]);
});

Deno.test("input_file is NOT buffered", async () => {
  const history: ChatItem[] = [
    { type: "input_text", content: "what is this?" },
    {
      type: "input_file",
      kind: "image/png",
      content: "https://example.com/photo.png",
    },
    { type: "output_text", content: "it's a cat" },
  ];

  const result = await getAnthropicHistory(history, [], signal);

  // input_file should be placed immediately, not buffered
  assertEquals(result, [
    { role: "user", content: [{ type: "text", text: "what is this?" }] },
    {
      role: "user",
      content: [{
        type: "image",
        source: { type: "url", url: "https://example.com/photo.png" },
      }],
    },
    { role: "assistant", content: [{ type: "text", text: "it's a cat" }] },
  ]);
});

Deno.test("buffer flushes at end of history", async () => {
  const history: ChatItem[] = [
    { type: "input_text", content: "go" },
    { type: "tool_use", tool_use_id: "t1", kind: "img" },
    { type: "tool_result_text", tool_use_id: "t1", content: "got it" },
    {
      type: "tool_result_file",
      tool_use_id: "t1",
      kind: "image/png",
      content: "https://example.com/img.png",
    },
  ];

  const result = await getAnthropicHistory(history, [], signal);

  // image should still appear at the end, flushed by the final flush
  assertEquals(result[result.length - 1], {
    role: "user",
    content: [{
      type: "image",
      source: { type: "url", url: "https://example.com/img.png" },
    }],
  });
  // tool_result comes before the image
  assertEquals(result[result.length - 2], {
    role: "user",
    content: [{
      type: "tool_result",
      tool_use_id: "t1",
      content: "got it",
      is_error: false,
    }],
  });
});

Deno.test("error tool results have is_error true", async () => {
  const history: ChatItem[] = [
    { type: "input_text", content: "do it" },
    { type: "tool_use", tool_use_id: "t1", kind: "fail" },
    {
      type: "tool_result_text",
      tool_use_id: "t1",
      content: "Error: something broke",
    },
  ];

  const result = await getAnthropicHistory(history, [], signal);
  const toolResult = result[2];
  assertEquals(toolResult, {
    role: "user",
    content: [{
      type: "tool_result",
      tool_use_id: "t1",
      content: "Error: something broke",
      is_error: true,
    }],
  });
});

Deno.test("pdf tool_result_file is buffered", async () => {
  const history: ChatItem[] = [
    { type: "input_text", content: "get pdf" },
    { type: "tool_use", tool_use_id: "t1", kind: "fetcher" },
    { type: "tool_result_text", tool_use_id: "t1", content: "here" },
    {
      type: "tool_result_file",
      tool_use_id: "t1",
      kind: "application/pdf",
      content: "https://example.com/doc.pdf",
    },
    { type: "output_text", content: "got it" },
  ];

  const result = await getAnthropicHistory(history, [], signal);

  // pdf should appear after tool_result, before output_text
  assertEquals(result[3], {
    role: "user",
    content: [{
      type: "document",
      source: { type: "url", url: "https://example.com/doc.pdf" },
    }],
  });
  assertEquals(result[4], {
    role: "assistant",
    content: [{ type: "text", text: "got it" }],
  });
});

Deno.test("multi-turn: buffer flushes per turn, not across turns", async () => {
  const history: ChatItem[] = [
    // Turn 1: user asks, model calls two tools (one returns image)
    { type: "input_text", content: "find cats and show me one" },
    {
      type: "tool_use",
      tool_use_id: "t1",
      kind: "search",
      content: '{"q":"cats"}',
    },
    { type: "tool_use", tool_use_id: "t2", kind: "image_gen" },
    { type: "tool_result_text", tool_use_id: "t1", content: "found 3 cats" },
    { type: "tool_result_text", tool_use_id: "t2", content: "generated cat" },
    {
      type: "tool_result_file",
      tool_use_id: "t2",
      kind: "image/png",
      content: "https://example.com/cat.png",
    },
    // Model responds (flushes buffer from turn 1)
    { type: "output_text", content: "here's a cat!" },

    // Turn 2: user asks follow-up, model calls another tool with image
    { type: "input_text", content: "now show me a dog" },
    { type: "tool_use", tool_use_id: "t3", kind: "image_gen" },
    { type: "tool_result_text", tool_use_id: "t3", content: "generated dog" },
    {
      type: "tool_result_file",
      tool_use_id: "t3",
      kind: "image/png",
      content: "https://example.com/dog.png",
    },
    // Model responds (flushes buffer from turn 2)
    { type: "output_text", content: "here's a dog!" },
  ];

  const result = await getAnthropicHistory(history, [], signal);
  assertEquals(result, [
    // Turn 1
    {
      role: "user",
      content: [{ type: "text", text: "find cats and show me one" }],
    },
    {
      role: "assistant",
      content: [{
        type: "tool_use",
        id: "t1",
        name: "search",
        input: { q: "cats" },
      }],
    },
    {
      role: "assistant",
      content: [{ type: "tool_use", id: "t2", name: "image_gen", input: {} }],
    },
    {
      role: "user",
      content: [{
        type: "tool_result",
        tool_use_id: "t1",
        content: "found 3 cats",
        is_error: false,
      }],
    },
    {
      role: "user",
      content: [{
        type: "tool_result",
        tool_use_id: "t2",
        content: "generated cat",
        is_error: false,
      }],
    },
    // cat image flushed here (before output_text)
    {
      role: "user",
      content: [{
        type: "image",
        source: { type: "url", url: "https://example.com/cat.png" },
      }],
    },
    { role: "assistant", content: [{ type: "text", text: "here's a cat!" }] },

    // Turn 2
    { role: "user", content: [{ type: "text", text: "now show me a dog" }] },
    {
      role: "assistant",
      content: [{ type: "tool_use", id: "t3", name: "image_gen", input: {} }],
    },
    {
      role: "user",
      content: [{
        type: "tool_result",
        tool_use_id: "t3",
        content: "generated dog",
        is_error: false,
      }],
    },
    // dog image flushed here (before output_text)
    {
      role: "user",
      content: [{
        type: "image",
        source: { type: "url", url: "https://example.com/dog.png" },
      }],
    },
    { role: "assistant", content: [{ type: "text", text: "here's a dog!" }] },
  ]);
});
