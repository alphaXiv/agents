import { assertEquals } from "@std/assert";
import { addStreamItem } from "../src/client.ts";
import type { ChatItem, StreamItem, WithTraceId } from "../src/types.ts";

Deno.test("Basic addStreamItem", () => {
  const basic: StreamItem[] = [
    { type: "delta_output_text", index: 0, delta: "572" },
    { type: "delta_output_text", index: 0, delta: "361" },
    { type: "delta_output_text", index: 0, delta: "189" },
    { type: "delta_output_text", index: 0, delta: "4" },
  ];

  const output: ChatItem[] = [];
  for (const part of basic) {
    addStreamItem(output, part);
  }
  assertEquals(output, [{ type: "output_text", content: "5723611894" }]);
});

Deno.test("Basic addStreamItem with traces", () => {
  const basic: WithTraceId<StreamItem>[] = [
    { type: "delta_output_reasoning", index: 0, delta: "123", trace: "a" },
    { type: "delta_output_reasoning", index: 0, delta: "456", trace: "a" },
    { type: "delta_output_reasoning", index: 0, delta: "789", trace: "a" },
    { type: "delta_output_text", index: 1, delta: "yo", trace: "b" },
  ];

  const output: WithTraceId<ChatItem>[] = [];
  for (const part of basic) {
    addStreamItem(output, part);
  }
  assertEquals(output, [{
    type: "output_reasoning",
    content: "123456789",
    trace: "a",
  }, {
    type: "output_text",
    content: "yo",
    trace: "b",
  }]);
});

void function typeTest() {
  const streamItem: StreamItem = null as unknown as StreamItem;
  const traceStreamItem: WithTraceId<StreamItem> = null!;
  const chatItems: ChatItem[] = [];
  const traceChatItems: WithTraceId<ChatItem>[] = [];
  addStreamItem(chatItems, streamItem);
  addStreamItem(chatItems, traceStreamItem);
  addStreamItem(traceChatItems, traceStreamItem);
  // @ts-expect-error stream item is missing trace
  addStreamItem(traceChatItems, streamItem);
};

Deno.test("addStreamItem tool_use_start creates placeholder, tool_use fills it", () => {
  const items: StreamItem[] = [
    { type: "tool_use_start", index: 0, tool_use_id: "t1", kind: "my_tool" },
    {
      type: "tool_use",
      index: 0,
      tool_use_id: "t1",
      kind: "my_tool",
      content: '{"x":1}',
    },
  ];

  const output: ChatItem[] = [];
  for (const part of items) {
    addStreamItem(output, part);
  }
  assertEquals(output, [{
    type: "tool_use",
    tool_use_id: "t1",
    kind: "my_tool",
    content: '{"x":1}',
  }]);
});

Deno.test("Basic addStreamItem out of order", () => {
  const basic: StreamItem[] = [
    { type: "delta_output_reasoning", index: 0, delta: "Hello" },
    { type: "delta_output_text", index: 1, delta: "572" },
    { type: "delta_output_text", index: 1, delta: "361" },
    { type: "delta_output_text", index: 1, delta: "189" },
    { type: "delta_output_text", index: 1, delta: "4" },
    { type: "delta_output_reasoning", index: 0, delta: " world!" },
  ];

  const output: ChatItem[] = [];
  for (const part of basic) {
    addStreamItem(output, part);
  }
  assertEquals(output, [
    { type: "output_reasoning", content: "Hello world!" },
    {
      type: "output_text",
      content: "5723611894",
    },
  ]);
});
