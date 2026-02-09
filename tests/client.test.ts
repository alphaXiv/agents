import { assertEquals } from "@std/assert";
import { StreamCollector } from "../src/client.ts";
import type { StreamItem } from "../src/types.ts";
import { enableDebugMode } from "../src/constants.ts";

enableDebugMode();

Deno.test("Basic streaming", () => {
  const basic: StreamItem[] = [
    { type: "delta_output_text", index: 0, delta: "572" },
    { type: "delta_output_text", index: 0, delta: "361" },
    { type: "delta_output_text", index: 0, delta: "189" },
    { type: "delta_output_text", index: 0, delta: "4" },
  ];

  const collector = new StreamCollector();
  for (const part of basic) {
    collector.add(part);
  }

  assertEquals(collector.items, [{
    type: "output_text",
    content: "5723611894",
  }]);
});

Deno.test("Out of order reasoning and text", () => {
  const basic: StreamItem[] = [
    { type: "delta_output_reasoning", index: 0, delta: "Hello" },
    { type: "delta_output_text", index: 1, delta: "572" },
    { type: "delta_output_text", index: 1, delta: "361" },
    { type: "delta_output_text", index: 1, delta: "189" },
    { type: "delta_output_text", index: 1, delta: "4" },
    { type: "delta_output_reasoning", index: 0, delta: " world!" },
  ];

  const collector = new StreamCollector();
  for (const part of basic) {
    collector.add(part);
  }

  assertEquals(collector.items, [
    { type: "output_reasoning", content: "Hello world!" },
    { type: "output_text", content: "5723611894" },
  ]);
});

Deno.test("Interleaved deltas", () => {
  const items: StreamItem[] = [
    { type: "delta_output_text", index: 1, delta: "B" },
    { type: "delta_output_text", index: 0, delta: "A" },
    { type: "delta_output_text", index: 1, delta: "!" },
    { type: "delta_output_text", index: 0, delta: "!" },
  ];

  const collector = new StreamCollector();
  for (const part of items) {
    collector.add(part);
  }

  assertEquals(collector.items.length, 2);
  assertEquals(collector.items[0], { type: "output_text", content: "A!" });
  assertEquals(collector.items[1], { type: "output_text", content: "B!" });
});

Deno.test("Mixed types out of order", () => {
  const items: StreamItem[] = [
    {
      type: "tool_use",
      index: 2,
      tool_use_id: "tool_1",
      kind: "calculator",
      content: '{"a": 1}',
    },
    { type: "delta_output_reasoning", index: 0, delta: "Thinking" },
    {
      type: "tool_result_text",
      index: 3,
      tool_use_id: "tool_1",
      content: "42",
    },
    { type: "delta_output_text", index: 1, delta: "The answer is 42" },
    { type: "delta_output_reasoning", index: 0, delta: "..." },
  ];

  const collector = new StreamCollector();
  for (const part of items) {
    collector.add(part);
  }

  assertEquals(collector.items, [
    { type: "output_reasoning", content: "Thinking..." },
    { type: "output_text", content: "The answer is 42" },
    {
      type: "tool_use",
      tool_use_id: "tool_1",
      kind: "calculator",
      content: '{"a": 1}',
    },
    { type: "tool_result_text", tool_use_id: "tool_1", content: "42" },
  ]);
});

Deno.test("Tool result file", () => {
  const collector = new StreamCollector();
  collector.add({
    type: "tool_result_file",
    index: 0,
    tool_use_id: "tool_1",
    kind: "image/png",
    content: "https://example.com/image.png",
  });

  assertEquals(collector.items, [
    {
      type: "tool_result_file",
      tool_use_id: "tool_1",
      kind: "image/png",
      content: "https://example.com/image.png",
    },
  ]);
});
