import { assertEquals } from "@std/assert";
import { addStreamItem } from "../src/client.ts";
import type { ChatItem, StreamItem } from "../src/types.ts";

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
