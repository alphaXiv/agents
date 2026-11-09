import { assert, assertEquals } from "@std/assert";
import { getOmittedImages } from "../../src/adapters/shared/media.ts";
import type { ChatItem } from "../../src/types.ts";

function createImages(count: number): ChatItem[] {
  return Array.from({ length: count }, (_, index) => ({
    type: "input_file",
    kind: "image/png",
    content: `https://example.com/${index}.png`,
  }));
}

Deno.test("getOmittedImages() keeps a request within the cap by omitting the oldest images", () => {
  for (let count = 0; count <= 130; count += 1) {
    const images = createImages(count);
    const omitted = getOmittedImages(images, 50);

    assert(count - omitted.size <= 50);
    assertEquals([...omitted], images.slice(0, omitted.size));
  }
});

Deno.test("getOmittedImages() omits every image when the cap is zero", () => {
  const images = createImages(3);

  assertEquals(getOmittedImages(images, 0), new Set(images));
});

Deno.test("getOmittedImages() keeps earlier omissions as new images arrive, so the prompt prefix stays cached", () => {
  const images = createImages(52);

  assertEquals(getOmittedImages(images, 50), getOmittedImages(images.slice(0, 51), 50));
});

Deno.test("getOmittedImages() counts images returned by tools and leaves other files alone", () => {
  const toolImage: ChatItem = {
    type: "tool_result_file",
    tool_use_id: "call_1",
    kind: "image/png",
    content: "https://example.com/screenshot.png",
  };
  const history: ChatItem[] = [
    { type: "input_file", kind: "application/pdf", content: "https://example.com/paper.pdf" },
    toolImage,
    { type: "input_file", kind: "image/png", content: "https://example.com/photo.png" },
  ];

  assertEquals(getOmittedImages(history, 1), new Set([toolImage]));
});
