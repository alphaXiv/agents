import { Agent, registerAdapter } from "../mod.ts";
import { assertEquals } from "@std/assert";
import { TestingAdapter } from "./utils/testing-adapter.ts";
import { enableDebugMode } from "../src/constants.ts";
import { StreamCollector } from "../src/client.ts";

registerAdapter("__testing", TestingAdapter);
enableDebugMode();

Deno.test("Basic streaming test", async () => {
  const agent = new Agent({
    model: "__testing:deterministic",
    instructions: "Basic test",
  });
  const run = agent.stream("<nothing>");

  let count = 0;
  const collector = new StreamCollector();
  for await (const part of run) {
    collector.add(part);
    count++;
  }

  assertEquals(count, 18);
  assertEquals(collector.items, [{
    type: "output_text",
    content: "Basic test worked!",
  }]);
});
