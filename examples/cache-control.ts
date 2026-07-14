import { Agent } from "../mod.ts";
import { anthropicModel } from "../src/adapters/anthropic/adapter.ts";

const agent = new Agent({
  model: anthropicModel({
    model: "claude-opus-4-8",
    cache: { ttl: "1h" },
  }),
  instructions: "You are a helpful assistant",
});

console.log(result.output);
