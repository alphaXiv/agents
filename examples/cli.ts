import z from "zod";
import { Agent } from "../src/agent.ts";
import { Tool } from "../src/tool.ts";

const search = new Tool({
  name: "Searching the internet...",
  description: "Use when you want to search the internet",
  parameters: z.string().describe("Query parameter"),
  execute: ({ param }) => {
    if (param === "cats") {
      return JSON.stringify(["bingus.com", "bungus.com"]);
    }
    return "wtfrick.com";
  },
});

const agent = new Agent({
  model: "openrouter:nvidia/llama-3.3-nemotron-super-49b-v1.5",
  instructions: "You are a friendly assistant",
  tools: [search],
});

await agent.cli();
