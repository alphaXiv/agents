import z from "zod";
import { Agent, Tool } from "../mod.ts";

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
  model: "google:gemini-2.0-flash",
  instructions: "You are a friendly assistant",
  tools: [search],
});

await agent.cli();
