import z from "zod";
import { Agent, Tool } from "../mod.ts";

const calculator = new Tool({
  name: "Calculating...",
  description: "A simple calculator to make math operations easier!",
  parameters: z.object({
    operation: z.enum(["add", "multiply", "divide", "subtract"]).describe(
      "The operator you want to calculate with",
    ),
    left: z.number(),
    right: z.number(),
  }),
  execute: ({ param }) => {
    if (param.operation === "add") {
      return (param.left + param.right).toString();
    } else if (param.operation === "multiply") {
      return (param.left * param.right).toString();
    } else if (param.operation === "divide") {
      return (param.left / param.right).toString();
    } else if (param.operation === "subtract") {
      return (param.left - param.right).toString();
    }
    param.operation satisfies never;
    return "";
  },
});

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
  model: "google:gemini-2.5-flash-lite",
  instructions: "You are a friendly assistant",
  tools: [search, calculator],
});

await agent.cli();
