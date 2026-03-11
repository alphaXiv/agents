import z from "zod";
import { Agent, Tool } from "@alphaxiv/agents";
import { lmStudioAdapter } from "../src/adapters/lmstudio.ts";

const calculator = new Tool({
  name: "Calculating...",
  description: "A simple calculator to make math operations easier!",
  parameters: z.object({
    operation: z.enum(["add", "multiply", "divide", "subtract"])
      .describe("The operator you want to calculate with"),
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
    return "oh no";
  },
});

const lmStudio = lmStudioAdapter({
  name: "localhost lmstudio",
  url: "http://localhost:1234/v1",
  apiKey: null,
  // optionally indicate the supported models
  models: ["qwen/qwen3-coder-next", "qwen/qwen3.5-35b-a3b"],
});

const agent = new Agent({
  adapter: lmStudio,
  model: "qwen/qwen3.5-35b-a3b",

  instructions: "You are a friendly assistant",
  tools: [calculator],
});

console.info(await agent.run("figure out 1 + 3"));
