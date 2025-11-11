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

const complexSearch = new Tool({
  name: "Searching the paper database...",
  description: "Use when you want to search the internet",
  parameters: z.object({
    subcategories: z
      .array(z.enum(["ai", "ml", "whatever"]))
      .optional()
      .describe("List of arXiv subcategories to filter papers by."),
    categories: z
      .array(z.enum(["Computer Science", "Physics"]))
      .optional()
      .describe("List of arXiv categories to filter papers by"),
    days_ago: z
      .enum(["7", "30", "90", "-1"])
      .describe(
        "Time interval to filter papers by. Must be one of: 7, 30, 90, or -1 (for all time).",
      ),
  }),
  execute: () => {
    return "no results";
  },
});

const pingSupport = new Tool({
  name: "Pinging support...",
  description:
    "If a user asks for support, automatically ping support right away as soon as possible. They will support the user shortly.",
  parameters: z.void(),
  execute: () => {
    return "Successfully pinged support!";
  },
});

const agent = new Agent({
  model: "openrouter:qwen/qwen3-next-80b-a3b-instruct",
  instructions: "You are a friendly assistant",
  tools: [search, calculator, pingSupport, complexSearch],
  reasoningEffort: "normal",
});

await agent.cli();
