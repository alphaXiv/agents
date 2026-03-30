import z from "zod";
import { Agent, repl, Tool } from "../mod.ts";

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
  execute: ({ operation, left, right }) => {
    if (operation === "add") {
      return (left + right).toString();
    } else if (operation === "multiply") {
      return (left * right).toString();
    } else if (operation === "divide") {
      return (left / right).toString();
    } else if (operation === "subtract") {
      return (left - right).toString();
    }
    operation satisfies never;
    return "";
  },
});

const search = new Tool({
  name: "Searching the internet...",
  description: "Use when you want to search the internet",
  parameters: z.string().describe("Query parameter"),
  execute: (param) => {
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

const getMysteryImage = new Tool({
  name: "Get mystery image...",
  description: "Get an image",
  parameters: z.void(),
  execute: () => {
    return [
      {
        type: "tool_result_text",
        content: "Below is the image I was talking about",
      },
      {
        type: "tool_result_file",
        kind: "image/png",
        content: "https://paper-assets.alphaxiv.org/image/2510.18234v1.png",
      },
    ];
  },
});

const agent = new Agent({
  model: "openai:gpt-5.4-nano",
  instructions: "You are a friendly assistant",
  tools: [search, calculator, pingSupport, complexSearch, getMysteryImage],
});

await repl(agent);
