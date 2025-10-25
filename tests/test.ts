import z from "zod";
import { Agent } from "../src/agent.ts";
import { assertEquals } from "@std/assert";
import { assert } from "@std/assert/assert";
import { Tool } from "../src/tool.ts";

Deno.test("Basic input out of agents works", async () => {
  const agent = new Agent({
    model: "__testing:deterministic",
    instructions: "You are a friendly assistant",
  });
  const run = await agent.run("Hello!");
  assertEquals(run.history, [{
    type: "output_text",
    content: "Hey! How are you doing?",
  }]);
});

Deno.test("History input out of agents works", async () => {
  const agent = new Agent({
    model: "__testing:deterministic",
    instructions: "You are a friendly assistant",
  });
  const run = await agent.run([{
    type: "input_text",
    content: "Hello!",
  }]);
  assertEquals(run.history, [{
    type: "output_text",
    content: "Hey! How are you doing?",
  }]);
});

Deno.test("Structured output works", async () => {
  const agent = new Agent({
    model: "__testing:deterministic",
    instructions:
      "You are a friendly assistant who can spit out a temperature guesstimate",
    output: z.number(),
  });
  const run = await agent.run([{
    type: "input_text",
    content: "Can you give me a temperature estimate?",
  }]);
  run.output satisfies number;
  assert(typeof run.output === "number");
});

Deno.test("Structured output 2 works", async () => {
  const agent = new Agent({
    model: "__testing:deterministic",
    instructions: "You are a friendly assistant who can name cats",
    output: z.object({
      name: z.string().describe("The cat's name"),
    }),
  });
  const run = await agent.run([{
    type: "input_text",
    content: "Can you give me a cat name?",
  }]);
  assertEquals(run.output.name, "Bingus");
});

Deno.test("Tool calls can work", async () => {
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
    model: "__testing:deterministic",
    instructions:
      "You are a friendly assistant who can spit out a temperature guesstimate",
    tools: [search],
  });
  const run = await agent.run("Can you tell me what cat websites there are?");
  assert(
    run.outputText.includes("bingus.com"),
  );
  assertEquals(run.history.length, 3);
});
