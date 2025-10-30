import z from "zod";
import { delay } from "@std/async/delay";
import { Agent } from "../src/agent.ts";
import { assert, assertEquals, assertRejects } from "@std/assert";
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
    instructions: "You are a friendly assistant.",
    tools: [search],
  });
  const run = await agent.run("Can you tell me what cat websites there are?");
  assert(
    run.outputText.includes("bingus.com"),
  );
  assertEquals(run.history.length, 3);
});

Deno.test("Dubious calls without retry will fail", async () => {
  const search = new Tool({
    name: "Searching the internet...",
    description: "Use when you want to search the internet",
    parameters: z.string().describe("Query parameter"),
    execute: () => {
      throw new Error("Oopsies network error");
    },
  });

  const agent = new Agent({
    model: "__testing:deterministic",
    instructions: "You are a friendly assistant.",
    tools: [search],
  });
  const run = await agent.run("Can you tell me what cat websites there are?");
  assert(
    run.outputText.includes("Error: Oopsies network error"),
  );
});

Deno.test("Dubious calls will work with retry", async () => {
  let shouldFail = true;
  const search = new Tool({
    name: "Searching the internet...",
    description: "Use when you want to search the internet",
    parameters: z.string().describe("Query parameter"),
    retries: 1,
    execute: ({ param }) => {
      if (shouldFail) {
        shouldFail = false;
        throw new Error("Oopsies network error");
      }
      if (param === "cats") {
        return JSON.stringify(["bingus.com", "bungus.com"]);
      }
      return "wtfrick.com";
    },
  });

  const agent = new Agent({
    model: "__testing:deterministic",
    instructions: "You are a friendly assistant.",
    tools: [search],
  });
  const run = await agent.run("Can you tell me what cat websites there are?");
  assert(
    run.outputText.includes("bingus.com"),
  );
  assertEquals(run.history.length, 3);
});

Deno.test("Abort signal can work", async () => {
  const search = new Tool({
    name: "Searching the internet...",
    description: "Use when you want to search the internet",
    parameters: z.string().describe("Query parameter"),
    execute: async ({ param }) => {
      await delay(250);
      if (param === "cats") {
        return JSON.stringify(["bingus.com", "bungus.com"]);
      }
      return "wtfrick.com";
    },
  });

  const agent = new Agent({
    model: "__testing:deterministic",
    instructions: "You are a friendly assistant.",
    tools: [search],
  });

  // Pre-aborted signal works
  const abortController1 = new AbortController();
  abortController1.abort();
  await assertRejects(
    () =>
      agent.run("Can you tell me what cat websites there are?", {
        signal: abortController1.signal,
      }),
    DOMException,
  );

  // Abort-signal during tool call works
  const abortController2 = new AbortController();
  setTimeout(() => abortController2.abort(), 100);
  await assertRejects(
    () =>
      agent.run("Can you tell me what cat websites there are?", {
        signal: abortController2.signal,
      }),
    DOMException,
  );

  // wait for all delays to clear up so we don't leak timers
  await delay(250);
});
