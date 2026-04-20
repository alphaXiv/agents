import { assert, assertEquals, assertRejects } from "@std/assert";
import { assertObjectMatch } from "@std/assert/object-match";
import { delay } from "@std/async/delay";
import z from "zod";
import { Agent, type ChatItem, type StreamItem, Tool } from "../../mod.ts";
import {
  ContextWindowTestModel,
  DeterministicTestModel,
  FailingTestModel,
  HistoryRecordingTestModel,
  testingTracker,
} from "./testing-model.ts";

Deno.test("Basic input out of agents works", async () => {
  const agent = new Agent({
    model: new DeterministicTestModel(),
    instructions: "You are a friendly assistant",
  });
  const run = await agent.run("Hello!");
  assertEquals(run.history.length, 1);
  assertObjectMatch(run.history[0], {
    type: "output_text",
    content: "Hey! How are you doing?",
  });
});

Deno.test("History input out of agents works", async () => {
  const agent = new Agent({
    model: new DeterministicTestModel(),
    instructions: "You are a friendly assistant",
  });
  const run = await agent.run([{
    type: "input_text",
    content: "Hello!",
  }]);
  assertEquals(run.history.length, 1);
  assertObjectMatch(run.history[0], {
    type: "output_text",
    content: "Hey! How are you doing?",
  });
});

Deno.test("Structured output works", async () => {
  const agent = new Agent({
    model: new DeterministicTestModel(),
    instructions: "You are a friendly assistant who can spit out a temperature guesstimate",
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
    model: new DeterministicTestModel(),
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
    execute: (param) => {
      if (param === "cats") {
        return JSON.stringify(["bingus.com", "bungus.com"]);
      }
      return "wtfrick.com";
    },
  });

  const agent = new Agent({
    model: new DeterministicTestModel(),
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
    model: new DeterministicTestModel(),
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
    execute: (param) => {
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
    model: new DeterministicTestModel(),
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
    execute: async (param) => {
      await delay(250);
      if (param === "cats") {
        return JSON.stringify(["bingus.com", "bungus.com"]);
      }
      return "wtfrick.com";
    },
  });

  const agent = new Agent({
    model: new DeterministicTestModel(),
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

Deno.test("Tool timeout can work", async () => {
  let sawSignal = false;

  const search = new Tool({
    name: "Searching the internet...",
    description: "Use when you want to search the internet",
    parameters: z.string().describe("Query parameter"),
    timeout: 100,
    execute: async (param, { signal }) => {
      await new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(resolve, 250);
        const onAbort = () => {
          sawSignal = true;
          clearTimeout(timeout);
          reject(signal.reason);
        };

        if (signal.aborted) {
          onAbort();
          return;
        }

        signal.addEventListener("abort", onAbort, { once: true });
      });

      if (param === "cats") {
        return JSON.stringify(["bingus.com", "bungus.com"]);
      }
      return "wtfrick.com";
    },
  });

  const agent = new Agent({
    model: new DeterministicTestModel(),
    instructions: "You are a friendly assistant.",
    tools: [search],
  });

  const run = await agent.run("Can you tell me what cat websites there are?");
  assert(sawSignal);
  assert(
    run.outputText.includes("Error:"),
  );
});

Deno.test("Tool signal can work", async () => {
  let sawSignal = false;

  const search = new Tool({
    name: "Searching the internet...",
    description: "Use when you want to search the internet",
    parameters: z.string().describe("Query parameter"),
    signal: AbortSignal.timeout(100),
    execute: async (param, { signal }) => {
      await new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(resolve, 250);
        const onAbort = () => {
          sawSignal = true;
          clearTimeout(timeout);
          reject(signal.reason);
        };

        if (signal.aborted) {
          onAbort();
          return;
        }

        signal.addEventListener("abort", onAbort, { once: true });
      });

      if (param === "cats") {
        return JSON.stringify(["bingus.com", "bungus.com"]);
      }
      return "wtfrick.com";
    },
  });

  const agent = new Agent({
    model: new DeterministicTestModel(),
    instructions: "You are a friendly assistant.",
    tools: [search],
  });

  const run = await agent.run("Can you tell me what cat websites there are?");
  assert(sawSignal);
  assert(
    run.outputText.includes("Error:"),
  );
});

Deno.test("Agent LLM retries 3 times by default", async () => {
  const search = new Tool({
    name: "Searching the internet...",
    description: "Use when you want to search the internet",
    parameters: z.string().describe("Query parameter"),
    execute: () => {
      return "throw";
    },
  });

  const agent = new Agent({
    model: new FailingTestModel(),
    instructions: "You are a friendly assistant.",
    tools: [search],
  });

  const counter = { failures: 0 };
  testingTracker.enterWith(counter);

  await assertRejects(
    () => agent.run("Can you tell me what cat websites there are?"),
    Error,
    "Deterministic Provider Error",
  );
  assertEquals(counter.failures, 3);
});

Deno.test("Disable retrying of an agent", async () => {
  const search = new Tool({
    name: "Searching the internet...",
    description: "Use when you want to search the internet",
    parameters: z.string().describe("Query parameter"),
    execute: () => {
      return "throw";
    },
  });

  const agent = new Agent({
    model: new FailingTestModel(),
    instructions: "You are a friendly assistant.",
    tools: [search],
    retryStrategy: { modelCycles: 1, sameModelRetries: 0 },
  });

  const counter = { failures: 0 };
  testingTracker.enterWith(counter);

  await assertRejects(
    () => agent.run("Can you tell me what cat websites there are?"),
    Error,
    "Deterministic Provider Error",
  );
  assertEquals(counter.failures, 1);
});

Deno.test("handleModelError recovers from context window errors", async () => {
  let recoveryCalls = 0;

  const agent = new Agent({
    model: new ContextWindowTestModel(3),
    instructions: "You are a friendly assistant",
    async *handleModelError(error, history, context) {
      assertEquals(error.kind, "context_overflow");
      assertEquals(context.turn, 0);
      assertEquals(context.attempt, 0);
      assertEquals(context.model.provider, "deterministic");
      assertEquals(context.model.model, "deterministic");

      if (error.original instanceof Error && error.original.message.includes("context length")) {
        yield { type: "context_summary_start" as const };
        recoveryCalls += 1;
        return history.filter((item) => item.type !== "input_file");
      }
      return null;
    },
  });

  const run = await agent.run([
    { type: "input_text", content: "Hello!" },
    { type: "input_file", kind: "image/png", content: "aaa" },
    { type: "input_file", kind: "image/png", content: "bbb" },
    { type: "input_file", kind: "image/png", content: "ccc" },
    { type: "input_file", kind: "image/png", content: "ddd" },
  ]);

  assert(recoveryCalls > 0);
  assertEquals(run.outputText, "Recovery successful!");
});

Deno.test("beforeModelCall receives current model in context", async () => {
  let receivedContext:
    | { turn: number; reason: string; model: { provider: string; model: string } }
    | undefined;

  const agent = new Agent({
    model: new DeterministicTestModel(),
    instructions: "Test agent",
    // deno-lint-ignore require-yield
    async *beforeModelCall(history, context) {
      receivedContext = {
        turn: context.turn,
        reason: context.reason,
        model: context.model,
      };
      return history;
    },
  });

  await agent.run("Hello");

  assert(receivedContext);
  assertEquals(receivedContext.turn, 0);
  assertEquals(receivedContext.reason, "init");
  assertEquals(receivedContext.model.provider, "deterministic");
  assertEquals(receivedContext.model.model, "deterministic");
});

Deno.test("handleModelError returning null falls through to normal retry", async () => {
  let handleCalls = 0;

  const agent = new Agent({
    model: new FailingTestModel(),
    instructions: "You are a friendly assistant.",
    // deno-lint-ignore require-yield
    async *handleModelError(_error, _history, _context) {
      handleCalls += 1;
      return null;
    },
  });

  await assertRejects(
    () => agent.run("Hello!"),
    Error,
    "Deterministic Provider Error",
  );
  assertEquals(handleCalls, 3);
});

Deno.test("model receives history filtered from last context_summary", async () => {
  const model = new HistoryRecordingTestModel("response");

  // Simulate a history with a context_summary in the middle
  const historyWithSummary: ChatItem[] = [
    { type: "input_text", content: "old message 1" },
    { type: "output_text", content: "old response 1" },
    { type: "context_summary", content: "Summary of old conversation" },
    { type: "input_text", content: "recent message" },
    { type: "output_text", content: "recent response" },
  ];

  const agent = new Agent({
    model,
    instructions: "Test agent",
  });

  await agent.run(historyWithSummary);

  // Model should only see from the last context_summary onwards
  assertEquals(model.receivedHistories.length, 1);
  const receivedHistory = model.receivedHistories[0];

  assertEquals(receivedHistory.length, 3);
  assertEquals(receivedHistory[0].type, "context_summary");
  assertEquals(receivedHistory[0].content, "Summary of old conversation");
  assertEquals(receivedHistory[1].type, "input_text");
  assertEquals(receivedHistory[1].content, "recent message");
  assertEquals(receivedHistory[2].type, "output_text");
  assertEquals(receivedHistory[2].content, "recent response");
});

Deno.test("model receives history filtered from LATEST context_summary when multiple exist", async () => {
  const model = new HistoryRecordingTestModel("response");

  // History with multiple context_summaries:
  // [conversationA, compactionA, conversationB, compactionB, conversationC]
  const historyWithMultipleSummaries: ChatItem[] = [
    { type: "input_text", content: "conversationA message" },
    { type: "output_text", content: "conversationA response" },
    { type: "context_summary", content: "compactionA - summary of A" },
    { type: "input_text", content: "conversationB message" },
    { type: "output_text", content: "conversationB response" },
    { type: "context_summary", content: "compactionB - summary of A+B" },
    { type: "input_text", content: "conversationC message" },
  ];

  const agent = new Agent({
    model,
    instructions: "Test agent",
  });

  await agent.run(historyWithMultipleSummaries);

  // Model should only see [compactionB, conversationC]
  assertEquals(model.receivedHistories.length, 1);
  const receivedHistory = model.receivedHistories[0];

  assertEquals(receivedHistory.length, 2);
  assertEquals(receivedHistory[0].type, "context_summary");
  assertEquals(receivedHistory[0].content, "compactionB - summary of A+B");
  assertEquals(receivedHistory[1].type, "input_text");
  assertEquals(receivedHistory[1].content, "conversationC message");
});

Deno.test("beforeModelCall compaction items appear before model output in history", async () => {
  const model = new HistoryRecordingTestModel("model response");

  const agent = new Agent({
    model,
    instructions: "Test agent",
    async *beforeModelCall(history, _context) {
      // Always add a compaction summary at the start
      yield { type: "context_summary_start" };
      return [
        { type: "context_summary", content: "Summary of prior context" },
        ...history,
      ];
    },
  });

  const run = await agent.run("Hello");

  // History should have context_summary BEFORE the model's output
  const summaryIndex = run.history.findIndex((item) => item.type === "context_summary");
  const outputIndex = run.history.findIndex((item) => item.type === "output_text");

  assert(summaryIndex !== -1, "Should have a context_summary in history");
  assert(outputIndex !== -1, "Should have an output_text in history");
  assert(summaryIndex < outputIndex, "context_summary should appear before output_text");
});

Deno.test("compaction items are only added after successful model call", async () => {
  let beforeModelCallCount = 0;

  const agent = new Agent({
    model: new FailingTestModel(),
    instructions: "Test agent",
    retryStrategy: { modelCycles: 2, sameModelRetries: 0 },
    async *beforeModelCall(history, _context) {
      beforeModelCallCount++;
      yield { type: "context_summary_start" };
      return [
        { type: "context_summary", content: `Summary ${beforeModelCallCount}` },
        ...history,
      ];
    },
  });

  // The agent will fail, but beforeModelCall will be called multiple times
  await assertRejects(() => agent.run("Hello"), Error, "Deterministic Provider Error");

  // beforeModelCall was called multiple times due to retries
  assert(beforeModelCallCount >= 2, "beforeModelCall should be called on retries");
});

Deno.test("model_switched event is emitted when falling back to another model", async () => {
  const agent = new Agent({
    model: [new FailingTestModel(), new DeterministicTestModel()],
    instructions: "You are a friendly assistant",
    retryStrategy: { modelCycles: 1, sameModelRetries: 0 },
  });

  const streamItems: StreamItem[] = [];
  const stream = agent.stream("Hello!");

  for await (const item of stream) {
    streamItems.push(item);
  }

  const switchEvent = streamItems.find((item) => item.type === "model_switched");
  assert(switchEvent, "Should emit a model_switched event");
  assert(switchEvent.type === "model_switched");
  assertEquals(switchEvent.from.provider, "deterministic-failing");
  assertEquals(switchEvent.from.model, "deterministic-failing");
  assertEquals(switchEvent.to.provider, "deterministic");
  assertEquals(switchEvent.to.model, "deterministic");
  assert(switchEvent.cause instanceof Error, "cause should be the original error");
  assertEquals((switchEvent.cause as Error).message, "Deterministic Provider Error");
});
