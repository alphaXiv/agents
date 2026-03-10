// deno-lint-ignore-file require-await
import z from "zod";
import { assert, assertEquals, assertExists } from "@std/assert";
import { Agent, Tool } from "../mod.ts";
import type { Adapter } from "../src/adapters.ts";
import { convertChatItemsToStream } from "../src/client.ts";
import {
  type PartialTraceEvent,
  registerGlobalTracer,
  type TraceEvent,
  type Tracer,
} from "../src/tracing.ts";

function createRecorder() {
  const starts: PartialTraceEvent[] = [];
  const events: TraceEvent[] = [];
  const tracer: Tracer = {
    start: (event) => starts.push(event),
    event: (event) => events.push(event),
  };
  return { starts, events, tracer };
}

function filterTrace<T extends TraceEvent["type"]>(
  events: TraceEvent[],
  type: T,
): Extract<TraceEvent, { type: T }>[] {
  return events.filter((event): event is Extract<TraceEvent, { type: T }> =>
    event.type === type
  );
}

function traceSnapshot(events: TraceEvent[]) {
  const indexById = new Map(events.map((event, index) => [event.id, index]));
  return events.map((event) => ({
    type: event.type as string,
    parent: event.parent == null ? null : indexById.get(event.parent),
    error: event.errorMessage,
    content: event.content,
  }));
}

function assertStartsMatchEvents(
  starts: PartialTraceEvent[],
  events: TraceEvent[],
) {
  const nonLogEvents = events.filter((event) => event.type !== "log");
  assertEquals(
    starts.map((event) => event.id).sort(),
    nonLogEvents.map((event) => event.id).sort(),
  );
}

Deno.test("global tracer captures tool turns, message spans, and token counts", async () => {
  const { starts, events, tracer } = createRecorder();
  const unregister = registerGlobalTracer(tracer);

  try {
    const search = new Tool({
      name: "search",
      description: "Searches for things",
      parameters: z.string(),
      execute: ({ param }) => `results for ${param}`,
    });

    const adapter: Adapter<"tool-model"> = {
      name: "trace-test",
      async stream({ history, tools }) {
        const last = history.at(-1);
        if (!last || last.type === "input_text") {
          const tool = tools[0];
          assertExists(tool);
          return convertChatItemsToStream({
            items: [{
              type: "tool_use",
              tool_use_id: "tool-1",
              kind: tool.name,
              content: JSON.stringify("cats"),
            }],
            inputTokens: 3,
            outputTokens: 5,
          });
        }

        if (last.type !== "tool_result_text") {
          throw new Error(`Unexpected history item: ${last.type}`);
        }
        return convertChatItemsToStream({
          items: [{
            type: "output_text",
            content: `done: ${last.content}`,
          }],
          inputTokens: 7,
          outputTokens: 11,
        });
      },
    };

    const agent = new Agent({
      adapter,
      model: "tool-model",
      instructions: "Use the search tool.",
      tools: [search],
    });

    const run = await agent.run("find cats");
    assertEquals(run.outputText, "done: results for cats");
    assertEquals(run.inputTokens, 10);
    assertEquals(run.outputTokens, 16);
    assertStartsMatchEvents(starts, events);

    const agentTrace = filterTrace(events, "agent")[0];
    const toolTrace = filterTrace(events, "tool")[0];
    const messageTrace = filterTrace(events, "message")[0];
    const modelTraces = filterTrace(events, "model");

    assertExists(agentTrace);
    assertExists(toolTrace);
    assertExists(messageTrace);
    assertEquals(modelTraces.length, 2);

    assertEquals(agentTrace.content, {
      provider: "trace-test",
      model: "tool-model",
    });
    assertEquals(toolTrace.parent, agentTrace.id);
    assertEquals(toolTrace.content, { name: "search" });
    assertEquals(messageTrace.parent, modelTraces[1].id);
    assertEquals(messageTrace.content, { type: "output_text" });
    assert(modelTraces.every((trace) => trace.parent === agentTrace.id));
    assertEquals(
      modelTraces.map((trace) => trace.content.reason),
      ["init", "tool"],
    );
    assertEquals(
      modelTraces.map((trace) => [
        trace.content.inputTokens,
        trace.content.outputTokens,
      ]),
      [[3, 5], [7, 11]],
    );

    assertEquals(run.history, [
      {
        type: "tool_use",
        tool_use_id: "tool-1",
        kind: "search",
        content: JSON.stringify("cats"),
        trace: toolTrace.id,
      },
      {
        type: "tool_result_text",
        tool_use_id: "tool-1",
        content: "results for cats",
        trace: toolTrace.id,
      },
      {
        type: "output_text",
        content: "done: results for cats",
        trace: messageTrace.id,
      },
    ]);
  } finally {
    unregister();
  }
});

Deno.test("local tracer captures sub-agent spans and tags history items with the correct traces", async () => {
  const { starts, events, tracer } = createRecorder();

  const subAdapter: Adapter<"inner-model"> = {
    name: "inner",
    async stream() {
      return convertChatItemsToStream({
        items: [{ type: "output_text", content: "subagent result" }],
        inputTokens: 1,
        outputTokens: 2,
      });
    },
  };

  const outerAdapter: Adapter<"outer-model"> = {
    name: "outer",
    async stream({ history, tools }) {
      const last = history.at(-1);
      if (!last || last.type === "input_text") {
        const tool = tools[0];
        assertExists(tool);
        return convertChatItemsToStream({
          items: [{
            type: "tool_use",
            tool_use_id: "tool-1",
            kind: tool.name,
          }],
          inputTokens: 2,
          outputTokens: 3,
        });
      }

      if (last.type !== "tool_result_text") {
        throw new Error(`Unexpected history item: ${last.type}`);
      }
      return convertChatItemsToStream({
        items: [{ type: "output_text", content: last.content }],
        inputTokens: 4,
        outputTokens: 5,
      });
    },
  };

  const subagent = new Agent({
    adapter: subAdapter,
    model: "inner-model",
    instructions: "Answer plainly.",
  });

  const useSubagent = new Tool({
    name: "delegate",
    description: "Runs a subagent",
    parameters: z.void(),
    execute: async () => {
      const run = await subagent.run("help");
      return run.outputText;
    },
  });

  const agent = new Agent({
    adapter: outerAdapter,
    model: "outer-model",
    instructions: "Use the delegate tool.",
    tools: [useSubagent],
    tracers: [tracer],
  });

  const run = await agent.run("go");
  assertEquals(run.outputText, "subagent result");
  assertEquals(run.inputTokens, 6);
  assertEquals(run.outputTokens, 8);
  assertStartsMatchEvents(starts, events);

  assertEquals(traceSnapshot(events), [
    {
      type: "model",
      parent: 7,
      error: null,
      content: {
        reason: "init",
        provider: "outer",
        model: "outer-model",
        inputTokens: 2,
        outputTokens: 3,
      },
    },
    {
      type: "message",
      parent: 2,
      error: null,
      content: { type: "output_text" },
    },
    {
      type: "model",
      parent: 3,
      error: null,
      content: {
        reason: "init",
        provider: "inner",
        model: "inner-model",
        inputTokens: 1,
        outputTokens: 2,
      },
    },
    {
      type: "agent",
      parent: 4,
      error: null,
      content: {
        provider: "inner",
        model: "inner-model",
      },
    },
    {
      type: "tool",
      parent: 7,
      error: null,
      content: { name: "delegate" },
    },
    {
      type: "message",
      parent: 6,
      error: null,
      content: { type: "output_text" },
    },
    {
      type: "model",
      parent: 7,
      error: null,
      content: {
        reason: "tool",
        provider: "outer",
        model: "outer-model",
        inputTokens: 4,
        outputTokens: 5,
      },
    },
    {
      type: "agent",
      parent: null,
      error: null,
      content: {
        provider: "outer",
        model: "outer-model",
      },
    },
  ]);

  const toolTrace = filterTrace(events, "tool")[0];
  const innerAgentTrace = filterTrace(events, "agent").find((trace) =>
    trace.content.model === "inner-model"
  );
  const outerOutputMessage = filterTrace(events, "message").find((trace) =>
    trace.parent ===
      filterTrace(events, "model").find((event) =>
        event.content.reason === "tool"
      )?.id
  );
  const initialOuterModelTrace = filterTrace(events, "model").find((trace) =>
    trace.content.model === "outer-model" && trace.content.reason === "init"
  );

  assertExists(toolTrace);
  assertExists(innerAgentTrace);
  assertExists(outerOutputMessage);
  assertExists(initialOuterModelTrace);
  assertEquals(innerAgentTrace.parent, toolTrace.id);

  assertEquals(run.history, [
    {
      type: "tool_use",
      tool_use_id: "tool-1",
      kind: "delegate",
      content: undefined,
      trace: toolTrace.id,
    },
    {
      type: "tool_result_text",
      tool_use_id: "tool-1",
      content: "subagent result",
      trace: toolTrace.id,
    },
    {
      type: "output_text",
      content: "subagent result",
      trace: outerOutputMessage.id,
    },
  ]);

  const traceById = new Map(events.map((event) => [event.id, event]));
  for (const item of run.history) {
    const trace = traceById.get(item.trace);
    assertExists(trace, `missing trace event for history item ${item.type}`);

    if (item.type === "tool_use" || item.type.startsWith("tool_result")) {
      assertEquals(trace.type, "tool");
      assertEquals(trace.content, { name: "delegate" });
    } else {
      assertEquals(trace.type, "message");
      assertEquals(trace.content, { type: "output_text" });
    }
  }

  assert(
    run.history.every((item) => item.trace !== initialOuterModelTrace.id),
    "history items from the tool turn should not point at the initial model trace",
  );
});

Deno.test("unstable tool-use hints create message spans without changing tool trace history tags", async () => {
  const { starts, events, tracer } = createRecorder();

  const ping = new Tool({
    name: "ping",
    description: "Returns pong",
    parameters: z.void(),
    execute: () => "pong",
  });

  const adapter: Adapter<"hint-model"> = {
    name: "hint",
    async *stream({ history, tools }) {
      const last = history.at(-1);
      if (!last || last.type === "input_text") {
        const tool = tools[0];
        assertExists(tool);
        yield { type: "unstable_tracing_tool_use_start" };
        yield {
          type: "tool_use",
          index: 0,
          tool_use_id: "tool-1",
          kind: tool.name,
        };
        return { inputTokens: 1, outputTokens: 2 };
      }

      if (last.type !== "tool_result_text") {
        throw new Error(`Unexpected history item: ${last.type}`);
      }
      yield {
        type: "delta_output_text",
        index: 0,
        delta: last.content,
      };
      return { inputTokens: 3, outputTokens: 4 };
    },
  };

  const agent = new Agent({
    adapter,
    model: "hint-model",
    instructions: "Use the tool.",
    tools: [ping],
    tracers: [tracer],
  });

  const run = await agent.run("go");
  assertEquals(run.outputText, "pong");
  assertStartsMatchEvents(starts, events);

  const initialModel = filterTrace(events, "model").find((trace) =>
    trace.content.reason === "init"
  );
  const toolTrace = filterTrace(events, "tool")[0];
  const toolUseMessage = filterTrace(events, "message").find((trace) =>
    trace.content.type === "tool_use"
  );

  assertExists(initialModel);
  assertExists(toolTrace);
  assertExists(toolUseMessage);
  assertEquals(toolUseMessage.parent, initialModel.id);
  assertEquals(run.history[0]?.trace, toolTrace.id);
  assertEquals(run.history[1]?.trace, toolTrace.id);
});

Deno.test("tool failures are traced as errors while the agent run still completes", async () => {
  const { starts, events, tracer } = createRecorder();

  const failingTool = new Tool({
    name: "explode",
    description: "Always fails",
    parameters: z.void(),
    execute: () => {
      throw new Error("tool blew up");
    },
  });

  const adapter: Adapter<"tool-failure-model"> = {
    name: "trace-test",
    async stream({ history, tools }) {
      const last = history.at(-1);
      if (!last || last.type === "input_text") {
        const tool = tools[0];
        assertExists(tool);
        return convertChatItemsToStream({
          items: [{
            type: "tool_use",
            tool_use_id: "tool-1",
            kind: tool.name,
          }],
          inputTokens: 1,
          outputTokens: 1,
        });
      }

      if (last.type !== "tool_result_text") {
        throw new Error(`Unexpected history item: ${last.type}`);
      }
      return convertChatItemsToStream({
        items: [{ type: "output_text", content: last.content }],
        inputTokens: 1,
        outputTokens: 1,
      });
    },
  };

  const agent = new Agent({
    adapter,
    model: "tool-failure-model",
    instructions: "Use the tool.",
    tools: [failingTool],
    tracers: [tracer],
  });

  const run = await agent.run("go");
  assertEquals(run.outputText, "Error: tool blew up");
  assertStartsMatchEvents(starts, events);

  const toolTrace = filterTrace(events, "tool")[0];
  const outputMessage = filterTrace(events, "message")[0];
  const agentTrace = filterTrace(events, "agent")[0];

  assertExists(toolTrace);
  assertExists(outputMessage);
  assertExists(agentTrace);
  assertEquals(toolTrace.errorMessage, "tool blew up");
  assertEquals(agentTrace.errorMessage, null);
  assertEquals(run.history[0]?.trace, toolTrace.id);
  assertEquals(run.history[1]?.trace, toolTrace.id);
  assertEquals(run.history[2]?.trace, outputMessage.id);
});

Deno.test("provider retries after partial output keep failed message spans", async () => {
  const { starts, events, tracer } = createRecorder();
  let calls = 0;

  const adapter: Adapter<"retry-model"> = {
    name: "retry",
    async *stream() {
      calls += 1;
      if (calls === 1) {
        yield {
          type: "delta_output_text",
          index: 0,
          delta: "partial",
        };
        throw new Error("provider failed");
      }

      yield {
        type: "delta_output_text",
        index: 0,
        delta: "worked",
      };
      return { inputTokens: 2, outputTokens: 3 };
    },
  };

  const agent = new Agent({
    adapter,
    model: "retry-model",
    instructions: "Answer plainly.",
    tracers: [tracer],
  });

  const run = await agent.run("hello");
  assertEquals(run.outputText, "partial\n\nworked");
  assertStartsMatchEvents(starts, events);

  const modelTraces = filterTrace(events, "model");
  const messageTraces = filterTrace(events, "message");
  const agentTrace = filterTrace(events, "agent")[0];

  assertEquals(modelTraces.length, 2);
  assertEquals(messageTraces.length, 2);
  assertEquals(messageTraces[0].errorMessage, "Cancelled by provider error");
  assertEquals(modelTraces[0].errorMessage, "provider failed");
  assertEquals(modelTraces[1].errorMessage, null);
  assertEquals(modelTraces[1].content.reason, "retry-provider-error");
  assertEquals(agentTrace.errorMessage, null);
  assertEquals(run.history[0]?.trace, messageTraces[0].id);
  assertEquals(run.history[1]?.trace, messageTraces[1].id);
});

Deno.test("malformed structured output emits a log span and retries cleanly", async () => {
  const { starts, events, tracer } = createRecorder();
  let calls = 0;

  const adapter: Adapter<"structured-model"> = {
    name: "structured",
    async stream() {
      calls += 1;
      return convertChatItemsToStream({
        items: [{
          type: "output_text",
          content: calls === 1
            ? JSON.stringify({ name: 123 })
            : JSON.stringify({ name: "bingus" }),
        }],
        inputTokens: 1,
        outputTokens: 1,
      });
    },
  };

  const agent = new Agent({
    adapter,
    model: "structured-model",
    instructions: "Return structured output.",
    output: z.object({
      name: z.string(),
    }),
    tracers: [tracer],
  });

  const run = await agent.run("name a cat");
  assertEquals(run.output, { name: "bingus" });

  const messageTraces = filterTrace(events, "message");
  const modelTraces = filterTrace(events, "model");
  const logTrace = filterTrace(events, "log")[0];
  const agentTrace = filterTrace(events, "agent")[0];

  assertExists(logTrace);
  assertExists(agentTrace);
  assertEquals(messageTraces.length, 2);
  assertEquals(modelTraces.length, 2);
  assertEquals(modelTraces[0].errorMessage, null);
  assertEquals(modelTraces[1].errorMessage, null);
  assertEquals(modelTraces[1].content.reason, "retry-malformed-output");
  assertEquals(logTrace.parent, agentTrace.id);
  assert(
    logTrace.content.includes(
      'Retrying due to failed structured output parse from "{\\"name\\":123}"',
    ),
  );
  assert(logTrace.errorMessage?.includes("Invalid input: expected string"));
  assert(!starts.some((start) => start.id === logTrace.id));

  assertEquals(run.history[0]?.trace, messageTraces[0].id);
  assertEquals(run.history[1]?.trace, agentTrace.id);
  assertEquals(run.history[2]?.trace, messageTraces[1].id);
});
