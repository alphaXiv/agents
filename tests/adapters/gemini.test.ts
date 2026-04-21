import { ThinkingLevel as GenAiThinkingLevel } from "@google/genai";
import { assertEquals } from "@std/assert";
import { GeminiModel } from "../../mod.ts";
import { GeminiAdapter } from "../../src/adapters/gemini/adapter.ts";
import {
  createToolFixtures,
  INTEGRATION_TIMEOUT_MS,
  runAdapterToolStreamingTest,
  runAgentToolStreamingTest,
  runBackAndForthCalculatorConversationTest,
  runStructuredOutputStreamingTest,
  runStructuredToolParameterStreamingTest,
  runToolLessHandoffResumeTest,
} from "./shared.ts";

const HAS_GEMINI_KEY = Boolean(Deno.env.get("GEMINI_API_KEY"));

Deno.test({
  name: "GeminiAdapter streams a parameterized tool call (gemini-3.1-pro-preview, thinking-level)",
  ignore: !HAS_GEMINI_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    const fixtures = createToolFixtures();
    const adapter = new GeminiAdapter({
      model: "gemini-3.1-pro-preview",
      thinkingConfig: { includeThoughts: true, thinkingLevel: GenAiThinkingLevel.LOW },
    });

    await runAdapterToolStreamingTest(t, {
      stream: adapter.stream({
        history: [{
          type: "input_text",
          content: `Call ${fixtures.echoTool.normalizedName} exactly once with {\"query\":\"${fixtures.query}\"}.`,
        }],
        instructions: "You are a compliant live integration test assistant.",
        tools: [fixtures.echoTool],
        signal: AbortSignal.timeout(INTEGRATION_TIMEOUT_MS),
      }),
      toolName: fixtures.echoTool.normalizedName,
      expectedContentSubstring: fixtures.query,
    });
  },
});

Deno.test({
  name: "GeminiAdapter streams a parameterized tool call (gemini-3.1-flash-lite-preview, thinking-level)",
  ignore: !HAS_GEMINI_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    const fixtures = createToolFixtures();
    const adapter = new GeminiAdapter({
      model: "gemini-3.1-flash-lite-preview",
      thinkingConfig: { includeThoughts: true, thinkingLevel: GenAiThinkingLevel.LOW },
    });

    await runAdapterToolStreamingTest(t, {
      stream: adapter.stream({
        history: [{
          type: "input_text",
          content: `Call ${fixtures.echoTool.normalizedName} exactly once with {\"query\":\"${fixtures.query}\"}.`,
        }],
        instructions: "You are a compliant live integration test assistant.",
        tools: [fixtures.echoTool],
        signal: AbortSignal.timeout(INTEGRATION_TIMEOUT_MS),
      }),
      toolName: fixtures.echoTool.normalizedName,
      expectedContentSubstring: fixtures.query,
    });
  },
});

Deno.test({
  name: "GeminiAdapter streams a parameterized tool call (gemini-3-flash-preview, thinking-level)",
  ignore: !HAS_GEMINI_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    const fixtures = createToolFixtures();
    const adapter = new GeminiAdapter({
      model: "gemini-3-flash-preview",
      thinkingConfig: { includeThoughts: true, thinkingLevel: GenAiThinkingLevel.LOW },
    });

    await runAdapterToolStreamingTest(t, {
      stream: adapter.stream({
        history: [{
          type: "input_text",
          content: `Call ${fixtures.echoTool.normalizedName} exactly once with {\"query\":\"${fixtures.query}\"}.`,
        }],
        instructions: "You are a compliant live integration test assistant.",
        tools: [fixtures.echoTool],
        signal: AbortSignal.timeout(INTEGRATION_TIMEOUT_MS),
      }),
      toolName: fixtures.echoTool.normalizedName,
      expectedContentSubstring: fixtures.query,
    });
  },
});

Deno.test({
  name: "GeminiAdapter streams a parameterized tool call (gemini-2.5-pro, legacy thinking-budget)",
  ignore: !HAS_GEMINI_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    const fixtures = createToolFixtures();
    const adapter = new GeminiAdapter({
      model: "gemini-2.5-pro",
      thinkingConfig: { includeThoughts: true, thinkingBudget: 1024 },
    });

    await runAdapterToolStreamingTest(t, {
      stream: adapter.stream({
        history: [{
          type: "input_text",
          content: `Call ${fixtures.echoTool.normalizedName} exactly once with {\"query\":\"${fixtures.query}\"}.`,
        }],
        instructions: "You are a compliant live integration test assistant.",
        tools: [fixtures.echoTool],
        signal: AbortSignal.timeout(INTEGRATION_TIMEOUT_MS),
      }),
      toolName: fixtures.echoTool.normalizedName,
      expectedContentSubstring: fixtures.query,
    });
  },
});

Deno.test({
  name: "GeminiAdapter streams a parameterized tool call (gemini-2.0-flash-lite, no thinking)",
  ignore: !HAS_GEMINI_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    const fixtures = createToolFixtures();
    const adapter = new GeminiAdapter({
      model: "gemini-2.0-flash-lite",
      thinkingConfig: { includeThoughts: false, thinkingBudget: 0 } as never,
    });

    await runAdapterToolStreamingTest(t, {
      stream: adapter.stream({
        history: [{
          type: "input_text",
          content: `Call ${fixtures.echoTool.normalizedName} exactly once with {\"query\":\"${fixtures.query}\"}.`,
        }],
        instructions: "You are a compliant live integration test assistant.",
        tools: [fixtures.echoTool],
        signal: AbortSignal.timeout(INTEGRATION_TIMEOUT_MS),
      }),
      toolName: fixtures.echoTool.normalizedName,
      expectedContentSubstring: fixtures.query,
    });
  },
});

Deno.test({
  name: "GeminiModel streams tools and results (gemini-3.1-pro-preview, thinking-level)",
  ignore: !HAS_GEMINI_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runAgentToolStreamingTest(t, {
      model: new GeminiModel({ model: "gemini-3.1-pro-preview", thinkingLevel: "low" }),
    });
  },
});

Deno.test({
  name: "GeminiModel streams tools and results (gemini-3.1-flash-lite-preview, thinking-level)",
  ignore: !HAS_GEMINI_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runAgentToolStreamingTest(t, {
      model: new GeminiModel({ model: "gemini-3.1-flash-lite-preview", thinkingLevel: "low" }),
    });
  },
});

Deno.test({
  name: "GeminiModel streams tools and results (gemini-3-flash-preview, thinking-level)",
  ignore: !HAS_GEMINI_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runAgentToolStreamingTest(t, {
      model: new GeminiModel({ model: "gemini-3-flash-preview", thinkingLevel: "low" }),
    });
  },
});

Deno.test({
  name: "GeminiModel streams tools and results (gemini-2.5-flash, legacy thinking-budget)",
  ignore: !HAS_GEMINI_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runAgentToolStreamingTest(t, {
      model: new GeminiModel({ model: "gemini-2.5-flash", thinkingLevel: "low" }),
    });
  },
});

Deno.test({
  name: "GeminiModel streams tools and results (gemini-2.0-flash-lite, no thinking)",
  ignore: !HAS_GEMINI_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runAgentToolStreamingTest(t, {
      model: new GeminiModel({ model: "gemini-2.0-flash-lite" }),
    });
  },
});

Deno.test({
  name: "GeminiModel executes structured tool parameters (gemini-2.5-flash, legacy thinking-budget)",
  ignore: !HAS_GEMINI_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runStructuredToolParameterStreamingTest(t, {
      model: new GeminiModel({ model: "gemini-2.5-flash", thinkingLevel: "low" }),
    });
  },
});

Deno.test({
  name: "GeminiModel streams structured output (gemini-3.1-pro-preview, thinking-level)",
  ignore: !HAS_GEMINI_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runStructuredOutputStreamingTest(t, {
      model: new GeminiModel({ model: "gemini-3.1-pro-preview", thinkingLevel: "low" }),
    });
  },
});

Deno.test({
  name: "GeminiModel streams structured output (gemini-3.1-flash-lite-preview, thinking-level)",
  ignore: !HAS_GEMINI_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runStructuredOutputStreamingTest(t, {
      model: new GeminiModel({ model: "gemini-3.1-flash-lite-preview", thinkingLevel: "low" }),
    });
  },
});

Deno.test({
  name: "GeminiModel streams structured output (gemini-3-flash-preview, thinking-level)",
  ignore: !HAS_GEMINI_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runStructuredOutputStreamingTest(t, {
      model: new GeminiModel({ model: "gemini-3-flash-preview", thinkingLevel: "low" }),
    });
  },
});

Deno.test({
  name: "GeminiModel streams structured output (gemini-2.5-pro, legacy thinking-budget)",
  ignore: !HAS_GEMINI_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runStructuredOutputStreamingTest(t, {
      model: new GeminiModel({ model: "gemini-2.5-pro", thinkingLevel: "low" }),
    });
  },
});

Deno.test({
  name: "GeminiModel keeps a 5-turn calculator conversation (gemini-3.1-flash-lite-preview)",
  ignore: !HAS_GEMINI_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runBackAndForthCalculatorConversationTest(t, {
      model: new GeminiModel({ model: "gemini-3.1-flash-lite-preview", thinkingLevel: "low" }),
    });
  },
});

Deno.test({
  name: "GeminiModel resumes a handoff without tools (gemini-3.1-flash-lite-preview)",
  ignore: !HAS_GEMINI_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runToolLessHandoffResumeTest(t, {
      model: new GeminiModel({ model: "gemini-3.1-flash-lite-preview", thinkingLevel: "low" }),
    });
  },
});

Deno.test("GeminiAdapter replays tool calls and results without current tool definitions", async () => {
  const adapter = new GeminiAdapter({
    model: "gemini-3.1-flash-lite-preview",
    apiKey: "test-key",
    thinkingConfig: { includeThoughts: false, thinkingBudget: 0 } as never,
  });

  const history = await adapter.getHistory(
    [
      { type: "tool_use", tool_use_id: "call_1", kind: "Load Project Snapshot", content: undefined },
      { type: "tool_result_text", tool_use_id: "call_1", content: "ready" },
    ],
    [],
    AbortSignal.abort(),
  );

  assertEquals(history, [
    {
      role: "model",
      parts: [{
        functionCall: {
          id: "call_1",
          name: "load_project_snapshot",
          args: undefined,
        },
        thoughtSignature: "context_engineering_is_the_way_to_go",
      }],
    },
    {
      role: "user",
      parts: [{
        functionResponse: {
          id: "call_1",
          name: "load_project_snapshot",
          response: { content: "ready" },
        },
      }],
    },
  ]);
});
