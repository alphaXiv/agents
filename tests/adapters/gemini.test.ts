import { ThinkingLevel as GenAiThinkingLevel } from "@google/genai";
import { assertEquals } from "@std/assert";
import { geminiModel } from "../../src/adapters/gemini/adapter.ts";
import { googleGenerateContentAPIModel } from "../../src/adapters/google_genai/adapter.ts";
import { getGoogleGenerateContentAPIHistory } from "../../src/adapters/google_genai/history.ts";
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
    const adapter = googleGenerateContentAPIModel({
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
  name: "GeminiAdapter streams a parameterized tool call (gemini-3.1-flash-lite, thinking-level)",
  ignore: !HAS_GEMINI_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    const fixtures = createToolFixtures();
    const adapter = googleGenerateContentAPIModel({
      model: "gemini-3.1-flash-lite",
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
    const adapter = googleGenerateContentAPIModel({
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
    const adapter = googleGenerateContentAPIModel({
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
    const adapter = googleGenerateContentAPIModel({
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
      model: geminiModel({ model: "gemini-3.1-pro-preview", thinkingLevel: "low" }),
    });
  },
});

Deno.test({
  name: "GeminiModel streams tools and results (gemini-3.1-flash-lite, thinking-level)",
  ignore: !HAS_GEMINI_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runAgentToolStreamingTest(t, {
      model: geminiModel({ model: "gemini-3.1-flash-lite", thinkingLevel: "low" }),
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
      model: geminiModel({ model: "gemini-3-flash-preview", thinkingLevel: "low" }),
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
      model: geminiModel({ model: "gemini-2.5-flash", thinkingLevel: "low" }),
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
      model: geminiModel({ model: "gemini-2.0-flash-lite" }),
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
      model: geminiModel({ model: "gemini-2.5-flash", thinkingLevel: "low" }),
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
      model: geminiModel({ model: "gemini-3.1-pro-preview", thinkingLevel: "low" }),
    });
  },
});

Deno.test({
  name: "GeminiModel streams structured output (gemini-3.1-flash-lite, thinking-level)",
  ignore: !HAS_GEMINI_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runStructuredOutputStreamingTest(t, {
      model: geminiModel({ model: "gemini-3.1-flash-lite", thinkingLevel: "low" }),
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
      model: geminiModel({ model: "gemini-3-flash-preview", thinkingLevel: "low" }),
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
      model: geminiModel({ model: "gemini-2.5-pro", thinkingLevel: "low" }),
    });
  },
});

Deno.test({
  name: "GeminiModel keeps a 5-turn calculator conversation (gemini-3.1-flash-lite)",
  ignore: !HAS_GEMINI_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runBackAndForthCalculatorConversationTest(t, {
      model: geminiModel({ model: "gemini-3.1-flash-lite", thinkingLevel: "low" }),
    });
  },
});

Deno.test({
  name: "GeminiModel resumes a handoff without tools (gemini-3.1-flash-lite)",
  ignore: !HAS_GEMINI_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runToolLessHandoffResumeTest(t, {
      model: geminiModel({ model: "gemini-3.1-flash-lite", thinkingLevel: "low" }),
    });
  },
});

Deno.test("GeminiAdapter replays tool calls and results without current tool definitions", async () => {
  const history = await getGoogleGenerateContentAPIHistory({
    history: [
      { type: "tool_use", tool_use_id: "call_1", kind: "Load Project Snapshot", content: undefined },
      { type: "tool_result_text", tool_use_id: "call_1", content: "ready" },
    ],
    toolMap: [],
    signal: AbortSignal.abort(),
  });

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

Deno.test("GeminiAdapter wraps primitive tool arguments when replaying without current tool definitions", async () => {
  const history = await getGoogleGenerateContentAPIHistory({
    history: [
      {
        type: "tool_use",
        tool_use_id: "call_1",
        kind: "Search Papers",
        content: JSON.stringify("DeepSeek-V3 arXiv link"),
      },
    ],
    toolMap: [],
    signal: AbortSignal.abort(),
  });

  assertEquals(history, [
    {
      role: "model",
      parts: [{
        functionCall: {
          id: "call_1",
          name: "search_papers",
          args: { content: "DeepSeek-V3 arXiv link" },
        },
        thoughtSignature: "context_engineering_is_the_way_to_go",
      }],
    },
  ]);
});

Deno.test("GeminiAdapter replays mixed tool history for a tool-less handoff", async () => {
  const history = await getGoogleGenerateContentAPIHistory({
    history: [
      {
        type: "tool_use",
        tool_use_id: "embed_1",
        kind: "Embedding Similarity Search",
        content: JSON.stringify({
          query: "Recent reasoning papers related to DeepSeek-R1.",
        }),
      },
      {
        type: "tool_result_text",
        tool_use_id: "embed_1",
        content: "1. Step 3.5 Flash",
      },
      {
        type: "tool_use",
        tool_use_id: "web_1",
        kind: "Search Web",
        content: JSON.stringify("DeepSeek-V3 arXiv link"),
      },
      {
        type: "tool_use",
        tool_use_id: "web_2",
        kind: "Search Web",
        content: JSON.stringify("Kimi k1.5 arXiv paper"),
      },
      {
        type: "tool_result_text",
        tool_use_id: "web_1",
        content: 'Error: Property accessor is not of type "number"',
      },
      {
        type: "tool_result_text",
        tool_use_id: "web_2",
        content: 'Error: Property accessor is not of type "number"',
      },
      {
        type: "input_text",
        content: "Summarize the prior history without using any tools.",
      },
    ],
    toolMap: [],
    signal: AbortSignal.abort(),
  });

  assertEquals(history, [
    {
      role: "model",
      parts: [{
        functionCall: {
          id: "embed_1",
          name: "embedding_similarity_search",
          args: { query: "Recent reasoning papers related to DeepSeek-R1." },
        },
        thoughtSignature: "context_engineering_is_the_way_to_go",
      }],
    },
    {
      role: "user",
      parts: [{
        functionResponse: {
          id: "embed_1",
          name: "embedding_similarity_search",
          response: { content: "1. Step 3.5 Flash" },
        },
      }],
    },
    {
      role: "model",
      parts: [{
        functionCall: {
          id: "web_1",
          name: "search_web",
          args: { content: "DeepSeek-V3 arXiv link" },
        },
        thoughtSignature: "context_engineering_is_the_way_to_go",
      }],
    },
    {
      role: "model",
      parts: [{
        functionCall: {
          id: "web_2",
          name: "search_web",
          args: { content: "Kimi k1.5 arXiv paper" },
        },
        thoughtSignature: "context_engineering_is_the_way_to_go",
      }],
    },
    {
      role: "user",
      parts: [{
        functionResponse: {
          id: "web_1",
          name: "search_web",
          response: { content: 'Error: Property accessor is not of type "number"' },
        },
      }],
    },
    {
      role: "user",
      parts: [{
        functionResponse: {
          id: "web_2",
          name: "search_web",
          response: { content: 'Error: Property accessor is not of type "number"' },
        },
      }],
    },
    {
      role: "user",
      parts: [{ text: "Summarize the prior history without using any tools." }],
    },
  ]);
});
