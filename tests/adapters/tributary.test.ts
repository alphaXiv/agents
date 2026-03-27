import { TributaryModel } from "../../mod.ts";
import { TributaryAdapter } from "../../src/adapters/tributary/adapter.ts";
import {
  createToolFixtures,
  INTEGRATION_TIMEOUT_MS,
  runAdapterToolStreamingTest,
  runAgentToolStreamingTest,
  runBackAndForthCalculatorConversationTest,
  runStructuredOutputStreamingTest,
} from "./shared.ts";

const HAS_TRIBUTARY_KEY = Boolean(Deno.env.get("TRIBUTARY_API_KEY"));

Deno.test({
  name: "TributaryAdapter streams a parameterized tool call (openai:gpt-5.4-mini)",
  ignore: !HAS_TRIBUTARY_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    const fixtures = createToolFixtures();
    const adapter = new TributaryAdapter({
      model: "openai:gpt-5.4-mini",
      reasoningEffort: "low",
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
  name: "TributaryModel streams tools and results (openai:gpt-5.4-mini)",
  ignore: !HAS_TRIBUTARY_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runAgentToolStreamingTest(t, {
      model: new TributaryModel({ model: "openai:gpt-5.4-mini", effort: "low" }),
    });
  },
});

Deno.test({
  name: "TributaryModel streams tools and results (anthropic:claude-sonnet-4.5)",
  ignore: !HAS_TRIBUTARY_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runAgentToolStreamingTest(t, {
      model: new TributaryModel({ model: "anthropic:claude-sonnet-4.5", effort: "low" }),
    });
  },
});

Deno.test({
  name: "TributaryModel streams structured output (openai:gpt-5.4-mini)",
  ignore: !HAS_TRIBUTARY_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runStructuredOutputStreamingTest(t, {
      model: new TributaryModel({ model: "openai:gpt-5.4-mini", effort: "low" }),
    });
  },
});

Deno.test({
  name: "TributaryModel keeps a 5-turn calculator conversation (openai:gpt-4.1-nano)",
  ignore: !HAS_TRIBUTARY_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runBackAndForthCalculatorConversationTest(t, {
      model: new TributaryModel({ model: "openai:gpt-4.1-nano" }),
    });
  },
});
