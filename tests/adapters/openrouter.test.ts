import { openrouterModel } from "../../src/adapters/openrouter/adapter.ts";
import {
  createToolFixtures,
  INTEGRATION_TIMEOUT_MS,
  runAdapterToolStreamingTest,
  runAgentToolStreamingTest,
  runBackAndForthCalculatorConversationTest,
  runStructuredOutputStreamingTest,
  runStructuredToolParameterStreamingTest,
} from "./shared.ts";

const HAS_OPENROUTER_KEY = Boolean(Deno.env.get("OPENROUTER_API_KEY"));

Deno.test({
  name: "OpenRouterAdapter streams a parameterized tool call (openai/gpt-5.4-mini)",
  ignore: !HAS_OPENROUTER_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    const fixtures = createToolFixtures();
    const adapter = openrouterModel({
      model: "openai/gpt-5.4-mini",
      reasoning: { enabled: true, effort: "medium" },
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
  name: "OpenRouterModel streams tools and results (openai/gpt-5-mini)",
  ignore: !HAS_OPENROUTER_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runAgentToolStreamingTest(t, {
      model: openrouterModel({ model: "openai/gpt-5-mini", reasoning: { enabled: true, effort: "medium" } }),
    });
  },
});

Deno.test({
  name: "OpenRouterModel streams tools and results (anthropic/claude-sonnet-4.5)",
  ignore: !HAS_OPENROUTER_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runAgentToolStreamingTest(t, {
      model: openrouterModel({ model: "anthropic/claude-sonnet-4.5", reasoning: { enabled: true, effort: "medium" } }),
    });
  },
});

Deno.test({
  name: "OpenRouterModel executes structured tool parameters (openai/gpt-5-mini)",
  ignore: !HAS_OPENROUTER_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runStructuredToolParameterStreamingTest(t, {
      model: openrouterModel({ model: "openai/gpt-5-mini", reasoning: { enabled: true, effort: "medium" } }),
    });
  },
});

Deno.test({
  name: "OpenRouterModel streams structured output (openai/gpt-5-mini)",
  ignore: !HAS_OPENROUTER_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runStructuredOutputStreamingTest(t, {
      model: openrouterModel({ model: "openai/gpt-5-mini", reasoning: { enabled: true, effort: "medium" } }),
    });
  },
});

Deno.test({
  name: "OpenRouterModel keeps a 5-turn calculator conversation (minimax/minimax-m2.7)",
  ignore: !HAS_OPENROUTER_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runBackAndForthCalculatorConversationTest(t, {
      model: openrouterModel({ model: "minimax/minimax-m2.7" }),
    });
  },
});
