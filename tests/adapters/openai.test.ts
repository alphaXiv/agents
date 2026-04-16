import { OpenAIModel } from "../../mod.ts";
import { OpenAIAdapter } from "../../src/adapters/openai/adapter.ts";
import {
  createToolFixtures,
  INTEGRATION_TIMEOUT_MS,
  runAdapterToolStreamingTest,
  runAgentToolStreamingTest,
  runBackAndForthCalculatorConversationTest,
  runStructuredOutputStreamingTest,
  runStructuredToolParameterStreamingTest,
} from "./shared.ts";

const HAS_OPENAI_KEY = Boolean(Deno.env.get("OPENAI_API_KEY"));

Deno.test({
  name: "OpenAIAdapter streams a parameterized tool call (gpt-5.4-mini, reasoning)",
  ignore: !HAS_OPENAI_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    const fixtures = createToolFixtures();
    const adapter = new OpenAIAdapter({
      model: "gpt-5.4-mini",
      reasoning: { effort: "medium", summary: "auto" },
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
  name: "OpenAIModel streams tools and results (gpt-5.4, reasoning)",
  ignore: !HAS_OPENAI_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runAgentToolStreamingTest(t, {
      model: new OpenAIModel({ model: "gpt-5.4", effort: "low" }),
    });
  },
});

Deno.test({
  name: "OpenAIModel streams tools and results (gpt-5.4-mini, reasoning)",
  ignore: !HAS_OPENAI_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runAgentToolStreamingTest(t, {
      model: new OpenAIModel({ model: "gpt-5.4-mini", effort: "low" }),
    });
  },
});

Deno.test({
  name: "OpenAIModel executes structured tool parameters (gpt-5.4-mini, non-reasoning)",
  ignore: !HAS_OPENAI_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runStructuredToolParameterStreamingTest(t, {
      model: new OpenAIModel({ model: "gpt-5.4-mini" }),
    });
  },
});

Deno.test({
  name: "OpenAIModel streams structured output (gpt-5.4-mini, reasoning)",
  ignore: !HAS_OPENAI_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runStructuredOutputStreamingTest(t, {
      model: new OpenAIModel({ model: "gpt-5.4-mini", effort: "low" }),
    });
  },
});

Deno.test({
  name: "OpenAIModel keeps a 5-turn calculator conversation (gpt-5.4-nano)",
  ignore: !HAS_OPENAI_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runBackAndForthCalculatorConversationTest(t, {
      model: new OpenAIModel({ model: "gpt-5.4-nano" }),
    });
  },
});
