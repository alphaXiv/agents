import { assertEquals, assertRejects } from "@std/assert";
import { APIError } from "openai";
import { classifyError } from "../../src/errors.ts";
import type { OpenAICompletionsClient } from "../../src/adapters/openai_completions/adapter.ts";
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

function createMockClient(captureRequest: (request: unknown) => void): OpenAICompletionsClient {
  return {
    chat: {
      completions: {
        stream(request: unknown) {
          captureRequest(request);
          return {
            async *[Symbol.asyncIterator]() {},
            // deno-lint-ignore require-await
            async finalChatCompletion() {
              return { usage: { prompt_tokens: 0, completion_tokens: 0 } };
            },
          };
        },
      },
    },
  } as unknown as OpenAICompletionsClient;
}

Deno.test("OpenRouterModel uses the provided completions client", async () => {
  let capturedRequest: unknown;
  const adapter = openrouterModel({
    model: "openai/gpt-5-mini",
    client: createMockClient((request) => capturedRequest = request),
    reasoning: { enabled: true, effort: "medium" },
  });

  const stream = adapter.stream({
    history: [],
    instructions: "test",
    tools: [],
    signal: AbortSignal.abort(),
  });

  assertEquals(await stream.next(), {
    done: true,
    value: { inputTokens: 0, outputTokens: 0, cacheReadTokens: null, cacheWriteTokens: 0 },
  });
  assertEquals((capturedRequest as { model?: unknown; reasoning?: unknown }).model, "openai/gpt-5-mini");
  assertEquals((capturedRequest as { reasoning?: unknown }).reasoning, { enabled: true, effort: "medium" });
});

Deno.test("OpenRouterModel surfaces the upstream provider error behind its own message", async () => {
  const providerError = APIError.generate(
    400,
    {
      error: {
        message: "Provider returned error",
        code: 400,
        metadata: {
          raw: JSON.stringify({
            error: {
              message: "This model's maximum context length is 400000 tokens.",
              type: "invalid_request_error",
            },
          }),
          provider_name: "OpenAI",
        },
      },
    },
    "400 Provider returned error",
    new Headers(),
  );

  const adapter = openrouterModel({
    model: "openai/gpt-5-mini",
    client: {
      chat: {
        completions: {
          stream() {
            throw providerError;
          },
        },
      },
    } as unknown as OpenAICompletionsClient,
  });

  const error = await assertRejects(
    () =>
      Array.fromAsync(adapter.stream({
        history: [],
        instructions: "test",
        tools: [],
        signal: AbortSignal.abort(),
      })),
    Error,
  );

  assertEquals(
    error.message,
    "400 Provider returned error: This model's maximum context length is 400000 tokens.",
  );
  // The point of surfacing it: an overflow behind the wrapper stops looking like a client error.
  assertEquals(classifyError(error).kind, "context_overflow");
});

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
