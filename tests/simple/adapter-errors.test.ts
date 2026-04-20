import {
  APIConnectionError,
  APIConnectionTimeoutError,
  APIError,
  AuthenticationError,
  BadRequestError,
  InternalServerError,
  PermissionDeniedError,
  RateLimitError,
} from "@anthropic-ai/sdk";
import { assertEquals } from "@std/assert";
import {
  APIConnectionError as OpenAIAPIConnectionError,
  APIConnectionTimeoutError as OpenAIAPIConnectionTimeoutError,
  APIError as OpenAIAPIError,
  AuthenticationError as OpenAIAuthenticationError,
  BadRequestError as OpenAIBadRequestError,
  InternalServerError as OpenAIInternalServerError,
  PermissionDeniedError as OpenAIPermissionDeniedError,
  RateLimitError as OpenAIRateLimitError,
} from "openai";
import type { Adapter } from "../../src/adapters/adapter.ts";
import { AnthropicAdapter } from "../../src/adapters/anthropic/adapter.ts";
import { OpenResponsesAdapter } from "../../src/adapters/open_responses/adapter.ts";
import { OpenAICompletionsAdapter } from "../../src/adapters/openai_completions/adapter.ts";
import { type ClassifiedError, classifyError, type ErrorKind } from "../../src/errors.ts";

// Mirrors the real pipeline in agent.ts: adapter-specific classification first, heuristic fallback second
function classify(adapter: Adapter<string>, error: unknown): ClassifiedError {
  return adapter.classifyError?.(error) ?? classifyError(error);
}

function createMockAnthropicClient() {
  return {
    messages: {
      create: () => Promise.reject(new Error("Not implemented")),
    },
  } as never;
}

function createMockOpenAIClient() {
  return {
    responses: {
      create: () => Promise.reject(new Error("Not implemented")),
    },
    chat: {
      completions: {
        create: () => Promise.reject(new Error("Not implemented")),
      },
    },
  } as never;
}

const anthropicAdapter = new AnthropicAdapter({
  model: "claude-sonnet-4-20250514",
  client: createMockAnthropicClient(),
  streamConfig: {},
});

const openResponsesAdapter = new OpenResponsesAdapter({
  model: "gpt-4o",
  name: "OpenAI",
  client: createMockOpenAIClient(),
});

const openAICompletionsAdapter = new OpenAICompletionsAdapter({
  model: "gpt-4o",
  name: "OpenAI",
  client: createMockOpenAIClient(),
});

interface AdapterErrorTestCase {
  name: string;
  adapter: Adapter<string>;
  error: unknown;
  expectedKind: ErrorKind;
  expectedStatus?: number;
}

const testCases: AdapterErrorTestCase[] = [
  // -- Anthropic --
  {
    name: "Anthropic: APIConnectionTimeoutError -> timeout",
    adapter: anthropicAdapter,
    error: new APIConnectionTimeoutError({ message: "Request timed out" }),
    expectedKind: "timeout",
  },
  {
    name: "Anthropic: APIConnectionError -> network",
    adapter: anthropicAdapter,
    error: new APIConnectionError({ message: "Connection failed" }),
    expectedKind: "network",
  },
  {
    name: "Anthropic: RateLimitError -> rate_limit",
    adapter: anthropicAdapter,
    error: new RateLimitError(429, { message: "Too many requests" }, "Too many requests", new Headers()),
    expectedKind: "rate_limit",
    expectedStatus: 429,
  },
  {
    name: "Anthropic: AuthenticationError -> auth",
    adapter: anthropicAdapter,
    error: new AuthenticationError(401, { message: "Invalid API key" }, "Invalid API key", new Headers()),
    expectedKind: "auth",
  },
  {
    name: "Anthropic: PermissionDeniedError -> auth",
    adapter: anthropicAdapter,
    error: new PermissionDeniedError(403, { message: "Permission denied" }, "Permission denied", new Headers()),
    expectedKind: "auth",
  },
  {
    name: "Anthropic: InternalServerError -> server",
    adapter: anthropicAdapter,
    error: new InternalServerError(500, { message: "Internal error" }, "Internal error", new Headers()),
    expectedKind: "server",
  },
  {
    name: "Anthropic: BadRequestError with context overflow -> context_overflow",
    adapter: anthropicAdapter,
    error: new BadRequestError(
      400,
      { message: "prompt is too long: too many tokens" },
      "prompt is too long: too many tokens",
      new Headers(),
    ),
    expectedKind: "context_overflow",
  },
  {
    name: "Anthropic: BadRequestError with media type -> unsupported_file_type",
    adapter: anthropicAdapter,
    error: new BadRequestError(
      400,
      { message: "Unsupported media type: video/mp4" },
      "Unsupported media type: video/mp4",
      new Headers(),
    ),
    expectedKind: "unsupported_file_type",
  },
  {
    name: "Anthropic: generic BadRequestError -> client",
    adapter: anthropicAdapter,
    error: new BadRequestError(
      400,
      { message: "Invalid request format" },
      "Invalid request format",
      new Headers(),
    ),
    expectedKind: "client",
  },
  {
    name: "Anthropic: APIError with 503 -> model_unavailable",
    adapter: anthropicAdapter,
    error: new APIError(503, { message: "Service unavailable" }, "Service unavailable", new Headers()),
    expectedKind: "model_unavailable",
  },
  {
    name: "Anthropic: unknown error falls through to heuristics",
    adapter: anthropicAdapter,
    error: new Error("Something unexpected"),
    expectedKind: "unknown",
  },

  // -- OpenResponses --
  {
    name: "OpenResponses: APIConnectionTimeoutError -> timeout",
    adapter: openResponsesAdapter,
    error: new OpenAIAPIConnectionTimeoutError({ message: "Request timed out" }),
    expectedKind: "timeout",
  },
  {
    name: "OpenResponses: APIConnectionError -> network",
    adapter: openResponsesAdapter,
    error: new OpenAIAPIConnectionError({ message: "Connection failed" }),
    expectedKind: "network",
  },
  {
    name: "OpenResponses: RateLimitError -> rate_limit",
    adapter: openResponsesAdapter,
    error: new OpenAIRateLimitError(429, { message: "Rate limit exceeded" }, "Rate limit exceeded", new Headers()),
    expectedKind: "rate_limit",
  },
  {
    name: "OpenResponses: AuthenticationError -> auth",
    adapter: openResponsesAdapter,
    error: new OpenAIAuthenticationError(401, { message: "Invalid API key" }, "Invalid API key", new Headers()),
    expectedKind: "auth",
  },
  {
    name: "OpenResponses: BadRequestError with context overflow -> context_overflow",
    adapter: openResponsesAdapter,
    error: new OpenAIBadRequestError(
      400,
      { message: "This request exceeds maximum context length" },
      "This request exceeds maximum context length",
      new Headers(),
    ),
    expectedKind: "context_overflow",
  },
  {
    name: "OpenResponses: APIError with context_length_exceeded code -> context_overflow",
    adapter: openResponsesAdapter,
    error: new OpenAIAPIError(
      400,
      {
        message: "Your input exceeds the context window of this model. Please adjust your input and try again.",
        type: "invalid_request_error",
        code: "context_length_exceeded",
        param: "input",
      },
      "Your input exceeds the context window of this model. Please adjust your input and try again.",
      new Headers(),
    ),
    expectedKind: "context_overflow",
    expectedStatus: 400,
  },
  {
    name: "OpenResponses: unknown error falls through to heuristics",
    adapter: openResponsesAdapter,
    error: new TypeError("Cannot read property"),
    expectedKind: "unknown",
  },

  // -- OpenAICompletions --
  {
    name: "OpenAICompletions: APIConnectionTimeoutError -> timeout",
    adapter: openAICompletionsAdapter,
    error: new OpenAIAPIConnectionTimeoutError({ message: "Request timed out" }),
    expectedKind: "timeout",
  },
  {
    name: "OpenAICompletions: RateLimitError -> rate_limit",
    adapter: openAICompletionsAdapter,
    error: new OpenAIRateLimitError(429, { message: "Too many requests" }, "Too many requests", new Headers()),
    expectedKind: "rate_limit",
    expectedStatus: 429,
  },
  {
    name: "OpenAICompletions: InternalServerError -> server",
    adapter: openAICompletionsAdapter,
    error: new OpenAIInternalServerError(
      500,
      { message: "Internal server error" },
      "Internal server error",
      new Headers(),
    ),
    expectedKind: "server",
  },
  {
    name: "OpenAICompletions: PermissionDeniedError -> auth",
    adapter: openAICompletionsAdapter,
    error: new OpenAIPermissionDeniedError(403, { message: "Access denied" }, "Access denied", new Headers()),
    expectedKind: "auth",
  },
  {
    name: "OpenAICompletions: APIError with 503 -> model_unavailable",
    adapter: openAICompletionsAdapter,
    error: new OpenAIAPIError(503, { message: "Service overloaded" }, "Service overloaded", new Headers()),
    expectedKind: "model_unavailable",
  },
  {
    name: "OpenAICompletions: APIError with context_length_exceeded code -> context_overflow",
    adapter: openAICompletionsAdapter,
    error: new OpenAIAPIError(
      400,
      {
        message: "Your input exceeds the context window of this model. Please adjust your input and try again.",
        type: "invalid_request_error",
        code: "context_length_exceeded",
        param: "input",
      },
      "Your input exceeds the context window of this model. Please adjust your input and try again.",
      new Headers(),
    ),
    expectedKind: "context_overflow",
    expectedStatus: 400,
  },
  {
    name: "OpenAICompletions: unknown error falls through to heuristics",
    adapter: openAICompletionsAdapter,
    error: { message: "Random object error" },
    expectedKind: "unknown",
  },
];

for (const tc of testCases) {
  Deno.test(tc.name, () => {
    const result = classify(tc.adapter, tc.error);
    assertEquals(result.kind, tc.expectedKind);
    if (tc.expectedStatus !== undefined) {
      assertEquals(result.status, tc.expectedStatus);
    }
  });
}

Deno.test("classify preserves original error reference", () => {
  const originalError = new RateLimitError(429, { message: "Rate limited" }, "Rate limited", new Headers());
  const result = classify(anthropicAdapter, originalError);
  assertEquals(result.original, originalError);
});
