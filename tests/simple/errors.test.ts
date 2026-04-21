import { assertEquals } from "@std/assert";
import { classifyError, ERROR_KINDS, type ErrorKind } from "../../src/errors.ts";

function createAbortError() {
  try {
    const abortController = new AbortController();
    abortController.abort();
    abortController.signal.throwIfAborted();
    return new Error("should not reach");
  } catch (error) {
    return error as Error;
  }
}

interface ClassifyErrorTestCase {
  name: string;
  error: unknown;
  status?: number;
  expected: {
    kind: ErrorKind;
    status?: number;
  };
}

const testCases: ClassifyErrorTestCase[] = [
  // Abort errors
  {
    name: "classifies AbortError by name",
    error: createAbortError(),
    expected: { kind: "aborted" },
  },

  // Timeout errors
  {
    name: "classifies TimeoutError by name",
    error: { name: "TimeoutError", message: "Request timed out" },
    expected: { kind: "timeout" },
  },
  {
    name: "classifies timeout from message heuristics",
    error: "Connection timed out",
    expected: { kind: "timeout" },
  },
  {
    name: "classifies ETIMEDOUT from message",
    error: "connect ETIMEDOUT 192.168.1.1:443",
    expected: { kind: "timeout" },
  },

  // Network errors
  {
    name: "classifies network error from networkError flag",
    error: { message: "Failed to fetch", networkError: true },
    expected: { kind: "network" },
  },
  {
    name: "classifies Failed to fetch message",
    error: { message: "Failed to fetch" },
    expected: { kind: "network" },
  },
  {
    name: "classifies ECONNREFUSED from message",
    error: "connect ECONNREFUSED 127.0.0.1:3000",
    expected: { kind: "network" },
  },
  {
    name: "classifies ECONNRESET from message",
    error: "read ECONNRESET",
    expected: { kind: "network" },
  },
  {
    name: "classifies socket hang up from message",
    error: "socket hang up",
    expected: { kind: "network" },
  },

  // Rate limit errors
  {
    name: "classifies rate limit from 429 status",
    error: { status: 429, message: "Too many requests" },
    expected: { kind: "rate_limit", status: 429 },
  },
  {
    name: "classifies quota exceeded from 429 with quota message",
    error: { status: 429, message: "Quota exceeded for this workspace" },
    expected: { kind: "quota_exceeded", status: 429 },
  },
  {
    name: "classifies rate limit from too many requests message",
    error: "too many requests, please slow down",
    expected: { kind: "rate_limit" },
  },

  // Auth errors
  {
    name: "classifies auth error from 401 status",
    error: { status: 401, message: "Unauthorized" },
    expected: { kind: "auth", status: 401 },
  },
  {
    name: "classifies auth error from 403 status",
    error: { status: 403, message: "Forbidden" },
    expected: { kind: "auth", status: 403 },
  },

  // Server errors
  {
    name: "classifies server error from 500 status",
    error: { status: 500, message: "Internal Server Error" },
    expected: { kind: "server", status: 500 },
  },
  {
    name: "classifies server error from 502 status",
    error: { status: 502, message: "Bad Gateway" },
    expected: { kind: "server", status: 502 },
  },
  {
    name: "classifies model unavailable from 503 status",
    error: { statusCode: 503, message: "Service Unavailable" },
    expected: { kind: "model_unavailable", status: 503 },
  },
  {
    name: "classifies model unavailable from 529 status",
    error: { status: 529, message: "API is temporarily overloaded" },
    expected: { kind: "model_unavailable", status: 529 },
  },
  {
    name: "classifies timeout from 504 status",
    error: { status: 504, message: "Gateway Timeout" },
    expected: { kind: "timeout", status: 504 },
  },

  // Client errors
  {
    name: "classifies client error from 400 status",
    error: { status: 400, message: "Bad Request" },
    expected: { kind: "client", status: 400 },
  },
  {
    name: "classifies client error from 404 status",
    error: { status: 404, message: "Not Found" },
    expected: { kind: "client", status: 404 },
  },

  // Model unavailable
  {
    name: "classifies model unavailable from usage limits message",
    error: "You have reached your specified API usage limits. You will regain access on 2026-04-01 at 00:00 UTC.",
    expected: { kind: "model_unavailable" },
  },
  {
    name: "classifies model unavailable from overloaded message",
    error: "The model is currently overloaded with requests",
    expected: { kind: "model_unavailable" },
  },
  {
    name: "classifies model unavailable from high demand message",
    error: "This model is currently experiencing high demand",
    expected: { kind: "model_unavailable" },
  },

  // Context overflow
  {
    name: "classifies context overflow from token count exceeded message",
    error: "The input token count exceeds the maximum number of tokens allowed 1048576",
    expected: { kind: "context_overflow" },
  },
  {
    name: "classifies context overflow from context length message",
    error: "This model's maximum context length is 128000 tokens",
    expected: { kind: "context_overflow" },
  },
  {
    name: "classifies context overflow from context window message",
    error: "Your input exceeds the context window of this model. Please adjust your input and try again.",
    expected: { kind: "context_overflow" },
  },
  {
    name: "classifies context overflow from token limit message",
    error: "Request exceeds token limit",
    expected: { kind: "context_overflow" },
  },
  {
    name: "classifies context overflow from request body too large message",
    error: { status: 413, message: "Request body too large" },
    expected: { kind: "context_overflow", status: 413 },
  },
  {
    name: "classifies context overflow from 413 status without helpful message",
    error: { status: 413, message: "Payload rejected" },
    expected: { kind: "context_overflow", status: 413 },
  },
  {
    name: "classifies context overflow from failed to buffer request body message",
    error: { status: 413, message: "413 Failed to buffer the request body: length limit exceeded" },
    expected: { kind: "context_overflow", status: 413 },
  },
  {
    name: "classifies context overflow from input too long message",
    error: "Input too long",
    expected: { kind: "context_overflow" },
  },
  {
    name: "classifies context overflow from input too large message",
    error: "Input too large",
    expected: { kind: "context_overflow" },
  },
  {
    name: "classifies context overflow from body too long message",
    error: "Body too long",
    expected: { kind: "context_overflow" },
  },
  {
    name: "classifies context overflow from body too large message",
    error: "Body too large",
    expected: { kind: "context_overflow" },
  },
  {
    name: "classifies context overflow from request length exceeded message",
    error: "Request length exceeded",
    expected: { kind: "context_overflow" },
  },
  {
    name: "classifies context overflow from context_length_exceeded code",
    error: { error: { code: "context_length_exceeded" } },
    expected: { kind: "context_overflow" },
  },

  // Unsupported file type
  {
    name: "classifies unsupported file type from mime type message",
    error: "Unsupported MIME type: application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    expected: { kind: "unsupported_file_type" },
  },
  {
    name: "classifies unsupported file type from Anthropic-style message",
    error: "Anthropic models don't support the following media type: application/pdf",
    expected: { kind: "unsupported_file_type" },
  },
  {
    name: "classifies unsupported file type from model-style message",
    error: 'Model "gpt-4" does not support media type "video/mp4"',
    expected: { kind: "unsupported_file_type" },
  },

  // Image too large
  {
    name: "classifies image too large from dimension message",
    error: "image dimensions exceed max allowed size for many-image requests: 2000 pixels",
    expected: { kind: "image_too_large" },
  },

  // Prompt too long
  {
    name: "classifies prompt too long from provider message",
    error: "prompt is too long: 2891660 tokens > 1000000 maximum",
    expected: { kind: "context_overflow" },
  },

  // Nested status extraction
  {
    name: "extracts nested status from error object",
    error: { error: { status: 503, message: "Service temporarily unavailable" } },
    expected: { kind: "model_unavailable", status: 503 },
  },
  {
    name: "extracts nested status from JSON string",
    error: '{"error": {"code": 429, "message": "Rate limited"}}',
    expected: { kind: "rate_limit", status: 429 },
  },

  // Unknown errors
  {
    name: "classifies unknown error as unknown",
    error: "Something unexpected happened",
    expected: { kind: "unknown" },
  },
  {
    name: "handles null error",
    error: null,
    expected: { kind: "unknown" },
  },
  {
    name: "handles undefined error",
    error: undefined,
    expected: { kind: "unknown" },
  },
];

Deno.test("classifyError covers all error kinds", () => {
  const coveredKinds = new Set(testCases.map((tc) => tc.expected.kind));
  const missingKinds = ERROR_KINDS.filter((kind) => !coveredKinds.has(kind));

  assertEquals(
    missingKinds.length,
    0,
    `Missing test coverage for error kinds: ${missingKinds.join(", ")}`,
  );
});

for (const testCase of testCases) {
  Deno.test(`classifyError: ${testCase.name}`, () => {
    const result = classifyError(testCase.error, testCase.status);

    assertEquals(result.kind, testCase.expected.kind);

    if (testCase.expected.status !== undefined) {
      assertEquals(result.status, testCase.expected.status);
    }
  });
}

Deno.test("classifyError prefers explicit status argument over extracted status", () => {
  const result = classifyError({ error: { status: 503 } }, 429);
  assertEquals(result.status, 429);
  assertEquals(result.kind, "rate_limit");
});

Deno.test("classifyError preserves original error reference", () => {
  const originalError = new Error("Test error");
  const result = classifyError(originalError);
  assertEquals(result.original, originalError);
});
