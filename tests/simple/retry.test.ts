import { assertEquals } from "@std/assert";
import type { ClassifiedError, ErrorKind } from "../../src/errors.ts";
import {
  DEFAULT_RETRY_STRATEGY,
  determineRetryBehavior,
  resolveRetryStrategy,
  type RetryBehavior,
  type RetryStrategy,
} from "../../src/retry.ts";

function createClassifiedError(kind: ErrorKind, status?: number): ClassifiedError {
  return { kind, status, original: new Error("Test error") };
}

Deno.test("resolveRetryStrategy returns defaults when no strategy provided", () => {
  const resolved = resolveRetryStrategy();
  assertEquals(resolved, DEFAULT_RETRY_STRATEGY);
});

Deno.test("resolveRetryStrategy merges partial strategy with defaults", () => {
  const partial: RetryStrategy = {
    sameModelRetries: 5,
    onTimeout: "retry-same",
  };

  const resolved = resolveRetryStrategy(partial);

  assertEquals(resolved.sameModelRetries, 5);
  assertEquals(resolved.onTimeout, "retry-same");
  assertEquals(resolved.modelCycles, DEFAULT_RETRY_STRATEGY.modelCycles);
  assertEquals(resolved.onRateLimit, DEFAULT_RETRY_STRATEGY.onRateLimit);
  assertEquals(resolved.onNetworkError, DEFAULT_RETRY_STRATEGY.onNetworkError);
  assertEquals(resolved.onServerError, DEFAULT_RETRY_STRATEGY.onServerError);
  assertEquals(resolved.onModelUnavailable, DEFAULT_RETRY_STRATEGY.onModelUnavailable);
});

Deno.test("resolveRetryStrategy allows full override of all fields", () => {
  const full: RetryStrategy = {
    sameModelRetries: 10,
    modelCycles: 3,
    onTimeout: "no-retry",
    onRateLimit: "retry-same",
    onModelUnavailable: "retry-same",
    onNetworkError: "switch-model",
    onServerError: "no-retry",
    customHandler: () => null,
  };

  const resolved = resolveRetryStrategy(full);

  assertEquals(resolved.sameModelRetries, 10);
  assertEquals(resolved.modelCycles, 3);
  assertEquals(resolved.onTimeout, "no-retry");
  assertEquals(resolved.onRateLimit, "retry-same");
  assertEquals(resolved.onModelUnavailable, "retry-same");
  assertEquals(resolved.onNetworkError, "switch-model");
  assertEquals(resolved.onServerError, "no-retry");
  assertEquals(typeof resolved.customHandler, "function");
});

Deno.test("resolveRetryStrategy returns a new object each time (no shared mutation)", () => {
  const a = resolveRetryStrategy();
  const b = resolveRetryStrategy();
  a.sameModelRetries = 999;
  assertEquals(b.sameModelRetries, DEFAULT_RETRY_STRATEGY.sameModelRetries);
});

const defaultBehaviorMatrix: Array<{ kind: ErrorKind; expected: RetryBehavior }> = [
  { kind: "aborted", expected: "no-retry" },
  { kind: "timeout", expected: "switch-model" },
  { kind: "rate_limit", expected: "switch-model" },
  { kind: "model_unavailable", expected: "switch-model" },
  { kind: "network", expected: "retry-same" },
  { kind: "server", expected: "retry-same" },
  { kind: "unknown", expected: "retry-same" },
  { kind: "auth", expected: "switch-model" },
  { kind: "client", expected: "switch-model" },
  { kind: "quota_exceeded", expected: "switch-model" },
  { kind: "unsupported_file_type", expected: "switch-model" },
  { kind: "context_overflow", expected: "switch-model" },
  { kind: "image_too_large", expected: "switch-model" },
];

for (const { kind, expected } of defaultBehaviorMatrix) {
  Deno.test(`determineRetryBehavior: ${kind} -> ${expected} with defaults`, () => {
    assertEquals(
      determineRetryBehavior(createClassifiedError(kind), DEFAULT_RETRY_STRATEGY, 0),
      expected,
    );
  });
}

Deno.test("determineRetryBehavior: timeout -> retry-same when configured", () => {
  const strategy = resolveRetryStrategy({ onTimeout: "retry-same" });
  assertEquals(determineRetryBehavior(createClassifiedError("timeout"), strategy, 0), "retry-same");
});

Deno.test("determineRetryBehavior: timeout -> no-retry when configured", () => {
  const strategy = resolveRetryStrategy({ onTimeout: "no-retry" });
  assertEquals(determineRetryBehavior(createClassifiedError("timeout"), strategy, 0), "no-retry");
});

Deno.test("determineRetryBehavior: rate_limit -> retry-same when configured", () => {
  const strategy = resolveRetryStrategy({ onRateLimit: "retry-same" });
  assertEquals(determineRetryBehavior(createClassifiedError("rate_limit"), strategy, 0), "retry-same");
});

Deno.test("determineRetryBehavior: rate_limit -> no-retry when configured", () => {
  const strategy = resolveRetryStrategy({ onRateLimit: "no-retry" });
  assertEquals(determineRetryBehavior(createClassifiedError("rate_limit"), strategy, 0), "no-retry");
});

Deno.test("determineRetryBehavior: model_unavailable -> retry-same when configured", () => {
  const strategy = resolveRetryStrategy({ onModelUnavailable: "retry-same" });
  assertEquals(determineRetryBehavior(createClassifiedError("model_unavailable"), strategy, 0), "retry-same");
});

Deno.test("determineRetryBehavior: network -> switch-model when configured", () => {
  const strategy = resolveRetryStrategy({ onNetworkError: "switch-model" });
  assertEquals(determineRetryBehavior(createClassifiedError("network"), strategy, 0), "switch-model");
});

Deno.test("determineRetryBehavior: network -> no-retry when configured", () => {
  const strategy = resolveRetryStrategy({ onNetworkError: "no-retry" });
  assertEquals(determineRetryBehavior(createClassifiedError("network"), strategy, 0), "no-retry");
});

Deno.test("determineRetryBehavior: server -> switch-model when configured", () => {
  const strategy = resolveRetryStrategy({ onServerError: "switch-model" });
  assertEquals(determineRetryBehavior(createClassifiedError("server"), strategy, 0), "switch-model");
});

Deno.test("determineRetryBehavior: server -> no-retry when configured", () => {
  const strategy = resolveRetryStrategy({ onServerError: "no-retry" });
  assertEquals(determineRetryBehavior(createClassifiedError("server"), strategy, 0), "no-retry");
});

Deno.test("determineRetryBehavior: network escalates to switch-model after exhausting retries", () => {
  const strategy = resolveRetryStrategy({ sameModelRetries: 2 });
  const classified = createClassifiedError("network");

  assertEquals(determineRetryBehavior(classified, strategy, 0), "retry-same");
  assertEquals(determineRetryBehavior(classified, strategy, 1), "retry-same");
  assertEquals(determineRetryBehavior(classified, strategy, 2), "switch-model");
  assertEquals(determineRetryBehavior(classified, strategy, 3), "switch-model");
});

Deno.test("determineRetryBehavior: server escalates to switch-model after exhausting retries", () => {
  const strategy = resolveRetryStrategy({ sameModelRetries: 2 });
  const classified = createClassifiedError("server");

  assertEquals(determineRetryBehavior(classified, strategy, 0), "retry-same");
  assertEquals(determineRetryBehavior(classified, strategy, 1), "retry-same");
  assertEquals(determineRetryBehavior(classified, strategy, 2), "switch-model");
});

Deno.test("determineRetryBehavior: sameModelRetries = 0 escalates immediately", () => {
  const strategy = resolveRetryStrategy({ sameModelRetries: 0 });
  assertEquals(determineRetryBehavior(createClassifiedError("network"), strategy, 0), "switch-model");
});

Deno.test("determineRetryBehavior: sameModelRetries = 1 allows exactly one retry", () => {
  const strategy = resolveRetryStrategy({ sameModelRetries: 1 });
  const classified = createClassifiedError("network");

  assertEquals(determineRetryBehavior(classified, strategy, 0), "retry-same");
  assertEquals(determineRetryBehavior(classified, strategy, 1), "switch-model");
});

Deno.test("determineRetryBehavior: exhaustion only affects retry-same behaviors", () => {
  const strategy = resolveRetryStrategy({ sameModelRetries: 0 });

  assertEquals(determineRetryBehavior(createClassifiedError("timeout"), strategy, 0), "switch-model");
  assertEquals(determineRetryBehavior(createClassifiedError("rate_limit"), strategy, 0), "switch-model");
  assertEquals(determineRetryBehavior(createClassifiedError("aborted"), strategy, 0), "no-retry");
});

Deno.test("determineRetryBehavior: configured retry-same for timeout escalates after exhaustion", () => {
  const strategy = resolveRetryStrategy({ onTimeout: "retry-same", sameModelRetries: 1 });
  const classified = createClassifiedError("timeout");

  assertEquals(determineRetryBehavior(classified, strategy, 0), "retry-same");
  assertEquals(determineRetryBehavior(classified, strategy, 1), "switch-model");
});

Deno.test("determineRetryBehavior: configured retry-same for rate_limit escalates after exhaustion", () => {
  const strategy = resolveRetryStrategy({ onRateLimit: "retry-same", sameModelRetries: 2 });
  const classified = createClassifiedError("rate_limit");

  assertEquals(determineRetryBehavior(classified, strategy, 0), "retry-same");
  assertEquals(determineRetryBehavior(classified, strategy, 1), "retry-same");
  assertEquals(determineRetryBehavior(classified, strategy, 2), "switch-model");
});

Deno.test("determineRetryBehavior: configured retry-same for model_unavailable escalates after exhaustion", () => {
  const strategy = resolveRetryStrategy({ onModelUnavailable: "retry-same", sameModelRetries: 1 });
  const classified = createClassifiedError("model_unavailable");

  assertEquals(determineRetryBehavior(classified, strategy, 0), "retry-same");
  assertEquals(determineRetryBehavior(classified, strategy, 1), "switch-model");
});

Deno.test("determineRetryBehavior: custom handler returning null falls through to default", () => {
  const strategy = resolveRetryStrategy({ customHandler: () => null });
  assertEquals(determineRetryBehavior(createClassifiedError("network"), strategy, 0), "retry-same");
});

Deno.test("determineRetryBehavior: custom handler receives the classified error", () => {
  let receivedError: ClassifiedError | null = null;

  const strategy = resolveRetryStrategy({
    customHandler: (classified) => {
      receivedError = classified;
      return null;
    },
  });

  const classified = createClassifiedError("rate_limit", 429);
  determineRetryBehavior(classified, strategy, 0);

  assertEquals(receivedError, classified);
  assertEquals(receivedError!.kind, "rate_limit");
  assertEquals(receivedError!.status, 429);
});

Deno.test("determineRetryBehavior: custom handler with conditional logic per kind", () => {
  const strategy = resolveRetryStrategy({
    customHandler: (classified) => {
      if (classified.kind === "rate_limit") return "retry-same";
      if (classified.kind === "auth") return "no-retry";
      return null;
    },
  });

  assertEquals(determineRetryBehavior(createClassifiedError("rate_limit"), strategy, 0), "retry-same");
  assertEquals(determineRetryBehavior(createClassifiedError("auth"), strategy, 0), "no-retry");
  assertEquals(determineRetryBehavior(createClassifiedError("network"), strategy, 0), "retry-same");
  assertEquals(determineRetryBehavior(createClassifiedError("timeout"), strategy, 0), "switch-model");
});

Deno.test("determineRetryBehavior: custom handler can override aborted (normally no-retry)", () => {
  const strategy = resolveRetryStrategy({
    customHandler: (classified) => {
      if (classified.kind === "aborted") return "retry-same";
      return null;
    },
  });

  assertEquals(determineRetryBehavior(createClassifiedError("aborted"), strategy, 0), "retry-same");
});

Deno.test("determineRetryBehavior: custom handler returning retry-same respects exhaustion limit", () => {
  const strategy = resolveRetryStrategy({
    sameModelRetries: 0,
    customHandler: () => "retry-same",
  });

  assertEquals(determineRetryBehavior(createClassifiedError("network"), strategy, 10), "switch-model");
});

Deno.test("determineRetryBehavior: custom handler can use status to decide", () => {
  const strategy = resolveRetryStrategy({
    customHandler: (classified) => {
      if (classified.status === 503) return "retry-same";
      return null;
    },
  });

  assertEquals(
    determineRetryBehavior(createClassifiedError("model_unavailable", 503), strategy, 0),
    "retry-same",
  );
  assertEquals(
    determineRetryBehavior(createClassifiedError("model_unavailable", 529), strategy, 0),
    "switch-model",
  );
});
