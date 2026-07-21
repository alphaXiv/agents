import { assertEquals } from "@std/assert";
import { splitCacheInclusiveUsage } from "../../src/adapters/shared/usage.ts";

Deno.test("splitCacheInclusiveUsage subtracts cached tokens out of the prompt total", () => {
  assertEquals(splitCacheInclusiveUsage(1500, 1024), {
    inputTokens: 476,
    cacheReadTokens: 1024,
    cacheWriteTokens: 0,
  });
});

Deno.test("splitCacheInclusiveUsage reports an uncached call unchanged", () => {
  assertEquals(splitCacheInclusiveUsage(1500, 0), {
    inputTokens: 1500,
    cacheReadTokens: 0,
    cacheWriteTokens: 0,
  });
});

Deno.test("splitCacheInclusiveUsage treats an unreported cache as no cache", () => {
  assertEquals(splitCacheInclusiveUsage(1500, undefined), {
    inputTokens: 1500,
    cacheReadTokens: null,
    cacheWriteTokens: 0,
  });
});

Deno.test("splitCacheInclusiveUsage keeps unreported usage null rather than inventing 0", () => {
  assertEquals(splitCacheInclusiveUsage(undefined, undefined), {
    inputTokens: null,
    cacheReadTokens: null,
    cacheWriteTokens: null,
  });
});

Deno.test("splitCacheInclusiveUsage never reports negative input tokens", () => {
  assertEquals(splitCacheInclusiveUsage(100, 500), {
    inputTokens: 0,
    cacheReadTokens: 500,
    cacheWriteTokens: 0,
  });
});

Deno.test("splitCacheInclusiveUsage subtracts written tokens out of the prompt total too", () => {
  // A provider that bills a write premium (OpenAI from GPT-5.6) counts those tokens in the
  // prompt total as well, so leaving them in inputTokens would price them at the full rate.
  assertEquals(splitCacheInclusiveUsage(1500, 1024, 400), {
    inputTokens: 76,
    cacheReadTokens: 1024,
    cacheWriteTokens: 400,
  });
});

Deno.test("splitCacheInclusiveUsage reports a pure cache write with no read", () => {
  assertEquals(splitCacheInclusiveUsage(1500, 0, 1500), {
    inputTokens: 0,
    cacheReadTokens: 0,
    cacheWriteTokens: 1500,
  });
});

Deno.test("splitCacheInclusiveUsage never reports negative input tokens against writes", () => {
  assertEquals(splitCacheInclusiveUsage(100, 80, 80), {
    inputTokens: 0,
    cacheReadTokens: 80,
    cacheWriteTokens: 80,
  });
});
