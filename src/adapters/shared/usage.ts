import type { ProviderStreamMetadata } from "../../types.ts";

/**
 * Normalizes providers that report caching the OpenAI / Gemini way — cached and
 * written tokens counted *inside* the prompt total — into the disjoint buckets this
 * library reports, where `inputTokens` means uncached, unwritten input only.
 *
 * Pass `writtenTokens` for providers that bill a cache write premium (OpenAI from
 * GPT-5.6 onward, at 1.25x the uncached rate). Omit it for providers whose caches
 * populate as a free side effect, which report no write count to begin with.
 */
export function splitCacheInclusiveUsage(
  promptTokens: number | null | undefined,
  cachedTokens: number | null | undefined,
  writtenTokens?: number | null | undefined,
): Pick<ProviderStreamMetadata, "inputTokens" | "cacheReadTokens" | "cacheWriteTokens"> {
  const cacheReadTokens = cachedTokens ?? null;
  // A provider that reported no usage at all has said nothing about writes either; claiming 0
  // there would price an unknown as free.
  const cacheWriteTokens = writtenTokens ?? (promptTokens == null ? null : 0);
  return {
    // Clamped because OpenAI has shipped responses whose cached and written counts exceed the
    // prompt total, which would otherwise report negative input.
    inputTokens: promptTokens == null
      ? null
      : Math.max(0, promptTokens - (cacheReadTokens ?? 0) - (cacheWriteTokens ?? 0)),
    cacheReadTokens,
    cacheWriteTokens,
  };
}
