// THIS FILE IS NECESSARY DUE TO `no-slow-types` BEING ENFORCED BY JSR.
// The models.ts contains actual inferable types, so if you update the types there you can just hover over the variables
// and copy the full types here to make jsr happy. I don't like this hack but I like strong types.
import type { z } from "zod";

export interface AdaptiveThinkingSupport<T extends string> {
  type: "adaptive";
  effortSchema: z.ZodType<T>;
}

export interface AdaptiveOrExtendedThinkingSupport<
  TThinking extends string,
  TEffort extends string,
> {
  type: "adaptiveOrExtended";
  schema: z.ZodType<TThinking>;
  effortSchema: z.ZodType<TEffort>;
}

export interface ExtendedWithEffortThinkingSupport<
  TThinking extends string,
  TEffort extends string,
> {
  type: "extendedWithEffort";
  schema: z.ZodType<TThinking>;
  effortSchema: z.ZodType<TEffort>;
}

export interface ExtendedThinkingSupport<T extends string> {
  type: "extended";
  schema: z.ZodType<T>;
}

export interface AnthropicModelThinkingSupportMap {
  "claude-opus-4-7": AdaptiveThinkingSupport<"low" | "medium" | "high" | "xhigh" | "max">;
  "claude-opus-4-6": AdaptiveThinkingSupport<"low" | "medium" | "high" | "max">;
  "claude-sonnet-4-6": AdaptiveOrExtendedThinkingSupport<
    "adaptive" | "low" | "medium" | "high",
    "low" | "medium" | "high"
  >;
  "claude-opus-4-5": ExtendedWithEffortThinkingSupport<
    "minimal" | "low" | "medium" | "high",
    "low" | "medium" | "high"
  >;
  "claude-opus-4-5-20251101": ExtendedWithEffortThinkingSupport<
    "minimal" | "low" | "medium" | "high",
    "low" | "medium" | "high"
  >;
  "claude-sonnet-4-5": ExtendedThinkingSupport<"minimal" | "low" | "medium" | "high">;
  "claude-sonnet-4-5-20250929": ExtendedThinkingSupport<"minimal" | "low" | "medium" | "high">;
  "claude-haiku-4-5": ExtendedThinkingSupport<"minimal" | "low" | "medium" | "high">;
  "claude-haiku-4-5-20251001": ExtendedThinkingSupport<"minimal" | "low" | "medium" | "high">;
  "claude-opus-4-1": ExtendedThinkingSupport<"minimal" | "low" | "medium" | "high">;
  "claude-opus-4-1-20250805": ExtendedThinkingSupport<"minimal" | "low" | "medium" | "high">;
  "claude-opus-4-0": ExtendedThinkingSupport<"minimal" | "low" | "medium" | "high">;
  "claude-opus-4-20250514": ExtendedThinkingSupport<"minimal" | "low" | "medium" | "high">;
  "claude-sonnet-4-0": ExtendedThinkingSupport<"minimal" | "low" | "medium" | "high">;
  "claude-sonnet-4-20250514": ExtendedThinkingSupport<"minimal" | "low" | "medium" | "high">;
}
