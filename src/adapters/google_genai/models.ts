import { ThinkingLevel as GenAiThinkingLevel } from "@google/genai";
import { z } from "zod";
import type {
  GoogleModelSupportedThinkingLevelsMap,
  GoogleThinkingLevelSupport,
  LegacyGoogleThinkingSupport,
  UnsupportedGoogleThinkingSupport,
} from "./types.ts";

type GoogleThinkingLevel =
  | "off"
  | "minimal"
  | "low"
  | "medium"
  | "high"
  | "dynamic";

type ThinkingLevelOptions<T extends readonly GoogleThinkingLevel[]> = {
  levels: T;
  default: T[number];
};

function thinkingUnsupported(): UnsupportedGoogleThinkingSupport {
  return {
    type: "unsupported" as const,
    schema: z.enum(["off"]).default("off"),
  };
}

function legacyThinkingBudget(): LegacyGoogleThinkingSupport {
  return {
    type: "legacyThinkingBudget" as const,
    schema: z.enum(["minimal", "low", "medium", "high", "dynamic"]).default("dynamic"),
  };
}

function thinkingLevel<const T extends readonly [GoogleThinkingLevel, ...GoogleThinkingLevel[]]>(
  options: ThinkingLevelOptions<T>,
): GoogleThinkingLevelSupport<T[number]> {
  return {
    type: "thinkingLevel" as const,
    schema: z.enum(options.levels).default(options.default),
  };
}

const googleModelSupportedThinkingLevelsDefinition = {
  "gemini-3.1-flash-lite-preview": thinkingLevel({
    levels: ["minimal", "low", "medium", "high"],
    default: "high",
  }),
  "gemini-3.1-flash-image-preview": thinkingLevel({
    levels: ["minimal", "high"],
    default: "minimal",
  }),
  "gemini-3.1-pro-preview": thinkingLevel({
    levels: ["low", "medium", "high"],
    default: "high",
  }),
  "gemini-3-pro-image-preview": thinkingUnsupported(),
  "gemini-3-flash-preview": thinkingLevel({
    levels: ["minimal", "low", "medium", "high"],
    default: "high",
  }),
  // Gemini 2.5 models use thinking_budget rather than thinking_level, but we
  // still expose the same abstraction and map it later.
  "gemini-2.5-pro": legacyThinkingBudget(),
  "gemini-2.5-flash": legacyThinkingBudget(),
  "gemini-2.5-flash-image": thinkingUnsupported(),
  "gemini-2.5-flash-lite": legacyThinkingBudget(),
  "gemini-2.5-flash-lite-preview-09-2025": legacyThinkingBudget(),
  "gemini-2.5-flash-native-audio-preview-12-2025": legacyThinkingBudget(),
  // No thinking support
  "gemini-2.0-flash": thinkingUnsupported(),
  "gemini-2.0-flash-001": thinkingUnsupported(),
  "gemini-2.0-flash-lite": thinkingUnsupported(),
  "gemini-2.0-flash-lite-001": thinkingUnsupported(),
} as const satisfies GoogleModelSupportedThinkingLevelsMap;

export const googleModelSupportedThinkingLevels: GoogleModelSupportedThinkingLevelsMap =
  googleModelSupportedThinkingLevelsDefinition;

export type GoogleModels = keyof typeof googleModelSupportedThinkingLevels;
export type SupportedThinkingLevel<TModel extends GoogleModels> = z.infer<
  (typeof googleModelSupportedThinkingLevels)[TModel]["schema"]
>;

const legacyThinkingBudgets: Record<GoogleThinkingLevel, number> = {
  off: 0,
  minimal: 512,
  low: 1024,
  medium: 8192,
  high: 24576,
  dynamic: -1,
} as const;

export function getLegacyThinkingBudget(thinkingLevel: GoogleThinkingLevel): number {
  return legacyThinkingBudgets[thinkingLevel];
}

export function getDefaultThinkingLevel<TModel extends GoogleModels>(model: TModel): SupportedThinkingLevel<TModel> {
  return googleModelSupportedThinkingLevels[model].schema.parse(undefined) as SupportedThinkingLevel<TModel>;
}

export function getGenAiThinkingLevel(thinkingLevel: SupportedThinkingLevel<GoogleModels>): GenAiThinkingLevel {
  switch (thinkingLevel) {
    case "low":
      return GenAiThinkingLevel.LOW;
    case "minimal":
      return GenAiThinkingLevel.MINIMAL;
    case "medium":
      return GenAiThinkingLevel.MEDIUM;
    case "high":
      return GenAiThinkingLevel.HIGH;
    case "dynamic":
    case "off":
      console.warn(
        `Thinking level "${thinkingLevel}" does not have a direct mapping to GenAI thinking levels, defaulting to THINKING_LEVEL_UNSPECIFIED. Consider adjusting the thinking level for better performance.`,
      );
      return GenAiThinkingLevel.THINKING_LEVEL_UNSPECIFIED;
  }

  thinkingLevel satisfies never;
  return GenAiThinkingLevel.THINKING_LEVEL_UNSPECIFIED;
}
