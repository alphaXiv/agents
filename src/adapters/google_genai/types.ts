// THIS FILE IS NECESSARY DUE TO `no-slow-types` BEING ENFORCED BY JSR.
// The models.ts contains actual inferable types, so if you update the types there you can just hover over the variables
// and copy the full types here to make jsr happy. I don't like this hack but I like strong types.
import type { z } from "zod";

export interface UnsupportedGoogleThinkingSupport {
  type: "unsupported";
  schema: z.ZodType<"off">;
}

export interface LegacyGoogleThinkingSupport {
  type: "legacyThinkingBudget";
  schema: z.ZodType<"minimal" | "low" | "medium" | "high" | "dynamic">;
}

export interface GoogleThinkingLevelSupport<T extends string> {
  type: "thinkingLevel";
  schema: z.ZodType<T>;
}

export interface GoogleModelSupportedThinkingLevelsMap {
  "gemini-3.5-flash": GoogleThinkingLevelSupport<"minimal" | "low" | "medium" | "high">;
  "gemini-3.1-flash-lite": GoogleThinkingLevelSupport<"minimal" | "low" | "medium" | "high">;
  "gemini-3.1-flash-image-preview": GoogleThinkingLevelSupport<"minimal" | "high">;
  "gemini-3.1-pro-preview": GoogleThinkingLevelSupport<"low" | "medium" | "high">;
  "gemini-3-pro-image-preview": UnsupportedGoogleThinkingSupport;
  "gemini-3-flash-preview": GoogleThinkingLevelSupport<"minimal" | "low" | "medium" | "high">;
  "gemini-2.5-pro": LegacyGoogleThinkingSupport;
  "gemini-2.5-flash": LegacyGoogleThinkingSupport;
  "gemini-2.5-flash-image": UnsupportedGoogleThinkingSupport;
  "gemini-2.5-flash-lite": LegacyGoogleThinkingSupport;
  "gemini-2.5-flash-lite-preview-09-2025": LegacyGoogleThinkingSupport;
  "gemini-2.5-flash-native-audio-preview-12-2025": LegacyGoogleThinkingSupport;
  "gemini-2.0-flash": UnsupportedGoogleThinkingSupport;
  "gemini-2.0-flash-001": UnsupportedGoogleThinkingSupport;
  "gemini-2.0-flash-lite": UnsupportedGoogleThinkingSupport;
  "gemini-2.0-flash-lite-001": UnsupportedGoogleThinkingSupport;
}
