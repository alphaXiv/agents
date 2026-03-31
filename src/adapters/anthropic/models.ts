import type { BetaOutputConfig, BetaThinkingConfigParam } from "@anthropic-ai/sdk/resources/beta/messages/messages";
import { z } from "zod";
import type {
  AdaptiveOrExtendedThinkingSupport,
  AdaptiveThinkingSupport,
  AnthropicModelThinkingSupportMap,
  ExtendedThinkingSupport,
  ExtendedWithEffortThinkingSupport,
} from "./types.ts";

export type ThinkingLevel = "adaptive" | "minimal" | "low" | "medium" | "high";

export type EffortLevel = "low" | "medium" | "high" | "max";

type ThinkingLevelOptions<T extends readonly ThinkingLevel[]> = {
  levels: T;
  default: T[number];
};

type EffortLevelOptions<T extends readonly EffortLevel[]> = {
  levels: T;
  default: T[number];
};

function adaptive<const T extends readonly [EffortLevel, ...EffortLevel[]]>(
  options: EffortLevelOptions<T>,
): AdaptiveThinkingSupport<T[number]> {
  return {
    type: "adaptive" as const,
    effortSchema: z.enum(options.levels).default(options.default),
  };
}

/** Supports both adaptive thinking (effort-controlled) and extended thinking (budget_tokens-controlled). */
function adaptiveOrExtended<
  const TL extends readonly [ThinkingLevel, ...ThinkingLevel[]],
  const TE extends readonly [EffortLevel, ...EffortLevel[]],
>(
  options: ThinkingLevelOptions<TL> & { effortLevels: TE; effortDefault: TE[number] },
): AdaptiveOrExtendedThinkingSupport<TL[number], TE[number]> {
  return {
    type: "adaptiveOrExtended" as const,
    schema: z.enum(options.levels).default(options.default),
    effortSchema: z.enum(options.effortLevels).default(options.effortDefault),
  };
}

function extendedWithEffort<
  const TL extends readonly [ThinkingLevel, ...ThinkingLevel[]],
  const TE extends readonly [EffortLevel, ...EffortLevel[]],
>(
  options: ThinkingLevelOptions<TL> & { effortLevels: TE; effortDefault: TE[number] },
): ExtendedWithEffortThinkingSupport<TL[number], TE[number]> {
  return {
    type: "extendedWithEffort" as const,
    schema: z.enum(options.levels).default(options.default),
    effortSchema: z.enum(options.effortLevels).default(options.effortDefault),
  };
}

function extended<const T extends readonly [ThinkingLevel, ...ThinkingLevel[]]>(
  options: ThinkingLevelOptions<T>,
): ExtendedThinkingSupport<T[number]> {
  return {
    type: "extended" as const,
    schema: z.enum(options.levels).default(options.default),
  };
}

const anthropicModelThinkingSupportDefinition = {
  // Adaptive thinking only (effort controls thinking intensity)
  "claude-opus-4-6": adaptive({
    levels: ["low", "medium", "high", "max"],
    default: "high",
  }),

  // Both adaptive thinking (default) and extended thinking (opt-in via thinkingLevel)
  "claude-sonnet-4-6": adaptiveOrExtended({
    // "adaptive" means: use adaptive mode. Any other value switches to extended mode with budget_tokens.
    levels: ["adaptive", "low", "medium", "high"],
    default: "adaptive",
    effortLevels: ["low", "medium", "high"],
    effortDefault: "medium",
  }),

  // Extended thinking + Effort
  "claude-opus-4-5": extendedWithEffort({
    levels: ["minimal", "low", "medium", "high"],
    default: "high",
    effortLevels: ["low", "medium", "high"],
    effortDefault: "high",
  }),
  "claude-opus-4-5-20251101": extendedWithEffort({
    levels: ["minimal", "low", "medium", "high"],
    default: "high",
    effortLevels: ["low", "medium", "high"],
    effortDefault: "high",
  }),

  // Extended thinking only
  "claude-sonnet-4-5": extended({
    levels: ["minimal", "low", "medium", "high"],
    default: "high",
  }),
  "claude-sonnet-4-5-20250929": extended({
    levels: ["minimal", "low", "medium", "high"],
    default: "high",
  }),

  "claude-haiku-4-5": extended({
    levels: ["minimal", "low", "medium", "high"],
    default: "high",
  }),
  "claude-haiku-4-5-20251001": extended({
    levels: ["minimal", "low", "medium", "high"],
    default: "high",
  }),

  "claude-opus-4-1": extended({
    levels: ["minimal", "low", "medium", "high"],
    default: "high",
  }),
  "claude-opus-4-1-20250805": extended({
    levels: ["minimal", "low", "medium", "high"],
    default: "high",
  }),

  "claude-opus-4-0": extended({
    levels: ["minimal", "low", "medium", "high"],
    default: "high",
  }),
  "claude-opus-4-20250514": extended({
    levels: ["minimal", "low", "medium", "high"],
    default: "high",
  }),

  "claude-sonnet-4-0": extended({
    levels: ["minimal", "low", "medium", "high"],
    default: "high",
  }),
  "claude-sonnet-4-20250514": extended({
    levels: ["minimal", "low", "medium", "high"],
    default: "high",
  }),
} as const satisfies AnthropicModelThinkingSupportMap;

export const anthropicModelThinkingSupport: AnthropicModelThinkingSupportMap = anthropicModelThinkingSupportDefinition;

export type AnthropicModels = keyof typeof anthropicModelThinkingSupport;

type ModelConfig<TModel extends AnthropicModels> = (typeof anthropicModelThinkingSupport)[TModel];

/** Thinking level (controls budget_tokens) for models with extended thinking. `never` for adaptive models. */
export type SupportedThinkingLevel<TModel extends AnthropicModels> = ModelConfig<TModel> extends
  { schema: z.ZodType<infer V> } ? V : never;

/** Effort level for models with effort support. `never` for extended-only models. */
export type SupportedEffortLevel<TModel extends AnthropicModels> = ModelConfig<TModel> extends
  { effortSchema: z.ZodType<infer V> } ? V : never;

/** `true` for models that support interleaved thinking (all extended thinking models); `false` for adaptive-only models. */
export type SupportsInterleaved<TModel extends AnthropicModels> = ModelConfig<TModel>["type"] extends "adaptive" ? false
  : true;

/**
 * Model support for native structured ouput.
 */
export const anthropicModelStructuredOutputSupport = {
  "claude-opus-4-6": true,
  "claude-sonnet-4-6": true,

  "claude-opus-4-5": true,
  "claude-opus-4-5-20251101": true,
  "claude-sonnet-4-5": true,
  "claude-sonnet-4-5-20250929": true,
  "claude-haiku-4-5": true,
  "claude-haiku-4-5-20251001": true,

  "claude-opus-4-1": false,
  "claude-opus-4-1-20250805": false,

  "claude-opus-4-0": false,
  "claude-opus-4-20250514": false,
  "claude-sonnet-4-0": false,
  "claude-sonnet-4-20250514": false,
};

const thinkingBudgets: Record<Exclude<ThinkingLevel, "adaptive">, number> = {
  minimal: 1024,
  low: 2048,
  medium: 8192,
  high: 16000,
};

export function getDefaultThinkingLevel<TModel extends AnthropicModels>(
  model: TModel,
): SupportedThinkingLevel<TModel> {
  const config = anthropicModelThinkingSupport[model];
  if (!("schema" in config)) return undefined as never;
  return config.schema.parse(undefined) as SupportedThinkingLevel<TModel>;
}

export function getDefaultEffortLevel<TModel extends AnthropicModels>(
  model: TModel,
): SupportedEffortLevel<TModel> {
  const config = anthropicModelThinkingSupport[model];
  if (!("effortSchema" in config)) return undefined as never;
  return config.effortSchema.parse(undefined) as SupportedEffortLevel<TModel>;
}

export interface AnthropicMessagesStreamConfig {
  thinking?: BetaThinkingConfigParam;
  output_config?: BetaOutputConfig;
  betas?: string[];
}

export interface GetAnthropicMessagesStreamConfigOptions {
  model: AnthropicModels;
  thinkingLevel?: ThinkingLevel;
  effort?: EffortLevel;
  interleaved?: boolean;
}

export function getAnthropicMessagesStreamConfig(
  { model, thinkingLevel, effort, interleaved }: GetAnthropicMessagesStreamConfigOptions,
): AnthropicMessagesStreamConfig {
  const support = anthropicModelThinkingSupport[model];

  // When betas is just an empty array, the SDK throws an error about unexpected value(s).
  let betas: string[] | undefined;

  const outputConfig: BetaOutputConfig | undefined = effort ? { effort } : undefined;

  switch (support.type) {
    case "adaptive": {
      // Interleaved thinking is enabled automatically by adaptive thinking - no beta header needed.
      return {
        thinking: { type: "adaptive" },
        output_config: outputConfig,
        betas,
      };
    }

    case "adaptiveOrExtended": {
      const level = thinkingLevel ?? "adaptive";
      if (level !== "adaptive") {
        // Extended thinking mode
        if (interleaved ?? true) (betas ??= []).push("interleaved-thinking-2025-05-14");
        return {
          thinking: { type: "enabled", budget_tokens: thinkingBudgets[level] },
          betas,
        };
      }
      // Adaptive thinking mode - interleaved is automatic, no beta header needed.
      return {
        thinking: { type: "adaptive" },
        output_config: outputConfig,
        betas,
      };
    }

    case "extendedWithEffort": {
      if (!thinkingLevel || thinkingLevel === "adaptive") {
        return {
          output_config: outputConfig,
          betas,
        };
      }

      if (interleaved ?? true) (betas ??= []).push("interleaved-thinking-2025-05-14");
      return {
        thinking: { type: "enabled", budget_tokens: thinkingBudgets[thinkingLevel] },
        output_config: outputConfig,
        betas,
      };
    }

    case "extended": {
      if (!thinkingLevel || thinkingLevel === "adaptive") {
        return {
          betas,
        };
      }

      if (interleaved ?? true) (betas ??= []).push("interleaved-thinking-2025-05-14");
      return {
        thinking: { type: "enabled", budget_tokens: thinkingBudgets[thinkingLevel] },
        betas,
      };
    }
  }

  support satisfies never;
  throw new Error(`Unsupported Anthropic thinking support config: ${JSON.stringify(support)}`);
}
