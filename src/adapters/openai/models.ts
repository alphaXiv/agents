import { z } from "zod";
import type { NonReasoningModelSupport, OpenAiModelsMap, ReasoningModelSupport } from "./types.ts";

export type OpenAIReasoningEffort = "none" | "minimal" | "low" | "medium" | "high" | "xhigh";
export type OpenAIModelModality = "text" | "image" | "audio" | "video";

type ModelModalitiesOptions = {
  modalities: readonly OpenAIModelModality[];
};

type ReasoningEffortOptions<T extends readonly [OpenAIReasoningEffort, ...OpenAIReasoningEffort[]]> =
  & ModelModalitiesOptions
  & {
    levels: T;
    default: T[number];
  };

function nonReasoning(options: ModelModalitiesOptions): NonReasoningModelSupport {
  return {
    type: "unsupported" as const,
    modalities: options.modalities,
  };
}

function reasoning<const T extends readonly [OpenAIReasoningEffort, ...OpenAIReasoningEffort[]]>(
  options: ReasoningEffortOptions<T>,
): ReasoningModelSupport<T> {
  return {
    type: "reasoning" as const,
    schema: z.enum(options.levels).default(options.default),
    modalities: options.modalities,
  };
}

const openAiModelsDefinition = {
  // Frontier
  "gpt-5.4": reasoning({
    levels: ["none", "low", "medium", "high", "xhigh"],
    default: "medium",
    modalities: ["text", "image"],
  }),
  "gpt-5.4-pro": reasoning({
    levels: ["medium", "high", "xhigh"],
    default: "medium",
    modalities: ["text", "image"],
  }),
  "gpt-5.4-mini": reasoning({
    levels: ["none", "low", "medium", "high"],
    default: "medium",
    modalities: ["text", "image"],
  }),
  "gpt-5.4-nano": reasoning({
    levels: ["none", "low", "medium", "high"],
    default: "medium",
    modalities: ["text", "image"],
  }),

  "gpt-5.3-codex": reasoning({
    levels: ["low", "medium", "high", "xhigh"],
    default: "medium",
    modalities: ["text", "image"],
  }),

  "gpt-5.2": reasoning({
    levels: ["none", "low", "medium", "high", "xhigh"],
    default: "none",
    modalities: ["text", "image"],
  }),
  "gpt-5.2-codex": reasoning({
    levels: ["low", "medium", "high", "xhigh"],
    default: "medium",
    modalities: ["text", "image"],
  }),
  "gpt-5.2-pro": reasoning({
    levels: ["medium", "high", "xhigh"],
    default: "medium",
    modalities: ["text", "image"],
  }),

  "gpt-5.1": reasoning({
    levels: ["none", "low", "medium", "high"],
    default: "none",
    modalities: ["text", "image"],
  }),
  "gpt-5.1-codex": reasoning({
    levels: ["none", "low", "medium", "high"],
    default: "none",
    modalities: ["text", "image"],
  }),
  "gpt-5.1-codex-max": reasoning({
    levels: ["none", "low", "medium", "high"],
    default: "none",
    modalities: ["text", "image"],
  }),
  "gpt-5.1-codex-mini": reasoning({
    levels: ["none", "low", "medium", "high"],
    default: "none",
    modalities: ["text", "image"],
  }),

  "gpt-5": reasoning({
    levels: ["minimal", "low", "medium", "high"],
    default: "medium",
    modalities: ["text", "image"],
  }),
  "gpt-5-pro": reasoning({
    levels: ["high"],
    default: "high",
    modalities: ["text", "image"],
  }),
  "gpt-5-mini": reasoning({
    levels: ["minimal", "low", "medium", "high"],
    default: "medium",
    modalities: ["text", "image"],
  }),
  "gpt-5-nano": reasoning({
    levels: ["minimal", "low", "medium", "high"],
    default: "medium",
    modalities: ["text", "image"],
  }),
  "gpt-5-codex": reasoning({
    levels: ["minimal", "low", "medium", "high"],
    default: "medium",
    modalities: ["text", "image"],
  }),

  "gpt-4.1": nonReasoning({ modalities: ["text", "image"] }),
  "gpt-4.1-mini": nonReasoning({ modalities: ["text", "image"] }),
  "gpt-4.1-nano": nonReasoning({ modalities: ["text", "image"] }),

  // Open-weight
  "gpt-oss-120b": reasoning({
    levels: ["low", "medium", "high"],
    default: "medium",
    modalities: ["text"],
  }),
  "gpt-oss-20b": reasoning({
    levels: ["low", "medium", "high"],
    default: "medium",
    modalities: ["text"],
  }),

  // Other currently listed text models that support streaming
  "o4-mini-deep-research": reasoning({
    levels: ["low", "medium", "high"],
    default: "medium",
    modalities: ["text", "image"],
  }),
  "o4-mini": reasoning({
    levels: ["low", "medium", "high"],
    default: "medium",
    modalities: ["text", "image"],
  }),
  "o1": reasoning({
    levels: ["low", "medium", "high"],
    default: "medium",
    modalities: ["text", "image"],
  }),
  "gpt-4o": nonReasoning({ modalities: ["text", "image"] }),
  "gpt-4o-mini": nonReasoning({ modalities: ["text", "image"] }),
  "gpt-4-turbo": nonReasoning({ modalities: ["text", "image"] }),
  "gpt-4": nonReasoning({ modalities: ["text"] }),
} as const satisfies OpenAiModelsMap;

export const openAiModels: OpenAiModelsMap = openAiModelsDefinition;

export const openAiModelReasoningSupport: OpenAiModelsMap = openAiModels;

export type OpenAIModels = keyof typeof openAiModels;

type ModelConfig<TModel extends OpenAIModels> = (typeof openAiModels)[TModel];

export type SupportedReasoningEffort<TModel extends OpenAIModels> = ModelConfig<TModel> extends
  { schema: z.ZodType<infer V> } ? V : never;

export function getModelModalities<TModel extends OpenAIModels>(model: TModel): readonly OpenAIModelModality[] {
  return openAiModels[model].modalities as readonly OpenAIModelModality[];
}

export function getDefaultReasoningEffort<TModel extends OpenAIModels>(
  model: TModel,
): SupportedReasoningEffort<TModel> {
  const config = openAiModels[model];
  if (!("schema" in config)) return undefined as never;
  return config.schema.parse(undefined) as SupportedReasoningEffort<TModel>;
}
