import { crossPlatformEnv, requireEnv } from "../../util.ts";
import { openResponsesModel, type OpenResponsesServiceTier } from "../open_responses/adapter.ts";
import { getOpenAISupportedMimeTypes } from "./mimes.ts";
import {
  getDefaultReasoningEffort,
  getModelModalities,
  openAiModelReasoningSupport,
  type OpenAIModels,
  type SupportedReasoningEffort,
} from "./models.ts";
import type { Adapter } from "../adapter.ts";
import type { OpenAIReasoningEffort } from "@alphaxiv/agents";

export function openAIModel<zO, zI, TModel extends OpenAIModels>(options: {
  model: TModel;
  apiKey?: string;
  baseUrl?: string;
  serviceTier?: OpenResponsesServiceTier;
  effort?: SupportedReasoningEffort<TModel>;
  parallelToolCalls?: boolean;
}): Adapter<zO, zI> {
  const modelConfig = openAiModelReasoningSupport[options.model];
  const typedOptions = options as { effort?: SupportedReasoningEffort<OpenAIModels> };
  const effort = "schema" in modelConfig ? typedOptions.effort ?? getDefaultReasoningEffort(options.model) : undefined;

  return openResponsesModel({
    provider: "OpenAI",
    model: options.model,
    supportedMimeTypes: getOpenAISupportedMimeTypes(getModelModalities(options.model)),
    openAIOptions: {
      apiKey: options.apiKey ?? requireEnv("OPENAI_API_KEY"),
      baseURL: options.baseUrl ?? crossPlatformEnv("OPENAI_BASE_URL") ?? "https://api.openai.com/v1",
    },
    reasoning: effort
      ? {
        effort: effort as OpenAIReasoningEffort,
        summary: effort === "none" ? undefined : "auto",
      }
      : undefined,
    parallelToolCalls: options.parallelToolCalls,
    serviceTier: options.serviceTier,
  });
}
