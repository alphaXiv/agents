import { crossPlatformEnv, requireEnv } from "../../util.ts";
import {
  type OpenResponsesClient,
  openResponsesModel,
  type OpenResponsesServiceTier,
} from "../open_responses/adapter.ts";
import type { Adapter } from "../adapter.ts";
import { getOpenAISupportedMimeTypes } from "./mimes.ts";
import {
  type OpenAIModelModality,
  openAiModelReasoningSupport,
  type OpenAIModels,
  type OpenAIReasoningEffort,
  type SupportedReasoningEffort,
} from "./models.ts";

export function openAIModel<zO, zI, TModel extends OpenAIModels | (string & Record<never, never>)>(options: {
  model: TModel;
  apiKey?: string;
  baseUrl?: string;
  serviceTier?: OpenResponsesServiceTier;
  effort?: TModel extends OpenAIModels ? SupportedReasoningEffort<TModel> : OpenAIReasoningEffort;
  /** Input modalities for a model ID that is not yet in the built-in list. Defaults to text. */
  modalities?: TModel extends OpenAIModels ? never : readonly OpenAIModelModality[];
  parallelToolCalls?: boolean;
  client?: OpenResponsesClient;
}): Adapter<zO, zI> {
  const modelConfig = openAiModelReasoningSupport[options.model as OpenAIModels];
  const effort = modelConfig && "schema" in modelConfig
    ? options.effort ?? modelConfig.schema.parse(undefined)
    : options.effort;

  return openResponsesModel({
    provider: "OpenAI",
    model: options.model,
    supportedMimeTypes: getOpenAISupportedMimeTypes(modelConfig?.modalities ?? options.modalities ?? ["text"]),
    client: options.client,
    openAIOptions: options.client ? undefined : {
      apiKey: options.apiKey ?? requireEnv("OPENAI_API_KEY"),
      baseURL: options.baseUrl ?? crossPlatformEnv("OPENAI_BASE_URL") ?? "https://api.openai.com/v1",
    },
    reasoning: effort
      ? {
        effort,
        summary: effort === "none" ? undefined : "auto",
      }
      : undefined,
    parallelToolCalls: options.parallelToolCalls,
    serviceTier: options.serviceTier,
  });
}
