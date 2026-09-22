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
  resolveOpenAIReasoning,
  type SupportedReasoningEffort,
} from "./models.ts";
import type { ProviderFileStore } from "../../types.ts";

/** Backup to the store's 7 day expiry, later so the store is always the one that deletes first. */
const FILE_EXPIRES_AFTER_SECONDS = 9 * 24 * 60 * 60;

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
  /** When set, images and PDFs are uploaded to the Files API and sent as `file_id`. */
  fileStore?: ProviderFileStore;
}): Adapter<zO, zI> {
  const modelConfig = openAiModelReasoningSupport[options.model as OpenAIModels];

  return openResponsesModel({
    provider: "OpenAI",
    model: options.model,
    supportedMimeTypes: getOpenAISupportedMimeTypes(modelConfig?.modalities ?? options.modalities ?? ["text"]),
    client: options.client,
    openAIOptions: options.client ? undefined : {
      apiKey: options.apiKey ?? requireEnv("OPENAI_API_KEY"),
      baseURL: options.baseUrl ?? crossPlatformEnv("OPENAI_BASE_URL") ?? "https://api.openai.com/v1",
    },
    reasoning: resolveOpenAIReasoning(options.model, options.effort),
    parallelToolCalls: options.parallelToolCalls,
    serviceTier: options.serviceTier,
    files: options.fileStore
      ? {
        purpose: "user_data",
        expiresAfterSeconds: FILE_EXPIRES_AFTER_SECONDS,
        store: options.fileStore,
      }
      : undefined,
  });
}
