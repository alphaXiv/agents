import { requireEnv } from "../../util.ts";
import {
  type OpenResponsesClient,
  openResponsesModel,
  type OpenResponsesServiceTier,
} from "../open_responses/adapter.ts";
import type { Adapter } from "../adapter.ts";
import { getOpenAISupportedMimeTypes } from "../openai/mimes.ts";
import {
  getModelModalities,
  type OpenAIModels,
  resolveOpenAIReasoning,
  type SupportedReasoningEffort,
} from "../openai/models.ts";
import type { ProviderFileStore } from "../../types.ts";

/**
 * OpenAI models on Azure Foundry, addressed by model name rather than by deployment name.
 * A deployment named after a model id takes that name over, so never create one.
 *
 * With a `fileStore`, files are capped at 50 MB each, and Azure never expires an upload on its own.
 * The store's sweep is the only thing that deletes them.
 */
export function azureOpenAIModel<zO, zI, TModel extends OpenAIModels>(options: {
  model: TModel;
  apiKey?: string;
  endpoint?: string;
  serviceTier?: OpenResponsesServiceTier;
  effort?: SupportedReasoningEffort<TModel>;
  parallelToolCalls?: boolean;
  client?: OpenResponsesClient;
  /** When set, images and PDFs are uploaded to the Files API and sent as `file_id`. */
  fileStore?: ProviderFileStore;
}): Adapter<zO, zI> {
  return openResponsesModel({
    provider: "Azure",
    model: options.model,
    supportedMimeTypes: getOpenAISupportedMimeTypes(getModelModalities(options.model)),
    client: options.client,
    openAIOptions: options.client ? undefined : {
      apiKey: options.apiKey ?? requireEnv("AZURE_OPENAI_API_KEY"),
      baseURL: `${(options.endpoint ?? requireEnv("AZURE_OPENAI_ENDPOINT")).replace(/\/$/, "")}/openai/v1`,
    },
    reasoning: resolveOpenAIReasoning(options.model, options.effort),
    parallelToolCalls: options.parallelToolCalls,
    serviceTier: options.serviceTier,
    // Responses on Azure only reads ids uploaded with this purpose, and Azure rejects `expires_after`.
    files: options.fileStore ? { purpose: "assistants", store: options.fileStore } : undefined,
  });
}
