import { crossPlatformEnv, requireEnv } from "../../util.ts";
import { openAICompletionsModel, type OpenAICompletionsPdfSupport } from "../openai_completions/adapter.ts";
import type { ReasoningEffort } from "openai/resources/shared";
import type { Adapter } from "../adapter.ts";

// TODO: Maybe make it strictly typed but will be extremely hard to keep up to date
export type TributaryModels = string;

export function tributaryModel<zO, zI, TModel extends TributaryModels>(options: {
  model: TModel;
  apiKey?: string;
  baseUrl?: string;
  pdfSupport?: OpenAICompletionsPdfSupport<TModel>;
  reasoningEffort?: ReasoningEffort;
  extraRequestBody?: Record<string, unknown>;
}): Adapter<zO, zI> {
  return openAICompletionsModel({
    provider: "Tributary",
    model: options.model,
    openAIOptions: {
      apiKey: options.apiKey ?? requireEnv("TRIBUTARY_API_KEY"),
      baseURL: options.baseUrl ?? crossPlatformEnv("TRIBUTARY_BASE_URL") ?? "https://api.tributary.cc/openai/v1",
    },
    pdfSupport: options.pdfSupport ?? { mode: "text" },
    extraRequestBody: () => ({
      ...(options.reasoningEffort ? { reasoning_effort: options.reasoningEffort } : {}),
      ...(options.extraRequestBody ?? {}),
    }),
  });
}
