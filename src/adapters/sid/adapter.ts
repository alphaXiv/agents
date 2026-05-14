import { crossPlatformEnv, requireEnv } from "../../util.ts";
import { type OpenAICompletionsClient, openAICompletionsModel } from "../openai_completions/adapter.ts";
import type { Adapter } from "../adapter.ts";

export type SidModels = "sid-1";

export function sidModel<zO, zI, TModel extends SidModels>(options: {
  model: TModel;
  apiKey?: string;
  baseUrl?: string;
  client?: OpenAICompletionsClient;
  extraRequestBody?: Record<string, unknown>;
}): Adapter<zO, zI> {
  return openAICompletionsModel({
    provider: "SID",
    model: options.model,
    client: options.client,
    openAIOptions: options.client ? undefined : {
      apiKey: options.apiKey ?? requireEnv("SID_API_KEY"),
      baseURL: options.baseUrl ?? crossPlatformEnv("SID_BASE_URL") ?? "https://api.sid-1.com/v1",
    },
    supportedMimeTypes: [],
    extraRequestBody: { tool_choice: "required", ...(options.extraRequestBody ?? {}) },
  });
}
