import OpenAI from "openai";
import { crossPlatformEnv, requireEnv } from "../../util.ts";
import {
  OpenAICompletionsAdapter,
  type OpenAICompletionsAdapterOptions,
  type OpenAICompletionsClient,
} from "../openai_completions/adapter.ts";

export type SidModels = "sid-1";

export interface SidAdapterOptions<TModel extends SidModels>
  extends Omit<OpenAICompletionsAdapterOptions<TModel>, "client" | "name" | "extraRequestBody"> {
  apiKey?: string;
  baseUrl?: string;
  client?: OpenAICompletionsClient;
  extraRequestBody?: Record<string, unknown>;
}

export class SidAdapter<TModel extends SidModels> extends OpenAICompletionsAdapter<TModel> {
  constructor(options: SidAdapterOptions<TModel>) {
    super({
      ...options,
      name: "SID",
      client: options.client ?? new OpenAI({
        apiKey: options.apiKey ?? requireEnv("SID_API_KEY"),
        baseURL: options.baseUrl ?? crossPlatformEnv("SID_BASE_URL") ?? "https://api.sid-1.com/v1",
      }),
      supportedMimeTypes: options.supportedMimeTypes ?? [],
      extraRequestBody: { tool_choice: "required", ...(options.extraRequestBody ?? {}) },
    });
  }
}
