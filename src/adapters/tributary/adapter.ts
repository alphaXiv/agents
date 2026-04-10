import OpenAI from "openai";
import { crossPlatformEnv, requireEnv } from "../../util.ts";
import {
  OpenAICompletionsAdapter,
  type OpenAICompletionsAdapterOptions,
  type OpenAICompletionsClient,
  type OpenAICompletionsPdfSupport,
} from "../openai_completions/adapter.ts";
import type { TributaryModels } from "./models.ts";
import type { ReasoningEffort } from "openai/resources/shared";

export interface TributaryAdapterOptions<TModel extends TributaryModels>
  extends Omit<OpenAICompletionsAdapterOptions<TModel>, "client" | "name" | "extraRequestBody" | "pdfSupport"> {
  apiKey?: string;
  baseUrl?: string;
  client?: OpenAICompletionsClient;
  reasoningEffort?: ReasoningEffort;
  extraRequestBody?: Record<string, unknown>;
  pdfSupport?: OpenAICompletionsPdfSupport<TModel>;
}

export class TributaryAdapter<TModel extends TributaryModels> extends OpenAICompletionsAdapter<TModel> {
  constructor(options: TributaryAdapterOptions<TModel>) {
    super({
      ...options,
      name: "Tributary",
      client: options.client ?? new OpenAI({
        apiKey: options.apiKey ?? requireEnv("TRIBUTARY_API_KEY"),
        baseURL: options.baseUrl ?? crossPlatformEnv("TRIBUTARY_BASE_URL") ?? "https://api.tributary.cc/openai/v1",
      }),
      pdfSupport: options.pdfSupport ?? { mode: "text" },
      extraRequestBody: () => ({
        ...(options.reasoningEffort ? { reasoning_effort: options.reasoningEffort } : {}),
        ...(options.extraRequestBody ?? {}),
      }),
    });
  }
}
