import OpenAI from "openai";
import { crossPlatformEnv, requireEnv } from "../../util.ts";
import {
  OpenAICompletionsAdapter,
  type OpenAICompletionsAdapterOptions,
  type OpenAICompletionsClient,
  type OpenAICompletionsPdfSupport,
} from "../openai_completions/adapter.ts";
import type { OpenRouterModels, OpenRouterReasoningEffort } from "./models.ts";
import { getOpenRouterNativePdfSupport } from "./models.ts";

export interface OpenRouterReasoningConfig {
  enabled?: boolean;
  effort?: OpenRouterReasoningEffort;
  max_tokens?: number;
  exclude?: boolean;
}

export interface OpenRouterPlugin {
  id: string;
  enabled?: boolean;
  [key: string]: unknown;
}

function getOpenRouterHeaders(options: {
  headers?: Record<string, string>;
  siteUrl?: string;
  appName?: string;
}) {
  return {
    ...(options.siteUrl ? { "HTTP-Referer": options.siteUrl } : {}),
    ...(options.appName ? { "X-OpenRouter-Title": options.appName } : {}),
    ...options.headers,
  };
}

export interface OpenRouterAdapterOptions<TModel extends OpenRouterModels>
  extends Omit<OpenAICompletionsAdapterOptions<TModel>, "client" | "name" | "extraRequestBody" | "pdfSupport"> {
  apiKey?: string;
  baseUrl?: string;
  client?: OpenAICompletionsClient;
  reasoning?: OpenRouterReasoningConfig;
  plugins?: OpenRouterPlugin[];
  provider?: Record<string, unknown>;
  models?: string[];
  route?: "fallback";
  headers?: Record<string, string>;
  siteUrl?: string;
  appName?: string;
  extraRequestBody?: Record<string, unknown>;
  pdfSupport?: OpenAICompletionsPdfSupport<TModel>;
}

export class OpenRouterAdapter<TModel extends OpenRouterModels> extends OpenAICompletionsAdapter<TModel> {
  constructor(options: OpenRouterAdapterOptions<TModel>) {
    super({
      ...options,
      name: "OpenRouter",
      client: options.client ?? new OpenAI({
        apiKey: options.apiKey ?? requireEnv("OPENROUTER_API_KEY"),
        baseURL: options.baseUrl ?? crossPlatformEnv("OPENROUTER_BASE_URL") ?? "https://openrouter.ai/api/v1",
        defaultHeaders: getOpenRouterHeaders(options),
      }),
      pdfSupport: options.pdfSupport ??
        ((model) =>
          getOpenRouterNativePdfSupport(model) ? { mode: "native", maxSize: 4 * 1024 * 1024 } : { mode: "text" }),
      extraRequestBody: () => ({
        ...(options.reasoning ? { reasoning: options.reasoning } : {}),
        ...(options.plugins?.length ? { plugins: options.plugins } : {}),
        ...(options.provider ? { provider: options.provider } : {}),
        ...(options.models?.length ? { models: options.models } : {}),
        ...(options.route ? { route: options.route } : {}),
        ...(options.extraRequestBody ?? {}),
      }),
    });
  }
}
