import { crossPlatformEnv, requireEnv } from "../../util.ts";
import type { Adapter } from "../adapter.ts";
import {
  type OpenAICompletionsClient,
  openAICompletionsModel,
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

/**
 * OpenRouter reports an upstream 4xx as a flat "Provider returned error" and keeps the
 * provider's own message in `metadata.raw`. Folding it into the error message is what lets
 * the retry layer see a context overflow for what it is instead of an opaque client error.
 */
function describeOpenRouterProviderError(error: unknown): string | null {
  const body = (error as { error?: { message?: unknown; metadata?: { raw?: unknown } } } | null)?.error;
  if (typeof body?.message !== "string" || typeof body.metadata?.raw !== "string") return null;

  const raw = body.metadata.raw;
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return raw;
  }

  const parsedMessage = (parsed as { error?: { message?: unknown } })?.error?.message;
  return typeof parsedMessage === "string" ? parsedMessage : raw;
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

export function openrouterModel<zO, zI, TModel extends OpenRouterModels>(options: {
  model: TModel;
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
}): Adapter<zO, zI> {
  const adapter = openAICompletionsModel<zO, zI, TModel>({
    provider: "OpenRouter",
    model: options.model,
    client: options.client,
    openAIOptions: options.client ? undefined : {
      apiKey: options.apiKey ?? requireEnv("OPENROUTER_API_KEY"),
      baseURL: options.baseUrl ?? crossPlatformEnv("OPENROUTER_BASE_URL") ?? "https://openrouter.ai/api/v1",
      defaultHeaders: getOpenRouterHeaders(options),
    },
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

  return {
    ...adapter,
    stream: async function* (streamOptions) {
      try {
        return yield* adapter.stream(streamOptions);
      } catch (error) {
        const providerError = describeOpenRouterProviderError(error);
        // Rewritten in place so the error stays an `OpenAI.APIError` for classification.
        if (providerError && error instanceof Error) {
          error.message = `${error.message}: ${providerError}`;
        }
        throw error;
      }
    },
  };
}
