import type z from "zod";
import { anthropicAdapter } from "./adapters/anthropic.ts";
import { getOpenRouterAdapter } from "./adapters/openrouter.ts";
import { openAiResponsesAdapter } from "./adapters/openai-responses.ts";
import type { Tool } from "./tool.ts";
import type {
  AdapterStreamIterator,
  AdapterStreamSingleResult,
  ChatItem,
  ReasoningEffort,
} from "./types.ts";
import { crossPlatformEnv } from "./util.ts";
import { openAiCompletionsAdapter } from "./adapters/openai-completions.ts";

export interface AdapterStreamOptions<
  zO,
  zI,
  Model extends string = string,
> {
  /** The model specified in the agent */
  model: Model;
  /** Structured output schema */
  output: z.ZodType<zO, zI> | undefined;
  /** Available tool definition. Do not call the tool implementations yourself. */
  tools: Tool<unknown, unknown, unknown>[];
  reasoningEffort: ReasoningEffort;
  /** Primary instructions / developer prompt / system prompt */
  systemPrompt: string;
  /** Previous conversation history */
  history: ChatItem[];
  /** Cancellation signal */
  signal: AbortSignal;
}

export interface AdapterTypeOptions<Models extends string> {
  /**
   * An advisory list of models. This is used only to set the TypeScript type
   * of the upstream, and does not influence runtime at all.
   */
  models?: Models[];
}

export interface Adapter<Model extends string = string> {
  name: string;

  stream<zO, zI>(
    config: AdapterStreamOptions<zO, zI, Model>,
  ): AdapterStreamIterator | Promise<AdapterStreamSingleResult>;

  // TODO: shouldRetry?(err: unknown): boolean;
}

function requireEnv(name: string) {
  const value = crossPlatformEnv(name);
  if (!value) {
    throw new Error(`Missing required environment variable ${name}`);
  }
  return value;
}

export const ADAPTERS: Record<string, () => Adapter | Promise<Adapter>> = {
  "openai": () =>
    openAiResponsesAdapter({
      url: crossPlatformEnv("OPENAI_BASE_URL") ?? "https://api.openai.com/v1",
      apiKey: requireEnv("OPENAI_API_KEY"),
    }),
  "google": () =>
    // google's sdk reads random env vars which makes permission prompts in deno anoying
    import("./adapters/google.ts")
      .then(({ googleAdapter }) =>
        googleAdapter({
          url: crossPlatformEnv("GOOGLE_BASE_URL") ??
            "https://generativelanguage.googleapis.com",
          apiKey: requireEnv("GEMINI_API_KEY"),
        })
      ),
  "anthropic": () =>
    anthropicAdapter({
      url: crossPlatformEnv("ANTHROPIC_BASE_URL") ??
        "https://api.anthropic.com",
      apiKey: requireEnv("ANTHROPIC_API_KEY"),
    }),
  "openrouter": getOpenRouterAdapter,
  "tributary": () =>
    openAiCompletionsAdapter({
      name: "tributary",
      url: crossPlatformEnv("TRIBUTARY_BASE_URL") ??
        "https://api.tributary.cc/openai/v1",
      apiKey: requireEnv("TRIBUTARY_API_KEY"),

      pdfSupport: {
        mode: "text",
      },
    }),
  "sid": () =>
    openAiCompletionsAdapter({
      name: "sid",
      url: crossPlatformEnv("SID_BASE_URL") ?? "https://api.sid-1.com/v1",
      apiKey: requireEnv("SID_API_KEY"),
      supportedMimeTypes: [], // disable files
    }),
};
