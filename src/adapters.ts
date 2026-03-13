import type z from "zod";
import type { Tool } from "./tool.ts";
import type {
  AdapterStreamIterator,
  Awaitable,
  ChatItem,
  ReasoningEffort,
} from "./types.ts";
import { crossPlatformEnv, requireEnv } from "./util.ts";

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
  /** Name of the provider. Shown in traces and error messages. */
  name: string;

  stream<zO, zI>(
    config: AdapterStreamOptions<zO, zI, Model>,
  ): Awaitable<AdapterStreamIterator>;
}

export const ADAPTERS: Record<string, () => Promise<Adapter>> = {
  "openai": () =>
    import("./adapters/openai-responses.ts")
      .then(({ openAiResponsesAdapter }) =>
        openAiResponsesAdapter({
          name: "openai",
          url: crossPlatformEnv("OPENAI_BASE_URL") ??
            "https://api.openai.com/v1",
          apiKey: requireEnv("OPENAI_API_KEY"),
        })
      ),
  "google": () =>
    // google's sdk reads random env vars which makes permission prompts in deno annoying
    import("./adapters/google.ts")
      .then(({ googleAdapter }) =>
        googleAdapter({
          name: "google",
          url: crossPlatformEnv("GOOGLE_BASE_URL") ??
            "https://generativelanguage.googleapis.com",
          apiKey: requireEnv("GEMINI_API_KEY"),
        })
      ),
  "anthropic": () =>
    import("./adapters/anthropic/anthropic.ts")
      .then(({ anthropicAdapter }) =>
        anthropicAdapter({
          name: "anthropic",
          url: crossPlatformEnv("ANTHROPIC_BASE_URL") ??
            "https://api.anthropic.com",
          apiKey: requireEnv("ANTHROPIC_API_KEY"),
        })
      ),
  "openrouter": () =>
    import("./adapters/openrouter.ts")
      .then(({ getOpenRouterAdapter }) => getOpenRouterAdapter()),
  "tributary": () =>
    import("./adapters/openai-completions.ts")
      .then(({ openAiCompletionsAdapter }) =>
        openAiCompletionsAdapter({
          name: "tributary",
          url: crossPlatformEnv("TRIBUTARY_BASE_URL") ??
            "https://api.tributary.cc/openai/v1",
          apiKey: requireEnv("TRIBUTARY_API_KEY"),

          pdfSupport: {
            mode: "text",
          },
        })
      ),
  "sid": () =>
    import("./adapters/openai-completions.ts")
      .then(({ openAiCompletionsAdapter }) =>
        openAiCompletionsAdapter({
          name: "sid",
          url: crossPlatformEnv("SID_BASE_URL") ?? "https://api.sid-1.com/v1",
          apiKey: requireEnv("SID_API_KEY"),
          supportedMimeTypes: [], // disable files
        })
      ),
};
