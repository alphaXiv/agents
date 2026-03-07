import type z from "zod";
import { AnthropicAdapter } from "./adapters/anthropic.ts";
import { GoogleAdapter } from "./adapters/google.ts";
import { OpenAIAdapter } from "./adapters/openai.ts";
import { OpenRouterAdapter } from "./adapters/openrouter.ts";
import type { Tool } from "./tool.ts";
import type {
  AdapterStreamIterator,
  ChatItem,
  ReasoningEffort,
} from "./types.ts";
import { SidAdapter } from "./adapters/sid.ts";
import { TributaryAdapter } from "./adapters/tributary.ts";

export interface Adapter<zO, zI> {
  new (config: {
    model: string;
    output?: z.ZodType<zO, zI>;
    tools: Tool<unknown, unknown, unknown>[];
    reasoningEffort: ReasoningEffort;
  }): AdapterInstance;
}

export interface AdapterInstance {
  stream(params: {
    systemPrompt: string;
    history: ChatItem[];
    signal: AbortSignal;
  }): AdapterStreamIterator;
}

export const ADAPTERS: Record<string, Adapter<unknown, unknown>> = {
  "openai": OpenAIAdapter,
  "google": GoogleAdapter,
  "anthropic": AnthropicAdapter,
  "openrouter": OpenRouterAdapter,
  "tributary": TributaryAdapter,
  "sid": SidAdapter,
};

/** Use to register an adapter for an unsupported provider */
export function registerAdapter(
  prefix: string,
  adapter: Adapter<unknown, unknown>,
) {
  ADAPTERS[prefix] = adapter;
}
