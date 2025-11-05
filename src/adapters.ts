import type z from "zod";
import { TestingAdapter } from "./adapters/__testing.ts";
import { AnthropicAdapter } from "./adapters/anthropic.ts";
import { GoogleAdapter } from "./adapters/google.ts";
import { OpenAIAdapter } from "./adapters/openai.ts";
import { OpenRouterAdapter } from "./adapters/openrouter.ts";
import type { Tool } from "./tool.ts";
import type {
  AsyncStreamItemGenerator,
  ChatItem,
  ReasoningEffort,
} from "./types.ts";

export interface Adapter<zO, zI> {
  new (config: {
    model: string;
    output?: z.ZodType<zO, zI>;
    tools: Tool<unknown, unknown>[];
    reasoningEffort: ReasoningEffort;
  }): AdapterInstance;
}

export interface AdapterInstance {
  run(params: {
    systemPrompt: string;
    history: ChatItem[];
    signal: AbortSignal;
  }): Promise<ChatItem[]>;

  stream(params: {
    systemPrompt: string;
    history: ChatItem[];
    signal: AbortSignal;
  }): AsyncStreamItemGenerator;
}

export const ADAPTERS: Record<string, Adapter<unknown, unknown>> = {
  "__testing": TestingAdapter,
  "openai": OpenAIAdapter,
  "google": GoogleAdapter,
  "anthropic": AnthropicAdapter,
  "openrouter": OpenRouterAdapter,
};
