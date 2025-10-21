import { TestingAdapter } from "./adapters/__testing.ts";
import { AnthropicAdapter } from "./adapters/anthropic.ts";
import { GoogleAdapter } from "./adapters/google.ts";
import { OpenAIAdapter } from "./adapters/openai.ts";
import { OpenRouterAdapter } from "./adapters/openrouter.ts";
import type { Tool } from "./tool.ts";
import type { ChatItem, ZodSchemaType } from "./types.ts";

export interface Adapter<O = unknown> {
  new (config: {
    model: string;
    output?: ZodSchemaType<O>;
    tools: Tool<unknown>[];
  }): AdapterInstance;
}

export interface AdapterInstance {
  run(params: {
    systemPrompt: string;
    history: ChatItem[];
  }): Promise<ChatItem[]>;
}

export const ADAPTERS: Record<string, Adapter> = {
  "__testing": TestingAdapter,
  "openai": OpenAIAdapter,
  "google": GoogleAdapter,
  "anthropic": AnthropicAdapter,
  "openrouter": OpenRouterAdapter,
};
