import { TestingAdapter } from "./adapters/__testing.ts";
import { OpenAIAdapter } from "./adapters/openai.ts";
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
};
