import { TestingAdapter } from "./adapters/__testing.ts";
import { OpenAIAdapter } from "./adapters/openai.ts";

export const ADAPTERS: Record<string, typeof TestingAdapter> = {
  "__testing": TestingAdapter,
  "openai": OpenAIAdapter,
};
