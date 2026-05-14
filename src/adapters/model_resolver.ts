import type { Adapter } from "./adapter.ts";
import { anthropicModel } from "./anthropic/adapter.ts";
import type { AnthropicModels } from "./anthropic/models.ts";
import { geminiModel } from "./gemini/adapter.ts";
import type { GoogleModels } from "./google_genai/models.ts";
import type { OpenAIModels } from "./openai/models.ts";
import type { OpenRouterModels } from "./openrouter/models.ts";
import { sidModel, type SidModels } from "./sid/adapter.ts";
import { tributaryModel, type TributaryModels } from "./tributary/adapter.ts";
import { vertexAIModel } from "./vertex_ai/adapter.ts";
import { openrouterModel } from "./openrouter/adapter.ts";
import { openAIModel } from "./openai/adapter.ts";

/**
 * A string shorthand for creating a model instance.
 *
 * Type-safe prefixes ensure autocomplete works for known model names:
 * - `"anthropic:claude-sonnet-4-5"`
 * - `"openai:gpt-4o"`
 * - `"gemini:gemini-2.5-pro"`
 * - `"vertex:gemini-2.5-pro"`
 * - `"openrouter:<any-model-path>"`
 * - `"tributary:<any-model>"`
 * - `"sid:sid-1"`
 */
export type ModelString =
  | `anthropic:${AnthropicModels}`
  | `openai:${OpenAIModels}`
  | `gemini:${GoogleModels}`
  | `vertex:${GoogleModels}`
  | `openrouter:${OpenRouterModels}`
  | `tributary:${TributaryModels}`
  | `sid:${SidModels}`;

/** A model instance or a string shorthand that can be resolved into one. */
export type AdapterLike = Adapter<unknown, unknown> | ModelString;

/**
 * Resolve a {@link ModelLike} value into a concrete {@link Model} instance.
 *
 * If the value is already a `Model`, it is returned as-is.
 * If it is a string in the form `"provider:model-name"`, the corresponding
 * model class is instantiated with default options.
 */
export function resolveModel(model: AdapterLike): Adapter<unknown, unknown> {
  if (typeof model !== "string") return model;

  const colonIndex = model.indexOf(":");
  if (colonIndex === -1) {
    throw new Error(
      `Invalid model string "${model}". Expected format "provider:model-name" (e.g. "anthropic:claude-sonnet-4-5", "openai:gpt-5.4-mini").`,
    );
  }

  const provider = model.slice(0, colonIndex) as ModelString extends `${infer P}:${string}` ? P : never;
  const modelName = model.slice(colonIndex + 1);

  switch (provider) {
    case "anthropic":
      return anthropicModel({ model: modelName as AnthropicModels });
    case "openai":
      return openAIModel({ model: modelName as OpenAIModels });
    case "gemini":
      return geminiModel({ model: modelName as GoogleModels });
    case "vertex":
      return vertexAIModel({ model: modelName as GoogleModels });
    case "openrouter":
      return openrouterModel({ model: modelName as OpenRouterModels });
    case "tributary":
      return tributaryModel({ model: modelName as TributaryModels });
    case "sid":
      return sidModel({ model: modelName as SidModels });
    default:
      provider satisfies never;
      throw new Error(`Unknown model provider "${provider}".`);
  }
}
