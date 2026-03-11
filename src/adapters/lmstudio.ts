/**
 * Adapter implementation for LM Studio's OpenAI-compatible API using the `openai` package.
 * ```ts
 * const adapter = lmStudioAdapter({
 *   name: "localhost lmstudio",
 *   url: "http://localhost:1234/v1",
 *   apiKey: null, // lmstudio defaults to unauthenticated
 * });
 *
 * const agent = new Agent({
 *   adapter,
 *   model: "qwen/qwen3.5-35b-a3b"
 * });
 * ```
 * @module
 */
import type { Adapter, AdapterTypeOptions } from "../adapters.ts";
import {
  openAiResponsesAdapter,
  type OpenaiResponsesAdapterOptions,
} from "./openai-responses.ts";

export function lmStudioAdapter<Models extends string>(
  options:
    & Omit<OpenaiResponsesAdapterOptions, "apiKey">
    & AdapterTypeOptions<Models>
    & {
      /** API Key, if required by the server */
      apiKey: string | null;
    },
): Adapter<Models> {
  return openAiResponsesAdapter({
    // LM Studio doesn't correctly support `custom_tool` and `custom_tool_output`
    toolCallFlavor: "function",

    ...options,
    apiKey: options.apiKey ?? "N/A",
  });
}
