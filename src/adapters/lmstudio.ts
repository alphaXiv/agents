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
