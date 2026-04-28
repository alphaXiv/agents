import OpenAI from "openai";
import type { ResponseCreateParamsStreaming } from "openai/resources/responses/responses";
import type { ReasoningEffort } from "openai/resources/shared";

import type { AdapterStreamIterator } from "../../types.ts";
import type { Adapter, AdapterStreamOptions } from "../adapter.ts";
import { classifyOpenAIError } from "../shared/classify_error.ts";
import { DEFAULT_SUPPORTED_MIME_TYPES } from "../shared/media.ts";
import { createOpenAICompatibleSchema } from "../shared/openai_compatibility.ts";
import { restoreWrappedToolArguments } from "../shared/tools.ts";
import { getOpenResponsesHistory } from "./history.ts";
import { normalizeOpenResponsesTools, type OpenResponsesToolMap } from "./tools.ts";
import type { ClientOptions } from "openai";

interface PendingToolCall {
  tool_use_id: string;
  kind: string;
  tool?: OpenResponsesToolMap;
}

export interface OpenResponsesReasoningConfig {
  effort?: ReasoningEffort;
  summary?: "auto" | "concise" | "detailed";
}

export type OpenResponsesServiceTier = "auto" | "default" | "flex" | "priority";

type OpenResponsesStreamingRequest = ResponseCreateParamsStreaming & {
  service_tier?: OpenResponsesServiceTier;
};

export type OpenResponsesClient = Pick<OpenAI, "responses">;

/** Generic adapter over an Open Responses compatible API */
export function openResponsesModel<zO, zI>(options: {
  model: string;
  provider?: string;
  openAIOptions?: ClientOptions;
  reasoning?: OpenResponsesReasoningConfig;
  serviceTier?: OpenResponsesServiceTier;
  parallelToolCalls?: boolean;
  supportedMimeTypes?: string[];
  client?: OpenResponsesClient;
}): Adapter<zO, zI> {
  const client = options.client ?? new OpenAI(options.openAIOptions);
  const parallelToolCalls = options.parallelToolCalls ?? true;
  const supportedMimeTypes = options.supportedMimeTypes ?? DEFAULT_SUPPORTED_MIME_TYPES;

  return {
    provider: options.provider ?? "OpenResponses",
    model: options.model,
    stream: async function* stream<zO, zI>(
      { history, instructions, tools, signal, output }: AdapterStreamOptions<zO, zI>,
    ): AdapterStreamIterator {
      const normalizedTools = normalizeOpenResponsesTools(tools);
      const structuredOutput = output && createOpenAICompatibleSchema(output, {
        kind: "output",
        rootPath: "output",
      });
      const shouldRestoreStructuredOutput = structuredOutput?.requiresValueTransformation ?? false;
      const responseHistory = await getOpenResponsesHistory({
        model: options.model,
        history,
        normalizedTools,
        signal,
        supportedMimeTypes,
      });
      const pendingToolCallsByOutputIndex: PendingToolCall[] = [];
      const pendingToolCallsByItemId: Record<string, PendingToolCall> = {};
      const pendingStructuredOutput: string[] = [];

      const request: OpenResponsesStreamingRequest = {
        model: options.model,
        input: responseHistory,
        instructions: structuredOutput?.instructions
          ? `${structuredOutput.instructions}\n\n${instructions}`
          : instructions,
        parallel_tool_calls: parallelToolCalls,
        service_tier: options.serviceTier,
        tools: normalizedTools.map((tool) => tool.openResponses),
        text: {
          format: structuredOutput
            ? {
              type: "json_schema",
              name: "output",
              strict: true,
              schema: structuredOutput.jsonSchema,
            }
            : { type: "text" },
        },
        reasoning: options.reasoning,
        stream: true,
      };

      const response = client.responses.stream(request, { signal });

      for await (const part of response) {
        switch (part.type) {
          case "response.output_text.delta":
          case "response.refusal.delta":
            if (part.delta) {
              if (shouldRestoreStructuredOutput) {
                pendingStructuredOutput[part.output_index] ??= "";
                pendingStructuredOutput[part.output_index] += part.delta;
              } else {
                yield {
                  type: "delta_output_text",
                  delta: part.delta,
                  index: part.output_index,
                };
              }
            }
            break;
          case "response.reasoning_summary_text.delta":
          case "response.reasoning_text.delta":
            if (part.delta) {
              yield {
                type: "delta_output_reasoning",
                delta: part.delta,
                index: part.output_index,
              };
            }
            break;
          case "response.output_item.added": {
            const item = part.item;
            if (item.type !== "function_call") break;

            const tool = normalizedTools.find((candidate) => candidate.openResponses.name === item.name);
            const pendingToolCall: PendingToolCall = {
              tool_use_id: item.call_id,
              kind: tool?.original.name ?? item.name,
              tool,
            };
            pendingToolCallsByOutputIndex[part.output_index] = pendingToolCall;
            if (item.id) {
              pendingToolCallsByItemId[item.id] = pendingToolCall;
            }
            yield {
              type: "tool_use_start",
              index: part.output_index,
              tool_use_id: pendingToolCall.tool_use_id,
              kind: pendingToolCall.kind,
            };
            break;
          }
          case "response.function_call_arguments.done": {
            const pendingToolCall = pendingToolCallsByItemId[part.item_id] ??
              pendingToolCallsByOutputIndex[part.output_index];
            const tool = pendingToolCall?.tool ??
              normalizedTools.find((candidate) => candidate.openResponses.name === part.name);
            const toolUseId = pendingToolCall?.tool_use_id ?? part.item_id;
            const kind = pendingToolCall?.kind ?? tool?.original.name ?? part.name;

            if (!pendingToolCall) {
              yield {
                type: "tool_use_start",
                index: part.output_index,
                tool_use_id: toolUseId,
                kind,
              };
            }

            yield {
              type: "tool_use",
              index: part.output_index,
              tool_use_id: toolUseId,
              kind,
              content: restoreWrappedToolArguments(part.arguments, tool),
            };
            break;
          }
        }
      }

      if (shouldRestoreStructuredOutput) {
        for (let index = 0; index < pendingStructuredOutput.length; index++) {
          const rawText = pendingStructuredOutput[index];
          if (!rawText) continue;

          let restoredText = rawText;
          try {
            restoredText = JSON.stringify(structuredOutput!.fromProvider(JSON.parse(rawText)));
          } catch {
            restoredText = rawText;
          }

          yield {
            type: "delta_output_text",
            delta: restoredText,
            index,
          };
        }
      }

      const final = await response.finalResponse();
      return {
        inputTokens: final.usage?.input_tokens ?? null,
        outputTokens: final.usage?.output_tokens ?? null,
      };
    },
    classifyError: classifyOpenAIError,
  };
}
