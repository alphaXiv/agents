import OpenAI from "openai";
import type { ResponseCreateParamsStreaming } from "openai/resources/responses/responses";
import type { ReasoningEffort } from "openai/resources/shared";

import type { AdapterStreamIterator } from "../../types.ts";
import type { Adapter, AdapterStreamOptions } from "../adapter.ts";
import { classifyOpenAIError } from "../shared/classify_error.ts";
import { DEFAULT_SUPPORTED_MIME_TYPES } from "../shared/media.ts";
import { createOpenAICompatibleSchema } from "../shared/openai_compatibility.ts";
import { restoreWrappedToolArguments } from "../shared/tools.ts";
import { splitCacheInclusiveUsage } from "../shared/usage.ts";
import { getOpenResponsesHistory, type ToolCallReplay } from "./history.ts";
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

const TOOL_CALL_REPLAY_LIMIT = 2048;

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

  // Only the live tool loop benefits from replaying provider ids, and that always runs in one
  // process, so a bounded cache is enough. A miss falls back to synthetic ids, which is what
  // every replay did before this existed.
  //
  // Replaying an id assumes the endpoint still holds the response it came from. That is the
  // default on OpenAI; an endpoint that does not retain responses should pass `client`.
  const toolCallReplays = new Map<string, ToolCallReplay>();
  const rememberToolCallReplay = (toolUseId: string, replay: ToolCallReplay) => {
    if (!toolCallReplays.has(toolUseId) && toolCallReplays.size >= TOOL_CALL_REPLAY_LIMIT) {
      const oldest = toolCallReplays.keys().next();
      if (!oldest.done) toolCallReplays.delete(oldest.value);
    }
    toolCallReplays.set(toolUseId, replay);
  };

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
        toolCallReplays,
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

      // Provider indices are per output item, so consecutive summary parts of one reasoning
      // item share an output_index and would collapse into a single block. Allocate our own
      // index per (output item, summary part) instead, in the order the events arrive.
      const streamIndices = new Map<string, number>();
      let nextStreamIndex = 0;
      let openReasoningItemId: string | undefined;
      const streamIndex = (key: string) => {
        const existing = streamIndices.get(key);
        if (existing !== undefined) return existing;
        streamIndices.set(key, nextStreamIndex);
        return nextStreamIndex++;
      };

      for await (const part of response) {
        switch (part.type) {
          case "response.output_text.delta":
          case "response.refusal.delta":
            if (part.delta) {
              if (shouldRestoreStructuredOutput) {
                // Claim the index now so the restored text keeps its position relative to
                // reasoning and tool blocks, even though it is only emitted once the stream ends.
                streamIndex(`text:${part.output_index}`);
                pendingStructuredOutput[part.output_index] ??= "";
                pendingStructuredOutput[part.output_index] += part.delta;
              } else {
                yield {
                  type: "delta_output_text",
                  delta: part.delta,
                  index: streamIndex(`text:${part.output_index}`),
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
                index: streamIndex(
                  "summary_index" in part
                    ? `reasoning:${part.output_index}:summary:${part.summary_index}`
                    : `reasoning:${part.output_index}:content:${part.content_index}`,
                ),
              };
            }
            break;
          case "response.output_item.added": {
            const item = part.item;
            if (item.type === "reasoning") {
              openReasoningItemId = item.id;
              // Peek rather than allocate: this item never becomes a ChatItem, and downstream
              // indices are derived from item counts, so consuming one would leave a gap that
              // collides a later tool result with the block before it.
              yield { type: "reasoning_start", index: nextStreamIndex };
              break;
            }
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
              index: streamIndex(`tool:${part.output_index}`),
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
                index: streamIndex(`tool:${part.output_index}`),
                tool_use_id: toolUseId,
                kind,
              };
            }

            // Recorded only once the call is complete, since ids from an abandoned response may
            // never have been stored. A preamble message between the two is fine; the API pairs a
            // call with any reasoning item from its response, not strictly the preceding one.
            if (openReasoningItemId) {
              rememberToolCallReplay(toolUseId, {
                callItemId: part.item_id,
                reasoningItemId: openReasoningItemId,
              });
            }

            yield {
              type: "tool_use",
              index: streamIndex(`tool:${part.output_index}`),
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
            index: streamIndex(`text:${index}`),
          };
        }
      }

      const final = await response.finalResponse();
      return {
        ...splitCacheInclusiveUsage(
          final.usage?.input_tokens,
          final.usage?.input_tokens_details?.cached_tokens,
          final.usage?.input_tokens_details?.cache_write_tokens,
        ),
        outputTokens: final.usage?.output_tokens ?? null,
      };
    },
    classifyError: classifyOpenAIError,
  };
}
