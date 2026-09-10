import { delay } from "@std/async/delay";
import OpenAI from "openai";
import type { ResponseCreateParamsStreaming } from "openai/resources/responses/responses";
import type { ReasoningEffort } from "openai/resources/shared";

import type { AdapterStreamIterator, ProviderFileStore } from "../../types.ts";
import { errMessage } from "../../util.ts";
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

export type OpenResponsesClient = Pick<OpenAI, "responses"> & Partial<Pick<OpenAI, "files">>;

const FILE_SEND_ATTEMPTS = 3;
/** Pause before each resend. A provider may reject a just-uploaded id before it has indexed it. */
const FILE_RESEND_DELAYS_MS = [500, 1500];

/**
 * Sends images and PDFs as provider `file_id`s instead of URLs or base64, uploading on a store miss.
 * Built by the provider adapter, which knows the purpose and expiry its provider accepts.
 */
export interface OpenResponsesFilesConfig {
  purpose: "user_data" | "assistants";
  /** Provider side expiry, sent on upload as a backup to the store's own. Azure rejects it. */
  expiresAfterSeconds?: number;
  store: ProviderFileStore;
}

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
  files?: OpenResponsesFilesConfig;
}): Adapter<zO, zI> {
  const client = options.client ?? new OpenAI(options.openAIOptions);
  if (options.files && !client.files) {
    throw new Error("openResponsesModel: the `files` option needs a client with a `files` resource");
  }
  const files = options.files && { ...options.files, client: client.files! };
  const parallelToolCalls = options.parallelToolCalls ?? true;
  const supportedMimeTypes = options.supportedMimeTypes ?? DEFAULT_SUPPORTED_MIME_TYPES;

  // Only the live tool loop benefits from replaying reasoning, and that always runs in one process,
  // so a bounded cache is enough.
  // A miss falls back to synthetic ids, which is what every replay did before this existed.
  //
  // Reasoning is replayed as the encrypted blob the endpoint returned, never by id alone.
  // An id is looked up in the endpoint's response store, and on Azure the store may not hold
  // the response yet when the next request arrives.
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
      const uploadedFileIds = new Set<string>();
      async function* sendOnce(sentFileIds: Map<string, string>): AdapterStreamIterator {
        const responseHistory = await getOpenResponsesHistory({
          model: options.model,
          history,
          normalizedTools,
          signal,
          supportedMimeTypes,
          toolCallReplays,
          files: files && { ...files, sent: sentFileIds, uploaded: uploadedFileIds },
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
          include: ["reasoning.encrypted_content"],
          stream: true,
        };

        yield { type: "request_start" };
        const response = client.responses.stream(request, { signal });

        // Provider indices are per output item, so consecutive summary parts of one reasoning
        // item share an output_index and would collapse into a single block. Allocate our own
        // index per (output item, summary part) instead, in the order the events arrive.
        const streamIndices = new Map<string, number>();
        let nextStreamIndex = 0;
        let openReasoningItemId: string | undefined;
        const encryptedReasoning = new Map<string, string>();
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
            case "response.output_item.done":
              if (part.item.type === "reasoning" && part.item.encrypted_content) {
                encryptedReasoning.set(part.item.id, part.item.encrypted_content);
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

              // A preamble message between the reasoning item and the call is fine.
              // The API pairs a call with any reasoning item from its response, not strictly the preceding one.
              const encryptedContent = openReasoningItemId && encryptedReasoning.get(openReasoningItemId);
              if (openReasoningItemId && encryptedContent) {
                rememberToolCallReplay(toolUseId, {
                  reasoningItemId: openReasoningItemId,
                  encryptedContent,
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
      }

      for (let attempt = 1;; attempt++) {
        // A fresh map per attempt, so only the ids of this send can match a rejection.
        const sentFileIds = new Map<string, string>();
        try {
          return yield* sendOnce(sentFileIds);
        } catch (error) {
          // Matching on the id rather than the wording still works when a provider changes its message,
          // and the attempt cap makes a wrong match cost a few extra uploads at most.
          const message = errMessage(error);
          const rejected = [...sentFileIds].filter(([, fileId]) => message.includes(fileId));
          if (rejected.length === 0) throw error;

          for (const [url, fileId] of rejected) {
            // A just-uploaded id may not be indexed yet, so it keeps its row and is resent after the pause.
            if (!uploadedFileIds.has(fileId)) await options.files?.store.delete(url, fileId);
          }
          if (attempt === FILE_SEND_ATTEMPTS) throw error;
          yield {
            type: "log",
            message: `Provider rejected file ids ${
              rejected.map(([, fileId]) => fileId).join(", ")
            } on attempt ${attempt}`,
            error,
          };
          await delay(FILE_RESEND_DELAYS_MS[attempt - 1], { signal });
        }
      }
    },
    classifyError: classifyOpenAIError,
  };
}
