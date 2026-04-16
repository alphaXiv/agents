import { assert } from "@std/assert";
import type OpenAI from "openai";
import type {
  ResponseCreateParamsStreaming,
  ResponseFunctionToolCallOutputItem,
  ResponseInputFile,
  ResponseInputImage,
  ResponseInputItem,
  ResponseInputText,
  ResponseOutputMessage,
} from "openai/resources/responses/responses";
import type { ReasoningEffort } from "openai/resources/shared";

import { isStructuredOutputRetryFeedback, RETRY_RESUMABILITY_PROMPT } from "../../constants.ts";
import type { ClassifiedError } from "../../errors.ts";
import type { AdapterStreamIterator, ChatItem } from "../../types.ts";
import { Adapter, type AdapterStreamOptions } from "../adapter.ts";
import { classifyOpenAIError } from "../shared/classify_error.ts";
import {
  DEFAULT_SUPPORTED_MIME_TYPES,
  fetchRemoteFileAsDataUrl,
  fetchTextLikeFileAsTaggedText,
  getFileNameFromUrl,
  IMAGE_MIME_TYPES,
  isTextLikeMimeType,
  PDF_MIME_TYPE,
  supportsMimeType,
  unsupportedMediaTypeError,
} from "../shared/media.ts";
import { createOpenAICompatibleSchema } from "../shared/openai_compatibility.ts";
import { restoreWrappedToolArguments, serializeWrappedToolArguments } from "../shared/tools.ts";
import { normalizeOpenResponsesTools, type OpenResponsesToolMap } from "./tools.ts";

interface PendingToolCall {
  tool_use_id: string;
  kind: string;
  tool?: OpenResponsesToolMap;
}

type FileHistoryItem = Extract<ChatItem, { type: "input_file" } | { type: "tool_result_file" }>;

export interface OpenResponsesReasoningConfig {
  effort?: ReasoningEffort;
  summary?: "auto" | "concise" | "detailed";
}

export type OpenResponsesServiceTier = "auto" | "default" | "flex" | "priority";

export interface OpenResponsesAdapterOptions<TModel extends string> {
  model: TModel;
  name: string;
  client: Pick<OpenAI, "responses">;
  reasoning?: OpenResponsesReasoningConfig;
  serviceTier?: OpenResponsesServiceTier;
  parallelToolCalls?: boolean;
  supportedMimeTypes?: string[];
}

type OpenResponsesStreamingRequest = ResponseCreateParamsStreaming & {
  service_tier?: OpenResponsesServiceTier;
};

function getSyntheticId(prefix: string) {
  return `${prefix}_${crypto.randomUUID().replaceAll("-", "")}`;
}

function createUserTextMessage(text: string, role: "user" | "developer" = "user"): ResponseInputItem {
  return {
    type: "message",
    role,
    status: "completed",
    content: [{ type: "input_text", text }],
  };
}

function createAssistantTextMessage(text: string): ResponseOutputMessage {
  return {
    id: getSyntheticId("msg"),
    type: "message",
    role: "assistant",
    status: "completed",
    content: [{ type: "output_text", text, annotations: [] }],
  };
}

function getOrCreateFunctionCallOutput(
  history: ResponseInputItem[],
  toolUseId: string,
): ResponseFunctionToolCallOutputItem {
  const existing = history.find((item): item is ResponseFunctionToolCallOutputItem => {
    return item.type === "function_call_output" && item.call_id === toolUseId;
  });

  if (existing) {
    if (typeof existing.output === "string") {
      existing.output = [{ type: "input_text", text: existing.output }];
    }
    return existing;
  }

  const output: ResponseFunctionToolCallOutputItem = {
    id: getSyntheticId("fco"),
    type: "function_call_output",
    call_id: toolUseId,
    status: "completed",
    output: [],
  };
  history.push(output);
  return output;
}

async function getOpenResponsesFileInput(
  model: string,
  historyItem: FileHistoryItem,
  supportedMimeTypes: string[],
  signal: AbortSignal,
): Promise<ResponseInputText | ResponseInputImage | ResponseInputFile> {
  if (!supportsMimeType(historyItem.kind, supportedMimeTypes)) {
    throw unsupportedMediaTypeError(model, historyItem.kind);
  }

  if (IMAGE_MIME_TYPES.some((mimeType) => mimeType === historyItem.kind)) {
    return {
      type: "input_image",
      image_url: historyItem.content,
      detail: "auto",
    };
  }

  if (isTextLikeMimeType(historyItem.kind)) {
    return {
      type: "input_text",
      text: await fetchTextLikeFileAsTaggedText(historyItem.content, historyItem.kind, signal),
    };
  }

  if (historyItem.kind === PDF_MIME_TYPE) {
    return {
      type: "input_file",
      file_data: await fetchRemoteFileAsDataUrl(historyItem.content, historyItem.kind, signal),
      filename: getFileNameFromUrl(historyItem.content),
    };
  }

  return {
    type: "input_file",
    file_url: historyItem.content,
    filename: getFileNameFromUrl(historyItem.content),
  };
}

export class OpenResponsesAdapter<TModel extends string> extends Adapter<TModel> {
  name: string;
  #client: Pick<OpenAI, "responses">;
  #reasoning?: OpenResponsesReasoningConfig;
  #serviceTier?: OpenResponsesServiceTier;
  #parallelToolCalls: boolean;
  #supportedMimeTypes: string[];

  constructor(options: OpenResponsesAdapterOptions<TModel>) {
    super(options);
    this.name = options.name;
    this.#client = options.client;
    this.#reasoning = options.reasoning;
    this.#serviceTier = options.serviceTier;
    this.#parallelToolCalls = options.parallelToolCalls ?? true;
    this.#supportedMimeTypes = options.supportedMimeTypes ?? DEFAULT_SUPPORTED_MIME_TYPES;
  }

  override classifyError(error: unknown): ClassifiedError | null {
    return classifyOpenAIError(error);
  }

  async getHistory(
    history: ChatItem[],
    normalizedTools: OpenResponsesToolMap[],
    signal: AbortSignal,
  ): Promise<ResponseInputItem[]> {
    const responseHistory: ResponseInputItem[] = [];

    for (const historyItem of history) {
      switch (historyItem.type) {
        case "input_text":
          responseHistory.push(createUserTextMessage(historyItem.content));
          break;
        case "output_text":
          responseHistory.push(
            isStructuredOutputRetryFeedback(historyItem.content)
              ? createUserTextMessage(historyItem.content)
              : createAssistantTextMessage(historyItem.content),
          );
          break;
        case "output_reasoning":
          // Responses API expects provider-issued reasoning item ids on replay.
          // We only persist the text summary, so skip it rather than sending fake ids.
          break;
        case "context_summary":
          responseHistory.push(createUserTextMessage(historyItem.content));
          break;
        case "tool_use": {
          const tool = normalizedTools.find((candidate) => candidate.original.name === historyItem.kind);
          responseHistory.push({
            id: getSyntheticId("fc"),
            type: "function_call",
            status: "completed",
            call_id: historyItem.tool_use_id,
            name: tool?.openResponses.name ?? historyItem.kind,
            arguments: serializeWrappedToolArguments(historyItem.content, tool),
          });
          break;
        }
        case "tool_result_text": {
          const output = getOrCreateFunctionCallOutput(responseHistory, historyItem.tool_use_id);
          assert(typeof output.output !== "string");
          output.output.push({ type: "input_text", text: historyItem.content });
          break;
        }
        case "tool_result_file": {
          const output = getOrCreateFunctionCallOutput(responseHistory, historyItem.tool_use_id);
          assert(typeof output.output !== "string");
          output.output.push(
            await getOpenResponsesFileInput(this.model, historyItem, this.#supportedMimeTypes, signal),
          );
          break;
        }
        case "input_file":
          responseHistory.push({
            type: "message",
            role: "user",
            status: "completed",
            content: [await getOpenResponsesFileInput(this.model, historyItem, this.#supportedMimeTypes, signal)],
          });
          break;
        default:
          historyItem satisfies never;
      }
    }

    const lastHistoryItem = history.at(-1);
    if (lastHistoryItem?.type === "output_text" && !isStructuredOutputRetryFeedback(lastHistoryItem.content)) {
      responseHistory.push(createUserTextMessage(RETRY_RESUMABILITY_PROMPT, "developer"));
    }

    return responseHistory;
  }

  async *stream<zO, zI>(
    { history, instructions, tools, signal, output }: AdapterStreamOptions<zO, zI>,
  ): AdapterStreamIterator {
    const normalizedTools = normalizeOpenResponsesTools(tools);
    const structuredOutput = output && createOpenAICompatibleSchema(output, {
      kind: "output",
      rootPath: "output",
    });
    const shouldRestoreStructuredOutput = structuredOutput?.requiresValueTransformation ?? false;
    const responseHistory = await this.getHistory(history, normalizedTools, signal);
    const pendingToolCallsByOutputIndex: PendingToolCall[] = [];
    const pendingToolCallsByItemId: Record<string, PendingToolCall> = {};
    const pendingStructuredOutput: string[] = [];

    const request: OpenResponsesStreamingRequest = {
      model: this.model,
      input: responseHistory,
      instructions: structuredOutput?.instructions
        ? `${structuredOutput.instructions}\n\n${instructions}`
        : instructions,
      parallel_tool_calls: this.#parallelToolCalls,
      service_tier: this.#serviceTier,
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
      reasoning: this.#reasoning,
      stream: true,
    };

    const response = this.#client.responses.stream(request, { signal });

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
  }
}
