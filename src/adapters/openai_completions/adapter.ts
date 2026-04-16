import type OpenAI from "openai";
import type {
  ChatCompletionCreateParamsStreaming,
  ChatCompletionMessageParam,
} from "openai/resources/chat/completions";
import type z from "zod";
import { classifyOpenAIError } from "../shared/classify_error.ts";
import { isStructuredOutputRetryFeedback, RETRY_RESUMABILITY_PROMPT } from "../../constants.ts";
import type { AdapterStreamIterator, ChatItem, ChatItemInputFile, ChatItemToolResultFile } from "../../types.ts";
import { Adapter, type AdapterStreamOptions } from "../adapter.ts";
import {
  DEFAULT_SUPPORTED_MIME_TYPES,
  fetchPdfAsText,
  fetchRemoteFileAsDataUrl,
  fetchTextLikeFileAsTaggedText,
  getContentLength,
  getFileNameFromUrl,
  IMAGE_MIME_TYPES,
  isTextLikeMimeType,
  PDF_MIME_TYPE,
  supportsMimeType,
  unsupportedMediaTypeError,
} from "../shared/media.ts";
import { restoreWrappedToolArguments, serializeWrappedToolArguments } from "../shared/tools.ts";
import { normalizeOpenAICompletionsTools, type OpenAICompletionsToolMap } from "./tools.ts";
import type { ClassifiedError } from "../../errors.ts";

type FileHistoryItem = ChatItemInputFile | ChatItemToolResultFile;
export type OpenAICompletionsClient = Pick<OpenAI, "chat">;

export type OpenAICompletionsPdfSupportConfig = false | null | { mode: "native" | "text"; maxSize?: number };

export type OpenAICompletionsPdfSupport<TModel extends string> =
  | OpenAICompletionsPdfSupportConfig
  | ((model: TModel) => OpenAICompletionsPdfSupportConfig);

export interface OpenAICompletionsExtraRequestBodyArgs<TModel extends string, zO, zI> {
  model: TModel;
  output?: z.ZodType<zO, zI>;
  normalizedTools: OpenAICompletionsToolMap[];
}

export type OpenAICompletionsExtraRequestBody<TModel extends string> =
  | Record<string, unknown>
  | (<zO, zI>(args: OpenAICompletionsExtraRequestBodyArgs<TModel, zO, zI>) => Record<string, unknown>);

export interface OpenAICompletionsAdapterOptions<TModel extends string> {
  model: TModel;
  name: string;
  client: OpenAICompletionsClient;
  parallelToolCalls?: boolean;
  supportedMimeTypes?: string[];
  pdfSupport?: OpenAICompletionsPdfSupport<TModel>;
  extraRequestBody?: OpenAICompletionsExtraRequestBody<TModel>;
}

function getPdfSupport<TModel extends string>(
  pdfSupport: OpenAICompletionsPdfSupport<TModel> | undefined,
  model: TModel,
): Exclude<OpenAICompletionsPdfSupportConfig, false | null> | false {
  const resolved = typeof pdfSupport === "function" ? pdfSupport(model) : (pdfSupport ?? { mode: "native" as const });
  return resolved || false;
}

export function resolveOpenAICompletionsExtraRequestBody<TModel extends string, zO, zI>(
  extraRequestBody: OpenAICompletionsExtraRequestBody<TModel> | undefined,
  args: OpenAICompletionsExtraRequestBodyArgs<TModel, zO, zI>,
): Record<string, unknown> {
  if (!extraRequestBody) return {};
  return typeof extraRequestBody === "function" ? extraRequestBody(args) : extraRequestBody;
}

async function getOpenAICompletionsFileMessage<TModel extends string>(
  model: TModel,
  item: FileHistoryItem,
  supportedMimeTypes: string[],
  pdfSupport: OpenAICompletionsPdfSupport<TModel> | undefined,
  signal: AbortSignal,
): Promise<ChatCompletionMessageParam> {
  if (!supportsMimeType(item.kind, supportedMimeTypes)) {
    throw unsupportedMediaTypeError(model, item.kind);
  }

  if (IMAGE_MIME_TYPES.some((mimeType) => mimeType === item.kind)) {
    return {
      role: "user",
      content: [{
        type: "image_url",
        image_url: {
          url: item.content,
          detail: "auto",
        },
      }],
    };
  }

  if (isTextLikeMimeType(item.kind)) {
    return {
      role: "user",
      content: [{
        type: "text",
        text: await fetchTextLikeFileAsTaggedText(item.content, item.kind, signal),
      }],
    };
  }

  if (item.kind === PDF_MIME_TYPE) {
    const resolvedPdfSupport = getPdfSupport(pdfSupport, model);
    if (!resolvedPdfSupport) {
      throw unsupportedMediaTypeError(model, item.kind);
    }

    if (
      resolvedPdfSupport.mode === "text" ||
      (resolvedPdfSupport.maxSize !== undefined &&
        (await getContentLength(item.content, signal)) > resolvedPdfSupport.maxSize)
    ) {
      return {
        role: "user",
        content: [{
          type: "text",
          text: await fetchPdfAsText(item.content, signal),
        }],
      };
    }
  }

  return {
    role: "user",
    content: [{
      type: "file",
      file: {
        file_data: await fetchRemoteFileAsDataUrl(item.content, item.kind, signal),
        filename: getFileNameFromUrl(item.content),
      },
    }],
  };
}

export class OpenAICompletionsAdapter<TModel extends string> extends Adapter<TModel> {
  name: string;
  #client: OpenAICompletionsClient;
  #parallelToolCalls: boolean;
  #supportedMimeTypes: string[];
  #pdfSupport?: OpenAICompletionsPdfSupport<TModel>;
  #extraRequestBody?: OpenAICompletionsExtraRequestBody<TModel>;

  constructor(options: OpenAICompletionsAdapterOptions<TModel>) {
    super(options);

    const supportedMimeTypes = options.supportedMimeTypes ?? DEFAULT_SUPPORTED_MIME_TYPES;
    if (options.pdfSupport && !supportsMimeType("application/pdf", supportedMimeTypes)) {
      throw new Error("pdfSupport requires application/pdf to be included in supportedMimeTypes");
    }

    this.name = options.name;
    this.#client = options.client;
    this.#parallelToolCalls = options.parallelToolCalls ?? true;
    this.#supportedMimeTypes = supportedMimeTypes;
    this.#pdfSupport = options.pdfSupport;
    this.#extraRequestBody = options.extraRequestBody;
  }

  override classifyError(error: unknown): ClassifiedError | null {
    return classifyOpenAIError(error);
  }

  async getHistory(
    history: ChatItem[],
    instructions: string,
    normalizedTools: OpenAICompletionsToolMap[],
    signal: AbortSignal,
  ): Promise<ChatCompletionMessageParam[]> {
    const messages: ChatCompletionMessageParam[] = [{
      role: "system",
      content: instructions,
    }];

    for (const historyItem of history) {
      switch (historyItem.type) {
        case "input_text":
          messages.push({
            role: "user",
            content: historyItem.content,
          });
          break;
        case "output_text":
          messages.push({
            role: isStructuredOutputRetryFeedback(historyItem.content) ? "user" : "assistant",
            content: historyItem.content,
          });
          break;
        case "output_reasoning":
          break;
        case "context_summary":
          messages.push({
            role: "user",
            content: historyItem.content,
          });
          break;
        case "tool_use": {
          const tool = normalizedTools.find((candidate) => candidate.original.name === historyItem.kind);
          messages.push({
            role: "assistant",
            content: null,
            tool_calls: [{
              id: historyItem.tool_use_id,
              type: "function",
              function: {
                name: tool?.openAI.function.name ?? historyItem.kind,
                arguments: serializeWrappedToolArguments(historyItem.content, tool),
              },
            }],
          });
          break;
        }
        case "tool_result_text":
          messages.push({
            role: "tool",
            tool_call_id: historyItem.tool_use_id,
            content: historyItem.content,
          });
          break;
        case "input_file":
        case "tool_result_file":
          messages.push(
            await getOpenAICompletionsFileMessage(
              this.model,
              historyItem,
              this.#supportedMimeTypes,
              this.#pdfSupport,
              signal,
            ),
          );
          break;
        default:
          historyItem satisfies never;
      }
    }

    const lastHistoryItem = history.at(-1);
    if (lastHistoryItem?.type === "output_text" && !isStructuredOutputRetryFeedback(lastHistoryItem.content)) {
      messages.push({
        role: "system",
        content: RETRY_RESUMABILITY_PROMPT,
      });
    }

    return messages;
  }

  async *stream<zO, zI>(
    { history, instructions, tools, signal, output }: AdapterStreamOptions<zO, zI>,
  ): AdapterStreamIterator {
    const normalizedTools = normalizeOpenAICompletionsTools(tools);
    const messages = await this.getHistory(history, instructions, normalizedTools, signal);
    const toolByName = new Map(normalizedTools.map((tool) => [tool.openAI.function.name, tool]));

    const extraRequestBody = resolveOpenAICompletionsExtraRequestBody(this.#extraRequestBody, {
      model: this.model,
      output,
      normalizedTools,
    });

    const request: ChatCompletionCreateParamsStreaming = {
      model: this.model,
      messages,
      parallel_tool_calls: normalizedTools.length > 0 ? this.#parallelToolCalls : undefined,
      tools: normalizedTools.length > 0 ? normalizedTools.map((tool) => tool.openAI) : undefined,
      response_format: output
        ? {
          type: "json_schema",
          json_schema: {
            name: "output",
            strict: true,
            schema: output.toJSONSchema(),
          },
        }
        : { type: "text" },
      stream: true,
      ...extraRequestBody,
      stream_options: {
        ...(extraRequestBody.stream_options ?? {}),
        include_usage: true,
      },
    };

    const response = this.#client.chat.completions.stream(request, { signal });

    const pendingToolUses = new Map<number, {
      streamIndex: number | null;
      callId?: string;
      name?: string;
      content: string;
    }>();

    let lastType = "";
    let lastIndex = -1;

    for await (const part of response) {
      const choice = part.choices?.[0];
      if (!choice?.delta) continue;
      const delta = choice.delta as {
        content?: string | null;
        reasoning?: string | null;
        tool_calls?: Array<{
          index?: number;
          id?: string;
          function?: { name?: string; arguments?: string };
        }>;
      };

      if (delta.reasoning) {
        if (lastType !== "reasoning") {
          lastType = "reasoning";
          lastIndex++;
        }

        yield {
          type: "delta_output_reasoning",
          index: lastIndex,
          delta: delta.reasoning,
        };
      }

      if (delta.content) {
        if (lastType !== "text") {
          lastType = "text";
          lastIndex++;
        }

        yield {
          type: "delta_output_text",
          index: lastIndex,
          delta: delta.content,
        };
      }

      for (const call of delta.tool_calls ?? []) {
        const callIndex = call.index ?? 0;
        const pending = pendingToolUses.get(callIndex) ?? { streamIndex: null, content: "" };

        if (call.id) pending.callId = call.id;
        if (call.function?.name) pending.name = call.function.name;
        if (call.function?.arguments) pending.content += call.function.arguments;

        if (pending.streamIndex === null && pending.callId && pending.name) {
          const tool = toolByName.get(pending.name);
          pending.streamIndex = ++lastIndex;
          lastType = "tool_use";

          yield {
            type: "tool_use_start",
            index: pending.streamIndex,
            tool_use_id: pending.callId,
            kind: tool?.original.name ?? pending.name,
          };
        }

        pendingToolUses.set(callIndex, pending);
      }
    }

    for (
      const pending of [...pendingToolUses.values()].sort((left, right) =>
        (left.streamIndex ?? 0) - (right.streamIndex ?? 0)
      )
    ) {
      if (pending.streamIndex === null || !pending.callId || !pending.name) continue;
      const tool = toolByName.get(pending.name);

      yield {
        type: "tool_use",
        index: pending.streamIndex,
        tool_use_id: pending.callId,
        kind: tool?.original.name ?? pending.name,
        content: restoreWrappedToolArguments(pending.content, tool),
      };
    }

    const usage = await response.totalUsage();
    return {
      inputTokens: usage?.prompt_tokens ?? null,
      outputTokens: usage?.completion_tokens ?? null,
    };
  }
}
