import type { ChatCompletionMessageParam, ChatCompletionMessageToolCall } from "openai/resources/chat/completions";
import { isStructuredOutputRetryFeedback, RETRY_RESUMABILITY_PROMPT } from "../../constants.ts";
import { normalizeToolName } from "../../tool.ts";
import type { ChatItem, ChatItemInputFile, ChatItemToolResultFile } from "../../types.ts";
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
import { serializeWrappedToolArguments } from "../shared/tools.ts";
import type { OpenAICompletionsToolMap } from "./tools.ts";

type FileHistoryItem = ChatItemInputFile | ChatItemToolResultFile;

export type OpenAICompletionsPdfSupportConfig = false | null | { mode: "native" | "text"; maxSize?: number };

export type OpenAICompletionsPdfSupport<TModel extends string> =
  | OpenAICompletionsPdfSupportConfig
  | ((model: TModel) => OpenAICompletionsPdfSupportConfig);

function getPdfSupport<TModel extends string>(
  pdfSupport: OpenAICompletionsPdfSupport<TModel> | undefined,
  model: TModel,
): Exclude<OpenAICompletionsPdfSupportConfig, false | null> | false {
  const resolved = typeof pdfSupport === "function" ? pdfSupport(model) : (pdfSupport ?? { mode: "native" as const });
  return resolved || false;
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

export async function getOpenAICompletionsHistory<TModel extends string>(options: {
  model: TModel;
  history: ChatItem[];
  instructions: string;
  normalizedTools: OpenAICompletionsToolMap[];
  signal: AbortSignal;
  supportedMimeTypes?: string[];
  pdfSupport?: OpenAICompletionsPdfSupport<TModel>;
}): Promise<ChatCompletionMessageParam[]> {
  const supportedMimeTypes = options.supportedMimeTypes ?? DEFAULT_SUPPORTED_MIME_TYPES;
  const messages: ChatCompletionMessageParam[] = [{
    role: "system",
    content: options.instructions,
  }];

  // DeepSeek rejects a replayed assistant turn that omits `reasoning_content`, so the
  // reasoning is buffered and re-attached to the assistant message the turn produced.
  // Providers that don't model reasoning this way ignore the extra field.
  let turnReasoning = "";

  // An assistant message carrying `tool_calls` must be followed by exactly one tool
  // message per call id, so a turn's consecutive tool uses are coalesced into one
  // assistant message and their results emitted as a contiguous block. File results
  // have no tool-role representation, so they trail the block as user messages.
  let toolTurn: {
    reasoning: string;
    calls: ChatCompletionMessageToolCall[];
    results: Map<string, string>;
    files: ChatCompletionMessageParam[];
  } | null = null;

  function flushToolTurn() {
    if (!toolTurn) return;
    // Unlike the text branch, `reasoning_content` is emitted even when empty: DeepSeek
    // rejects a tool-call turn that omits the field outright, and compaction can drop the
    // reasoning while keeping the use. Every other provider ignores an empty value.
    messages.push({
      role: "assistant",
      content: null,
      reasoning_content: toolTurn.reasoning,
      tool_calls: toolTurn.calls,
    } as ChatCompletionMessageParam);
    for (const call of toolTurn.calls) {
      messages.push({
        role: "tool",
        tool_call_id: call.id,
        content: toolTurn.results.get(call.id) ?? "",
      });
    }
    messages.push(...toolTurn.files);
    toolTurn = null;
    turnReasoning = "";
  }

  for (const historyItem of options.history) {
    switch (historyItem.type) {
      case "input_text":
        flushToolTurn();
        turnReasoning = "";
        messages.push({
          role: "user",
          content: historyItem.content,
        });
        break;
      case "output_text":
        flushToolTurn();
        if (isStructuredOutputRetryFeedback(historyItem.content)) {
          messages.push({ role: "user", content: historyItem.content });
        } else {
          messages.push({
            role: "assistant",
            content: historyItem.content,
            ...(turnReasoning ? { reasoning_content: turnReasoning } : {}),
          } as ChatCompletionMessageParam);
        }
        turnReasoning = "";
        break;
      case "output_reasoning":
        flushToolTurn();
        turnReasoning += historyItem.content;
        break;
      case "context_summary":
        flushToolTurn();
        turnReasoning = "";
        messages.push({
          role: "user",
          content: historyItem.content,
        });
        break;
      case "tool_use": {
        // A use arriving after results belongs to the next turn, not this one.
        if (toolTurn && toolTurn.results.size > 0) flushToolTurn();
        toolTurn ??= { reasoning: turnReasoning, calls: [], results: new Map(), files: [] };

        const tool = options.normalizedTools.find((candidate) => candidate.original.name === historyItem.kind);
        toolTurn.calls.push({
          id: historyItem.tool_use_id,
          type: "function",
          function: {
            name: tool?.openAI.function.name ?? normalizeToolName(historyItem.kind),
            arguments: serializeWrappedToolArguments(historyItem.content, tool),
          },
        });
        break;
      }
      case "tool_result_text":
        // A result whose use was compacted away cannot be paired; an unanswerable
        // tool message would be rejected outright, so drop it.
        toolTurn?.results.set(historyItem.tool_use_id, historyItem.content);
        break;
      case "tool_result_file": {
        if (!toolTurn) break;
        if (!toolTurn.results.has(historyItem.tool_use_id)) {
          toolTurn.results.set(historyItem.tool_use_id, "");
        }
        toolTurn.files.push(
          await getOpenAICompletionsFileMessage(
            options.model,
            historyItem,
            supportedMimeTypes,
            options.pdfSupport,
            options.signal,
          ),
        );
        break;
      }
      case "input_file":
        flushToolTurn();
        turnReasoning = "";
        messages.push(
          await getOpenAICompletionsFileMessage(
            options.model,
            historyItem,
            supportedMimeTypes,
            options.pdfSupport,
            options.signal,
          ),
        );
        break;
      default:
        historyItem satisfies never;
    }
  }

  flushToolTurn();

  const lastHistoryItem = options.history.at(-1);
  if (lastHistoryItem?.type === "output_text" && !isStructuredOutputRetryFeedback(lastHistoryItem.content)) {
    messages.push({
      role: "system",
      content: RETRY_RESUMABILITY_PROMPT,
    });
  }

  return messages;
}
