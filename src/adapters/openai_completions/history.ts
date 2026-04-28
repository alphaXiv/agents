import type { ChatCompletionMessageParam } from "openai/resources/chat/completions";
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

  for (const historyItem of options.history) {
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
        const tool = options.normalizedTools.find((candidate) => candidate.original.name === historyItem.kind);
        messages.push({
          role: "assistant",
          content: null,
          tool_calls: [{
            id: historyItem.tool_use_id,
            type: "function",
            function: {
              name: tool?.openAI.function.name ?? normalizeToolName(historyItem.kind),
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

  const lastHistoryItem = options.history.at(-1);
  if (lastHistoryItem?.type === "output_text" && !isStructuredOutputRetryFeedback(lastHistoryItem.content)) {
    messages.push({
      role: "system",
      content: RETRY_RESUMABILITY_PROMPT,
    });
  }

  return messages;
}
