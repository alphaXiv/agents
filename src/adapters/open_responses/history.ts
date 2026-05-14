import { assert } from "@std/assert";
import type {
  ResponseFunctionToolCallOutputItem,
  ResponseInputFile,
  ResponseInputImage,
  ResponseInputItem,
  ResponseInputText,
  ResponseOutputMessage,
} from "openai/resources/responses/responses";
import { isStructuredOutputRetryFeedback, RETRY_RESUMABILITY_PROMPT } from "../../constants.ts";
import { normalizeToolName } from "../../tool.ts";
import type { ChatItem } from "../../types.ts";
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
import { serializeWrappedToolArguments } from "../shared/tools.ts";
import type { OpenResponsesToolMap } from "./tools.ts";

type FileHistoryItem = Extract<ChatItem, { type: "input_file" } | { type: "tool_result_file" }>;

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

export async function getOpenResponsesHistory(options: {
  model: string;
  history: ChatItem[];
  normalizedTools: OpenResponsesToolMap[];
  signal: AbortSignal;
  supportedMimeTypes?: string[];
}): Promise<ResponseInputItem[]> {
  const supportedMimeTypes = options.supportedMimeTypes ?? DEFAULT_SUPPORTED_MIME_TYPES;
  const responseHistory: ResponseInputItem[] = [];

  for (const historyItem of options.history) {
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
        const tool = options.normalizedTools.find((candidate) => candidate.original.name === historyItem.kind);
        responseHistory.push({
          id: getSyntheticId("fc"),
          type: "function_call",
          status: "completed",
          call_id: historyItem.tool_use_id,
          name: tool?.openResponses.name ?? normalizeToolName(historyItem.kind),
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
          await getOpenResponsesFileInput(options.model, historyItem, supportedMimeTypes, options.signal),
        );
        break;
      }
      case "input_file":
        responseHistory.push({
          type: "message",
          role: "user",
          status: "completed",
          content: [await getOpenResponsesFileInput(options.model, historyItem, supportedMimeTypes, options.signal)],
        });
        break;
      default:
        historyItem satisfies never;
    }
  }

  const lastHistoryItem = options.history.at(-1);
  if (lastHistoryItem?.type === "output_text" && !isStructuredOutputRetryFeedback(lastHistoryItem.content)) {
    responseHistory.push(createUserTextMessage(RETRY_RESUMABILITY_PROMPT, "developer"));
  }

  return responseHistory;
}
