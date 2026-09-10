import { assert } from "@std/assert";
import type OpenAI from "openai";
import { toFile } from "openai";
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
import type { OpenResponsesFilesConfig } from "./adapter.ts";
import type { OpenResponsesToolMap } from "./tools.ts";

type FileHistoryItem = Extract<ChatItem, { type: "input_file" } | { type: "tool_result_file" }>;

function getSyntheticId(prefix: string) {
  return `${prefix}_${crypto.randomUUID().replaceAll("-", "")}`;
}

/**
 * The reasoning item that produced one tool call, as the endpoint returned it.
 * The encrypted blob is opaque, and only the endpoint that issued it can decrypt it.
 */
export interface ToolCallReplay {
  reasoningItemId: string;
  encryptedContent: string;
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

/** Counted from the upload, never extended. */
const FILE_LIFETIME_MS = 7 * 24 * 60 * 60 * 1000;

export interface OpenResponsesFiles extends OpenResponsesFilesConfig {
  client: OpenAI["files"];
  /** URL to file id sent in this attempt, so the adapter knows which rows to drop when the provider rejects one. */
  sent: Map<string, string>;
  /** File ids uploaded by this call, across attempts. A rejection of one of these is not a stale row. */
  uploaded: Set<string>;
}

/** Providers check the filename extension on upload, and the last segment of a URL often has none. */
function uploadFileName(url: string, mimeType: string): string {
  const name = getFileNameFromUrl(url) ?? "file";
  const subtype = mimeType.split("/")[1];
  const accepted = subtype === "jpeg" ? ["jpeg", "jpg"] : [subtype];
  if (accepted.some((extension) => name.toLowerCase().endsWith(`.${extension}`))) return name;
  return `${name}.${accepted.at(-1)}`;
}

async function getProviderFileId(
  url: string,
  mimeType: string,
  files: OpenResponsesFiles,
  signal: AbortSignal,
): Promise<string> {
  const now = new Date();
  const cached = await files.store.get(url);
  if (cached && cached.expiresAt > now) {
    files.sent.set(url, cached.fileId);
    return cached.fileId;
  }

  if (cached) {
    // Past its expiry every reader treats this row as a miss, so the provider file is ours to delete.
    await files.client.delete(cached.fileId, { signal }).catch(() => {});
  }

  const response = await fetch(url, { signal });
  if (!response.ok) {
    throw new Error(`Failed to download ${url}: ${response.status} ${response.statusText}`);
  }

  const uploaded = await files.client.create({
    file: await toFile(await response.blob(), uploadFileName(url, mimeType)),
    purpose: files.purpose,
    expires_after: files.expiresAfterSeconds ? { anchor: "created_at", seconds: files.expiresAfterSeconds } : undefined,
  }, { signal });

  const winner = await files.store.set(url, uploaded.id, new Date(now.getTime() + FILE_LIFETIME_MS));
  if (winner.fileId === uploaded.id) {
    files.uploaded.add(uploaded.id);
  } else {
    // Another upload of the same URL was stored first, so ours is an orphan.
    await files.client.delete(uploaded.id, { signal }).catch(() => {});
  }

  files.sent.set(url, winner.fileId);
  return winner.fileId;
}

async function getOpenResponsesFileInput(
  model: string,
  historyItem: FileHistoryItem,
  supportedMimeTypes: string[],
  signal: AbortSignal,
  files?: OpenResponsesFiles,
): Promise<ResponseInputText | ResponseInputImage | ResponseInputFile> {
  if (!supportsMimeType(historyItem.kind, supportedMimeTypes)) {
    throw unsupportedMediaTypeError(model, historyItem.kind);
  }

  if (IMAGE_MIME_TYPES.some((mimeType) => mimeType === historyItem.kind)) {
    if (files) {
      return {
        type: "input_image",
        file_id: await getProviderFileId(historyItem.content, historyItem.kind, files, signal),
        detail: "auto",
      };
    }

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
    if (files) {
      return {
        type: "input_file",
        file_id: await getProviderFileId(historyItem.content, historyItem.kind, files, signal),
      };
    }

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
  toolCallReplays?: Map<string, ToolCallReplay>;
  files?: OpenResponsesFiles;
}): Promise<ResponseInputItem[]> {
  const supportedMimeTypes = options.supportedMimeTypes ?? DEFAULT_SUPPORTED_MIME_TYPES;
  const responseHistory: ResponseInputItem[] = [];
  const calledToolUseIds = new Set<string>();
  const replayedReasoningItemIds = new Set<string>();

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
        calledToolUseIds.add(historyItem.tool_use_id);

        // Replaying the reasoning that produced the call lets the model reuse it instead of deriving
        // the whole chain again.
        // The reasoning item has to precede the call, and several calls can share one, so it is
        // emitted once for the group.
        // The call keeps a synthetic id so nothing in the request needs a server-side lookup.
        const replay = options.toolCallReplays?.get(historyItem.tool_use_id);
        if (replay && !replayedReasoningItemIds.has(replay.reasoningItemId)) {
          replayedReasoningItemIds.add(replay.reasoningItemId);
          responseHistory.push({
            type: "reasoning",
            id: replay.reasoningItemId,
            encrypted_content: replay.encryptedContent,
            summary: [],
          });
        }

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
        // A result whose call is not in this history cannot be paired, and the API
        // rejects the request outright over it, so drop it.
        if (!calledToolUseIds.has(historyItem.tool_use_id)) break;
        const output = getOrCreateFunctionCallOutput(responseHistory, historyItem.tool_use_id);
        assert(typeof output.output !== "string");
        output.output.push({ type: "input_text", text: historyItem.content });
        break;
      }
      case "tool_result_file": {
        if (!calledToolUseIds.has(historyItem.tool_use_id)) break;
        const output = getOrCreateFunctionCallOutput(responseHistory, historyItem.tool_use_id);
        assert(typeof output.output !== "string");
        output.output.push(
          await getOpenResponsesFileInput(
            options.model,
            historyItem,
            supportedMimeTypes,
            options.signal,
            options.files,
          ),
        );
        break;
      }
      case "input_file":
        responseHistory.push({
          type: "message",
          role: "user",
          status: "completed",
          content: [
            await getOpenResponsesFileInput(
              options.model,
              historyItem,
              supportedMimeTypes,
              options.signal,
              options.files,
            ),
          ],
        });
        break;
      default:
        historyItem satisfies never;
    }
  }

  // A call whose result never made it into the history is rejected the same way an
  // unpaired result is, so it answers with nothing rather than losing the whole turn.
  const answeredCallIds = new Set(
    responseHistory.flatMap((item) => item.type === "function_call_output" ? [item.call_id] : []),
  );
  for (let index = responseHistory.length - 1; index >= 0; index -= 1) {
    const item = responseHistory[index];
    if (item.type !== "function_call" || answeredCallIds.has(item.call_id)) continue;

    responseHistory.splice(index + 1, 0, {
      id: getSyntheticId("fco"),
      type: "function_call_output",
      call_id: item.call_id,
      status: "completed",
      output: [{ type: "input_text", text: "" }],
    });
  }

  const lastHistoryItem = options.history.at(-1);
  if (lastHistoryItem?.type === "output_text" && !isStructuredOutputRetryFeedback(lastHistoryItem.content)) {
    responseHistory.push(createUserTextMessage(RETRY_RESUMABILITY_PROMPT, "developer"));
  }

  return responseHistory;
}
