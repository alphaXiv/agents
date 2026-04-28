import type { Content } from "@google/genai";
import { assert } from "@std/assert";
import { normalizeToolName } from "../../tool.ts";
import type { ChatItem, ChatItemToolUse } from "../../types.ts";
import { serializeWrappedToolArguments } from "../shared/tools.ts";
import type { GoogleToolMap } from "./tools.ts";

type EnsureFileUploaded = (url: string, mimeType: string, abortSignal: AbortSignal) => Promise<string>;

const signatureMap = new Map<string, string>();

/**
 * Google requires functionCall.args to be an object-like Struct, so replayed
 * primitive tool inputs need wrapping when we no longer have the original schema.
 */
function normalizeGoogleFunctionCallArgs(content: string | undefined, tool: GoogleToolMap | undefined) {
  if (!content) return undefined;

  try {
    const parsed = JSON.parse(serializeWrappedToolArguments(content, tool));
    if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
      return parsed;
    }
    return { content: parsed };
  } catch {
    return { content };
  }
}

function getGoogleFileBaseUrl(url: string) {
  return new URL("v1beta/files/", url.endsWith("/") ? url : `${url}/`).toString();
}

export function rememberGoogleThoughtSignature(toolUseId: string, signature: string) {
  signatureMap.set(toolUseId, signature);
}

export async function getGoogleGenerateContentAPIHistory(options: {
  history: ChatItem[];
  toolMap: GoogleToolMap[];
  signal: AbortSignal;
  baseUrl?: string;
  ensureFileUploaded?: EnsureFileUploaded;
}): Promise<Content[]> {
  const googleHistory: Content[] = [];
  for (const item of options.history) {
    switch (item.type) {
      case "input_text":
        googleHistory.push({ role: "user", parts: [{ text: item.content }] });
        break;
      case "output_text":
        googleHistory.push({ role: "model", parts: [{ text: item.content }] });
        break;
      case "context_summary":
        googleHistory.push({ role: "user", parts: [{ text: item.content }] });
        break;
      case "tool_use": {
        const tool = options.toolMap.find((tool) => tool.original.name === item.kind);
        // Magic word comes from https://ai.google.dev/gemini-api/docs/gemini-3?thinking=high#migrating_from_other_models
        const thoughtSignature = signatureMap.get(item.tool_use_id) ?? "context_engineering_is_the_way_to_go";

        googleHistory.push({
          role: "model",
          parts: [{
            functionCall: {
              id: item.tool_use_id,
              name: tool?.google.name ?? normalizeToolName(item.kind),
              args: normalizeGoogleFunctionCallArgs(item.content, tool),
            },
            thoughtSignature,
          }],
        });
        break;
      }
      case "tool_result_text": {
        const toolCall = options.history.find((candidate): candidate is ChatItemToolUse =>
          candidate.type === "tool_use" &&
          candidate.tool_use_id === item.tool_use_id
        );
        assert(toolCall, "Tool result is present in the history without initial tool call");

        // We don't actually assert the definition's existence. Chat history might get reused without previously existing tool calls,
        //  e.g. for context compaction, or when user wants to implement custom tool selection system.
        // The kind is enough to normalize to the original function name.
        const definition = options.toolMap.find((tool) => tool.original.name === toolCall.kind);

        googleHistory.push({
          role: "user",
          parts: [{
            functionResponse: {
              id: item.tool_use_id,
              name: definition?.google.name ?? normalizeToolName(toolCall.kind),
              response: { content: item.content },
            },
          }],
        });
        break;
      }
      case "input_file":
      case "tool_result_file": {
        if (!options.ensureFileUploaded) {
          throw new Error("Google history file replay requires an upload handler");
        }
        const fileName = await options.ensureFileUploaded(item.content, item.kind, options.signal);
        googleHistory.push({
          role: "user",
          parts: [{
            fileData: {
              fileUri: new URL(
                fileName,
                getGoogleFileBaseUrl(options.baseUrl ?? "https://generativelanguage.googleapis.com"),
              ).toString(),
              mimeType: item.kind,
            },
          }],
        });
        break;
      }
      case "output_reasoning":
        // no-op, don't propagate reasoning
        break;
      default:
        item satisfies never;
    }
  }

  return googleHistory;
}
