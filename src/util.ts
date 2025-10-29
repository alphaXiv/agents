import process from "node:process";
import type { ChatItem, ChatLike, ToolResultLike } from "./types.ts";
import { encodeHex } from "@std/encoding/hex";

export function convertChatLikeToChatItem<T extends ChatItem["type"]>(
  chatLike: ChatLike,
  type: T,
): ChatItem[] {
  if (typeof chatLike === "string") {
    if (type === "input_text" || type === "output_text") {
      return [{
        type,
        content: chatLike,
      }];
    }
    throw new Error("Unhandled type");
  }

  return chatLike;
}

export function convertToolResultLikeToChatItem(
  toolResultLike: ToolResultLike,
  toolUseId: string,
): ChatItem[] {
  if (typeof toolResultLike === "string") {
    return [{
      type: "tool_result_text",
      tool_use_id: toolUseId,
      content: toolResultLike,
    }];
  }

  return toolResultLike.map((toolResult) => {
    if (toolResult.type === "tool_result_text") {
      return {
        ...toolResult,
        tool_use_id: toolUseId,
      };
    } else {
      return {
        ...toolResult,
        tool_use_id: toolUseId,
      };
    }
  });
}

export function crossPlatformEnv(key: string) {
  return process.env[key];
}

export function crossPlatformLog(str: string) {
  process.stdout.write(str);
}

export function crossPlatformHandleSigInt(handler: () => void) {
  process.on("SIGINT", handler);
}

export function crossPlatformRemoveHandleSigInt(handler: () => void) {
  process.off("SIGINT", handler);
}

export function removeDollarSchema(schema: any) {
  const { $schema: _$schema, ...result } = schema;

  return result;
}

export async function runWithRetries<T>(
  func: () => Promise<T>,
  retries: number,
) {
  let err: unknown;
  for (let i = 0; i < retries; i++) {
    try {
      return await func();
    } catch (error) {
      err = error;
      // no-op
    }
  }
  throw err;
}

export async function hashString(str: string) {
  // Encode the string as UTF-8
  const encoder = new TextEncoder();
  const data = encoder.encode(str);

  // Hash the data
  const hashBuffer = await globalThis.crypto.subtle.digest("SHA-256", data);

  // Convert to hex string
  return encodeHex(hashBuffer);
}
