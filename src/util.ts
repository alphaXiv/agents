import process from "node:process";
import type { ChatItem, ChatLike } from "./types.ts";

export function convertChatLikeToChatItem<T extends ChatItem["type"]>(
  chatLike: ChatLike,
  type: T,
  otherData?: Record<string, string>,
): ChatItem[] {
  if (typeof chatLike === "string") {
    if (type === "input_text" || type === "output_text") {
      return [{
        type,
        content: chatLike,
      }];
    } else if (type === "tool_use") {
      return [
        {
          type,
          tool_use_id: otherData!.tool_use_id,
          name: otherData!.name,
          content: chatLike,
        },
      ];
    } else if (type === "tool_result") {
      return [
        {
          type,
          tool_use_id: otherData!.tool_use_id,
          content: chatLike,
        },
      ];
    }
    throw new Error("Unhandled type");
  }

  return chatLike;
}

export function crossPlatformEnv(key: string) {
  return process.env[key];
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
