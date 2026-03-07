import process from "node:process";
import { Readable } from "node:stream";
import type {
  ChatItem,
  ChatItemToolResult,
  ChatLike,
  ToolResultLike,
} from "./types.ts";
import { encodeHex } from "@std/encoding/hex";

export function convertChatLikeToChatItem(
  chatLike: ChatLike,
  type: ChatItem["type"],
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
): ChatItemToolResult[] {
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

export function crossPlatformStdin() {
  return Readable.toWeb(process.stdin) as ReadableStream<
    Uint8Array<ArrayBuffer>
  >;
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

// deno-lint-ignore no-explicit-any
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

/**
 * Streaming `Promise.all` - Return an iterator of the promises in the order they complete.
 */
export async function* iteratePromiseArray<T>(
  promises: Iterable<Promise<T>>,
): AsyncIterableIterator<T> {
  const pending = Array.from(promises);
  const rest = new Set(pending);
  const errors: unknown[] = [];
  let resolutions: T[] = [];

  for (const promise of pending) {
    promise.then((result) => {
      resolutions.push(result);
      rest.delete(promise);
    }, (error) => {
      errors.push(error);
      rest.delete(promise);
    });
  }

  while (rest.size > 0) {
    await Promise.race(rest).catch(() => {});
    yield* resolutions;
    resolutions = [];
  }

  if (errors.length > 0) {
    if (errors.length === 1) throw errors[0];
    throw new AggregateError(errors);
  }
}

export function requireEnv(name: string) {
  const value = crossPlatformEnv(name);
  if (!value) {
    throw new Error(`Missing required environment variable ${name}`);
  }
  return value;
}
