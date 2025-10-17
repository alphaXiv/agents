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
        text: chatLike,
      }];
    } else if (type === "tool_use") {
      return [
        {
          type,
          id: otherData!.id,
          name: otherData!.name,
          input: chatLike,
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
