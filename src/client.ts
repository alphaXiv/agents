import type { ChatItem, StreamItem } from "./types.ts";

/** Mutates current chat items to add the new streamItem to it.  */
export function addStreamItem(
  currentChatItems: ChatItem[],
  streamItem: StreamItem,
) {
  if (!currentChatItems[streamItem.index]) {
    if (streamItem.type === "delta_output_text") {
      currentChatItems[streamItem.index] = {
        type: "output_text",
        content: "",
      };
    } else if (streamItem.type === "delta_output_reasoning") {
      currentChatItems[streamItem.index] = {
        type: "output_reasoning",
        content: "",
      };
    } else if (streamItem.type === "tool_use") {
      currentChatItems[streamItem.index] = {
        type: "tool_use",
        tool_use_id: streamItem.tool_use_id,
        kind: streamItem.kind,
        content: streamItem.content,
      };
    } else if (streamItem.type === "tool_result_text") {
      currentChatItems[streamItem.index] = {
        type: "tool_result_text",
        tool_use_id: streamItem.tool_use_id,
        content: streamItem.content,
      };
    } else if (streamItem.type === "tool_result_file") {
      currentChatItems[streamItem.index] = {
        type: "tool_result_file",
        tool_use_id: streamItem.tool_use_id,
        kind: streamItem.kind,
        content: streamItem.content,
      };
    }
  }

  if (
    streamItem.type === "delta_output_text" ||
    streamItem.type === "delta_output_reasoning"
  ) {
    currentChatItems[streamItem.index].content += streamItem.delta;
  }
}
