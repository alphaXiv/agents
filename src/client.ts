import type { AdapterStreamIterator, ChatItem, StreamItem } from "./types.ts";

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
    } else if (streamItem.type === "tool_use_start") {
      currentChatItems[streamItem.index] = {
        type: "tool_use",
        tool_use_id: streamItem.tool_use_id,
        kind: streamItem.kind,
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

  if (streamItem.type === "tool_use") {
    currentChatItems[streamItem.index] = {
      type: "tool_use",
      tool_use_id: streamItem.tool_use_id,
      kind: streamItem.kind,
      content: streamItem.content,
    };
  }

  if (
    streamItem.type === "delta_output_text" ||
    streamItem.type === "delta_output_reasoning"
  ) {
    currentChatItems[streamItem.index].content += streamItem.delta;
  }
}

/**
 * Convert a non-streaming chat result into a chat stream. This is useful to
 * create testing adapters that do not do any streaming, but need to fulfill the
 * streaming interface.
 */
export async function* convertChatItemsToStream(input: {
  items: ChatItem[];
  inputTokens: number;
  outputTokens: number;
}): AdapterStreamIterator {
  const { inputTokens, outputTokens, items } = input;
  let index = 0;
  for (const item of items) {
    if (item.type === "output_text") {
      yield {
        type: "delta_output_text",
        delta: item.content,
        index,
      };
      index += 1;
    } else if (item.type === "output_reasoning") {
      yield {
        type: "delta_output_reasoning",
        delta: item.content,
        index,
      };
      index += 1;
    } else if (item.type === "tool_use") {
      yield {
        type: "tool_use",
        tool_use_id: item.tool_use_id,
        kind: item.kind,
        content: item.content,
        index,
      };
      index += 1;
    } else if (item.type === "tool_result_text") {
      yield {
        type: "tool_result_text",
        tool_use_id: item.tool_use_id,
        content: item.content,
        index,
      };
      index += 1;
    } else if (item.type === "tool_result_file") {
      yield {
        type: "tool_result_file",
        tool_use_id: item.tool_use_id,
        kind: item.kind,
        content: item.content,
        index,
      };
      index += 1;
    } else {
      throw new Error(
        "Adapters cannot emit chat item type " + JSON.stringify(item.type),
      );
    }
  }
  return { inputTokens, outputTokens };
}
