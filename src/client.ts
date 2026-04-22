import type { AdapterStreamIterator, ChatItem, StreamItem, WithTraceId } from "./types.ts";

type MutableChatItem = ChatItem & { trace?: string };

const streamIndicesByChatItems = new WeakMap<MutableChatItem[], number[]>();

function ensureDenseIndex(
  chatItems: MutableChatItem[],
  streamIndex: number,
): { denseIndex: number; isNew: boolean } {
  let streamIndices = streamIndicesByChatItems.get(chatItems);
  if (!streamIndices || streamIndices.length !== chatItems.length) {
    streamIndices = chatItems.map((_, index) => index);
    streamIndicesByChatItems.set(chatItems, streamIndices);
  }

  for (let denseIndex = 0; denseIndex < streamIndices.length; denseIndex += 1) {
    const currentIndex = streamIndices[denseIndex];
    if (currentIndex === streamIndex) {
      return { denseIndex, isNew: false };
    }
    if (currentIndex > streamIndex) {
      streamIndices.splice(denseIndex, 0, streamIndex);
      return { denseIndex, isNew: true };
    }
  }

  streamIndices.push(streamIndex);
  return { denseIndex: streamIndices.length - 1, isNew: true };
}

function createChatItemFromStreamItem(streamItem: StreamItem): ChatItem {
  switch (streamItem.type) {
    case "delta_output_text":
      return {
        type: "output_text",
        content: "",
      };
    case "delta_output_reasoning":
      return {
        type: "output_reasoning",
        content: "",
      };
    case "tool_use_start":
      return {
        type: "tool_use",
        tool_use_id: streamItem.tool_use_id,
        kind: streamItem.kind,
      };
    case "tool_use":
      return {
        type: "tool_use",
        tool_use_id: streamItem.tool_use_id,
        kind: streamItem.kind,
        content: streamItem.content,
      };
    case "tool_result_text":
      return {
        type: "tool_result_text",
        tool_use_id: streamItem.tool_use_id,
        content: streamItem.content,
      };
    case "tool_result_file":
      return {
        type: "tool_result_file",
        tool_use_id: streamItem.tool_use_id,
        kind: streamItem.kind,
        content: streamItem.content,
      };
    case "context_summary":
      return {
        type: "context_summary",
        content: streamItem.content,
      };
    case "token_usage":
    case "context_summary_start":
    case "model_switched":
      throw new Error(
        `Cannot convert informational stream item "${streamItem.type}" into ChatItem.`,
      );
  }
}

/**
 * Mutates current chat items to add the new streamItem to it. This function is
 * generic so that it can properly catch errors if you attempt to call it with
 * an array of `WithTraceId<ChatItem>` and you do not actually provide it traced
 * stream items.
 */
export function addStreamItem<T extends ChatItem>(
  chatItems: T[],
  streamItem: T extends { trace: string } ? WithTraceId<StreamItem> : StreamItem,
): void {
  const currentChatItems = chatItems as MutableChatItem[];

  // Informational events don't produce conversation items.
  if (
    streamItem.type === "token_usage" ||
    streamItem.type === "context_summary_start" ||
    streamItem.type === "model_switched"
  ) {
    return;
  }

  const { denseIndex, isNew } = ensureDenseIndex(
    currentChatItems,
    streamItem.index,
  );

  if (isNew) {
    currentChatItems.splice(denseIndex, 0, createChatItemFromStreamItem(streamItem));
  }

  switch (streamItem.type) {
    case "tool_use":
      currentChatItems[denseIndex] = {
        type: "tool_use",
        tool_use_id: streamItem.tool_use_id,
        kind: streamItem.kind,
        content: streamItem.content,
      };
      break;
    case "delta_output_text":
    case "delta_output_reasoning":
      currentChatItems[denseIndex].content += streamItem.delta;
      break;
  }

  if ("trace" in streamItem) {
    currentChatItems[denseIndex].trace = streamItem.trace as string;
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
    } else if (item.type === "context_summary") {
      yield {
        type: "context_summary",
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
