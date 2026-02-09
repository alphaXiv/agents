import type { ChatItem, StreamItem } from "./types.ts";

/** Collects StreamItems into a ChatItem array sorted by index, handling out-of-order arrival. */
export class StreamCollector {
  readonly items: ChatItem[] = [];
  #indices: number[] = [];
  #map = new Map<number, ChatItem>();

  get size(): number {
    return this.items.length;
  }

  has(index: number): boolean {
    return this.#map.has(index);
  }

  #createItem(streamItem: StreamItem): ChatItem | null {
    switch (streamItem.type) {
      case "delta_output_text":
        return { type: "output_text", content: "" };
      case "delta_output_reasoning":
        return { type: "output_reasoning", content: "" };
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
      default:
        return null;
    }
  }

  #findInsertIndex(index: number): number {
    // Binary search insertion point to keep items sorted by index.
    let low = 0;
    let high = this.#indices.length;
    while (low < high) {
      const mid = (low + high) >> 1;
      if (this.#indices[mid] < index) {
        low = mid + 1;
      } else {
        high = mid;
      }
    }
    return low;
  }

  add(streamItem: StreamItem) {
    if (!this.#map.has(streamItem.index)) {
      const item = this.#createItem(streamItem);
      if (!item) return;

      const insertAt = this.#findInsertIndex(streamItem.index);
      this.#indices.splice(insertAt, 0, streamItem.index);
      this.items.splice(insertAt, 0, item);
      this.#map.set(streamItem.index, item);
    }

    if (
      streamItem.type === "delta_output_text" ||
      streamItem.type === "delta_output_reasoning"
    ) {
      this.#map.get(streamItem.index)!.content += streamItem.delta;
    }
  }
}
