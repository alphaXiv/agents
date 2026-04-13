import { AsyncLocalStorage } from "node:async_hooks";
import { convertChatItemsToStream } from "../../src/client.ts";
import { Adapter, type AdapterStreamOptions } from "../../src/adapters/adapter.ts";
import { Model } from "../../src/adapters/model.ts";
import type { AdapterStreamIterator, ChatItem } from "../../src/types.ts";

function isToolResult(item: ChatItem): item is Extract<ChatItem, { type: "tool_result_text" | "tool_result_file" }> {
  return item.type === "tool_result_text" || item.type === "tool_result_file";
}

export const testingTracker = new AsyncLocalStorage<{ failures: number }>();

async function* streamText(text: string): AdapterStreamIterator {
  for (const char of [...text]) {
    yield {
      type: "delta_output_text",
      index: 0,
      delta: char,
    };
  }

  return { inputTokens: 0, outputTokens: 0 };
}

class DeterministicTestAdapter extends Adapter<"deterministic"> {
  name = "deterministic";
  #structuredRetryCalls = 0;

  constructor() {
    super({ model: "deterministic" });
  }

  stream<zO, zI>(options: AdapterStreamOptions<zO, zI>): AdapterStreamIterator {
    options.signal.throwIfAborted();

    const { instructions, output, history, tools } = options;
    const lastItem = history.at(-1);

    if (instructions === "Basic test") {
      return streamText("Basic test worked!");
    }

    if (output) {
      if (instructions.includes("temperature guesstimate")) {
        return convertChatItemsToStream({
          items: [{ type: "output_text", content: "23" }],
          inputTokens: 0,
          outputTokens: 0,
        });
      }

      if (instructions.includes("name cats")) {
        return convertChatItemsToStream({
          items: [{ type: "output_text", content: JSON.stringify({ name: "Bingus" }) }],
          inputTokens: 0,
          outputTokens: 0,
        });
      }

      if (instructions === "Structured output retry stream test") {
        this.#structuredRetryCalls += 1;
        return convertChatItemsToStream({
          items: [{
            type: "output_text",
            content: this.#structuredRetryCalls === 1
              ? JSON.stringify({ name: 123 })
              : JSON.stringify({ name: "Bingus" }),
          }],
          inputTokens: 0,
          outputTokens: 0,
        });
      }
    }

    if (tools.length > 0) {
      const outputTool = tools.find((tool) =>
        tool.normalizedName === "output_tool" || tool.normalizedName === "other_output"
      );
      if (outputTool && !history.some(isToolResult)) {
        return convertChatItemsToStream({
          items: [{
            type: "tool_use",
            tool_use_id: `id-${outputTool.normalizedName}`,
            kind: outputTool.normalizedName,
          }],
          inputTokens: 0,
          outputTokens: 0,
        });
      }

      if (instructions === "Parallel tool test" && tools.length === 1 && !history.some(isToolResult)) {
        const [tool] = tools;
        return (async function* () {
          yield {
            type: "tool_use_start" as const,
            index: 0,
            tool_use_id: "id-my-tool",
            kind: tool.normalizedName,
          };
          yield {
            type: "tool_use" as const,
            index: 0,
            tool_use_id: "id-my-tool",
            kind: tool.normalizedName,
            content: JSON.stringify("go"),
          };
          return { inputTokens: 0, outputTokens: 0 };
        })();
      }

      if (instructions === "Parallel tool test" && tools.length === 2 && !history.some(isToolResult)) {
        return (async function* () {
          yield {
            type: "tool_use_start" as const,
            index: 0,
            tool_use_id: "id-slow",
            kind: tools[0].normalizedName,
          };
          yield {
            type: "tool_use" as const,
            index: 0,
            tool_use_id: "id-slow",
            kind: tools[0].normalizedName,
            content: JSON.stringify("go"),
          };
          yield {
            type: "tool_use_start" as const,
            index: 1,
            tool_use_id: "id-fast",
            kind: tools[1].normalizedName,
          };
          yield {
            type: "tool_use" as const,
            index: 1,
            tool_use_id: "id-fast",
            kind: tools[1].normalizedName,
            content: JSON.stringify("go"),
          };
          return { inputTokens: 0, outputTokens: 0 };
        })();
      }

      const searchTool = tools.find((tool) => tool.normalizedName === "searching_the_internet");
      if (searchTool && !history.some(isToolResult)) {
        return convertChatItemsToStream({
          items: [{
            type: "tool_use",
            tool_use_id: "id-search",
            kind: searchTool.normalizedName,
            content: JSON.stringify("cats"),
          }],
          inputTokens: 0,
          outputTokens: 0,
        });
      }

      if (lastItem && isToolResult(lastItem)) {
        return convertChatItemsToStream({
          items: [{ type: "output_text", content: lastItem.content }],
          inputTokens: 0,
          outputTokens: 0,
        });
      }
    }

    return convertChatItemsToStream({
      items: [{ type: "output_text", content: "Hey! How are you doing?" }],
      inputTokens: 0,
      outputTokens: 0,
    });
  }
}

class FailingTestAdapter extends Adapter<"deterministic-failing"> {
  name = "deterministic-failing";

  constructor() {
    super({ model: "deterministic-failing" });
  }

  stream<zO, zI>(_options: AdapterStreamOptions<zO, zI>): AdapterStreamIterator {
    const counter = testingTracker.getStore();
    if (counter) {
      counter.failures += 1;
    }
    throw new Error("Deterministic Provider Error");
  }
}

export class DeterministicTestModel extends Model<"deterministic"> {
  adapter: DeterministicTestAdapter;

  constructor() {
    super({ model: "deterministic" });
    this.adapter = new DeterministicTestAdapter();
  }
}

export class FailingTestModel extends Model<"deterministic-failing"> {
  adapter: FailingTestAdapter;

  constructor() {
    super({ model: "deterministic-failing" });
    this.adapter = new FailingTestAdapter();
  }
}

class ContextWindowTestAdapter extends Adapter<"deterministic"> {
  name = "deterministic";
  #threshold: number;

  constructor(threshold: number) {
    super({ model: "deterministic" });
    this.#threshold = threshold;
  }

  stream<zO, zI>(options: AdapterStreamOptions<zO, zI>): AdapterStreamIterator {
    if (options.history.length > this.#threshold) {
      throw new Error("This model's maximum context length is exceeded");
    }

    return convertChatItemsToStream({
      items: [{ type: "output_text", content: "Recovery successful!" }],
      inputTokens: 0,
      outputTokens: 0,
    });
  }
}

export class ContextWindowTestModel extends Model<"deterministic"> {
  adapter: ContextWindowTestAdapter;

  constructor(threshold: number) {
    super({ model: "deterministic" });
    this.adapter = new ContextWindowTestAdapter(threshold);
  }
}

class HistoryRecordingTestAdapter extends Adapter<"deterministic"> {
  name = "deterministic";
  receivedHistories: ChatItem[][] = [];
  #response: string;

  constructor(response: string = "ok") {
    super({ model: "deterministic" });
    this.#response = response;
  }

  stream<zO, zI>(options: AdapterStreamOptions<zO, zI>): AdapterStreamIterator {
    this.receivedHistories.push([...options.history]);

    return convertChatItemsToStream({
      items: [{ type: "output_text", content: this.#response }],
      inputTokens: 0,
      outputTokens: 0,
    });
  }
}

export class HistoryRecordingTestModel extends Model<"deterministic"> {
  adapter: HistoryRecordingTestAdapter;

  constructor(response: string = "ok") {
    super({ model: "deterministic" });
    this.adapter = new HistoryRecordingTestAdapter(response);
  }

  get receivedHistories(): ChatItem[][] {
    return this.adapter.receivedHistories;
  }
}
