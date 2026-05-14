import { AsyncLocalStorage } from "node:async_hooks";
import { convertChatItemsToStream } from "../../src/client.ts";
import type { Adapter, AdapterStreamOptions } from "../../src/adapters/adapter.ts";
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

export function deterministicTestModel(): Adapter<unknown, unknown> {
  let structuredRetryCalls = 0;

  return {
    provider: "deterministic",
    model: "deterministic",
    stream(options: AdapterStreamOptions<unknown, unknown>): AdapterStreamIterator {
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
          structuredRetryCalls += 1;
          return convertChatItemsToStream({
            items: [{
              type: "output_text",
              content: structuredRetryCalls === 1 ? JSON.stringify({ name: 123 }) : JSON.stringify({ name: "Bingus" }),
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
              tool_use_id: `id-${outputTool.name}`,
              kind: outputTool.name,
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
              kind: tool.name,
            };
            yield {
              type: "tool_use" as const,
              index: 0,
              tool_use_id: "id-my-tool",
              kind: tool.name,
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
              kind: tools[0].name,
            };
            yield {
              type: "tool_use" as const,
              index: 0,
              tool_use_id: "id-slow",
              kind: tools[0].name,
              content: JSON.stringify("go"),
            };
            yield {
              type: "tool_use_start" as const,
              index: 1,
              tool_use_id: "id-fast",
              kind: tools[1].name,
            };
            yield {
              type: "tool_use" as const,
              index: 1,
              tool_use_id: "id-fast",
              kind: tools[1].name,
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
              kind: searchTool.name,
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
    },
  };
}

export function failingTestModel(): Adapter<unknown, unknown> {
  return {
    provider: "deterministic-failing",
    model: "deterministic-failing",
    stream(_options: AdapterStreamOptions<unknown, unknown>): AdapterStreamIterator {
      const counter = testingTracker.getStore();
      if (counter) {
        counter.failures += 1;
      }
      throw new Error("Deterministic Provider Error");
    },
  };
}

export function contextWindowTestModel(threshold: number): Adapter<unknown, unknown> {
  return {
    provider: "deterministic",
    model: "deterministic",
    stream(options: AdapterStreamOptions<unknown, unknown>): AdapterStreamIterator {
      if (options.history.length > threshold) {
        throw new Error("This model's maximum context length is exceeded");
      }

      return convertChatItemsToStream({
        items: [{ type: "output_text", content: "Recovery successful!" }],
        inputTokens: 0,
        outputTokens: 0,
      });
    },
  };
}

export function historyRecordingTestModel(
  response: string = "ok",
): Adapter<unknown, unknown> & { receivedHistories: ChatItem[][] } {
  const receivedHistories: ChatItem[][] = [];

  return {
    provider: "deterministic",
    model: "deterministic",
    receivedHistories,
    stream(options: AdapterStreamOptions<unknown, unknown>): AdapterStreamIterator {
      receivedHistories.push([...options.history]);

      return convertChatItemsToStream({
        items: [{ type: "output_text", content: response }],
        inputTokens: 0,
        outputTokens: 0,
      });
    },
  };
}
