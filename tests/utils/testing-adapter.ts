import type { AdapterStreamIterator } from "../../src/types.ts";
import { AsyncLocalStorage } from "node:async_hooks";
import type { Adapter } from "../../src/adapters.ts";

export const testingTracker = new AsyncLocalStorage<{ failures: number }>();

async function* streamText(
  text: string,
  index: number,
): AdapterStreamIterator {
  for (const char of text) {
    yield { type: "delta_output_text", delta: char, index };
  }

  return { inputTokens: 0, outputTokens: 0 };
}

async function* streamToolUse(
  index: number,
  tool_use_id: string,
  kind: string,
  content?: string,
): AdapterStreamIterator {
  yield {
    type: "tool_use",
    index,
    tool_use_id,
    kind,
    content,
  };
  return { inputTokens: 0, outputTokens: 0 };
}

export const testingAdapter: Adapter<"deterministic"> = {
  name: "testing",

  async *stream({ systemPrompt, history, tools }) {
    if (systemPrompt === "Basic test") {
      yield* streamText("Basic test worked!", 0);
      return { inputTokens: 0, outputTokens: 0 };
    }

    if (systemPrompt === "Model output stream test") {
      const lastMessage = history.slice().pop();
      if (!lastMessage || lastMessage.type === "input_text") {
        yield {
          type: "tool_use",
          index: 0,
          tool_use_id: "output-tool-stream-id",
          kind: "output_tool",
        };
        return { inputTokens: 0, outputTokens: 0 };
      }
      // Should not reach here if ModelOutput terminates the loop
      yield* streamText("this should not appear", 0);
      return { inputTokens: 0, outputTokens: 0 };
    }

    if (systemPrompt === "Structured output retry stream test") {
      const lastMessage = history.at(-1);
      if (!lastMessage || lastMessage.type === "input_text") {
        yield* streamText(JSON.stringify({ name: 123 }), 0);
        return { inputTokens: 0, outputTokens: 0 };
      }

      if (
        lastMessage.type === "output_text" &&
        lastMessage.content.includes(
          "I will try again to produce a JSON response.",
        )
      ) {
        yield* streamText(JSON.stringify({ name: "Bingus" }), 0);
        return { inputTokens: 0, outputTokens: 0 };
      }

      throw new Error(
        `Unexpected history item for structured retry test: ${lastMessage.type}`,
      );
    }

    if (systemPrompt === "Parallel tool test") {
      const lastMessage = history.slice().pop();
      // First turn: yield two tool_use items to be run in parallel
      if (!lastMessage || lastMessage.type === "input_text") {
        yield {
          type: "tool_use",
          index: 0,
          tool_use_id: "id-slow",
          kind: "slow_tool",
          content: '"query"',
        };
        yield {
          type: "tool_use",
          index: 1,
          tool_use_id: "id-fast",
          kind: "fast_tool",
          content: '"query"',
        };
        return { inputTokens: 0, outputTokens: 0 };
      }
      // Second turn: after tool results, return a final text reply
      yield* streamText("done", 0);
      return { inputTokens: 0, outputTokens: 0 };
    }

    const lastMessage = history.at(-1);
    if (!lastMessage) {
      yield* streamText("How can I assist you today?", 0);
      return { inputTokens: 0, outputTokens: 0 };
    }

    if (
      lastMessage.type === "input_text" &&
      lastMessage.content.toLowerCase().includes("hello")
    ) {
      yield* streamText("Hey! How are you doing?", 0);
      return { inputTokens: 0, outputTokens: 0 };
    }

    if (
      lastMessage.type === "input_text" &&
      lastMessage.content === "Can you give me a temperature estimate?"
    ) {
      yield* streamText("0", 0);
      return { inputTokens: 0, outputTokens: 0 };
    }

    if (
      lastMessage.type === "input_text" &&
      lastMessage.content === "Can you give me a cat name?"
    ) {
      yield* streamText(JSON.stringify({ name: "Bingus" }), 0);
      return { inputTokens: 0, outputTokens: 0 };
    }

    if (
      lastMessage.type === "input_text" &&
      lastMessage.content === "Can you tell me what cat websites there are?"
    ) {
      const searchTool = tools[0];
      if (searchTool) {
        yield* streamToolUse(0, "search-tool-id", searchTool.name, '"cats"');
        return { inputTokens: 0, outputTokens: 0 };
      }
    }

    if (
      lastMessage.type === "input_text" &&
      lastMessage.content === "Call output tool"
    ) {
      const tool = tools.find((t) => t.name === "output_tool");
      if (tool) {
        yield* streamToolUse(0, "output-tool-id", tool.name);
        return { inputTokens: 0, outputTokens: 0 };
      }
    }

    if (
      lastMessage.type === "input_text" &&
      lastMessage.content === "Call output tool and search"
    ) {
      const outputTool = tools.find((t) => t.name === "output_tool");
      const searchTool = tools.find((t) => t.name !== "output_tool");
      if (outputTool && searchTool) {
        yield* streamToolUse(0, "search-tool-id", searchTool.name, '"cats"');
        yield* streamToolUse(1, "output-tool-id", outputTool.name);
        return { inputTokens: 0, outputTokens: 0 };
      }
    }

    if (lastMessage.type === "tool_result_text") {
      if (lastMessage.content === "throw") {
        const store = testingTracker.getStore();
        if (store) {
          store.failures += 1;
        }
        throw new Error("Deterministic Provider Error");
      }

      yield* streamText(
        "looks like the tool call got " + lastMessage.content,
        0,
      );
      return { inputTokens: 0, outputTokens: 0 };
    }

    yield* streamText("[undefined case]", 0);
    return { inputTokens: 0, outputTokens: 0 };
  },
};
