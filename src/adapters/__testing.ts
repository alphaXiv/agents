import type z from "zod";
import type { Tool } from "../tool.ts";
import type { AsyncStreamItemGenerator, ChatItem } from "../types.ts";
import { AsyncLocalStorage } from "node:async_hooks";

export const testingTracker = new AsyncLocalStorage<{ failures: number }>();

export class TestingAdapter<zO, zI> {
  #tools: Tool<unknown, unknown>[];

  constructor(
    { tools }: {
      model: string;
      output?: z.ZodType<zO, zI>;
      tools: Tool<unknown, unknown>[];
    },
  ) {
    this.#tools = tools;
  }

  async run({ history }: {
    systemPrompt: string;
    history: ChatItem[];
    signal: AbortSignal;
  }): Promise<ChatItem[]> {
    const lastMessage = history.slice().pop();
    if (!lastMessage) {
      return [{
        type: "output_text",
        content: "How can I assist you today?",
      }];
    }
    if (
      lastMessage.type === "input_text" &&
      lastMessage.content.toLowerCase().includes("hello")
    ) {
      return [{
        type: "output_text",
        content: "Hey! How are you doing?",
      }];
    }

    if (
      lastMessage.type === "input_text" &&
      lastMessage.content === "Can you give me a temperature estimate?"
    ) {
      return [{ type: "output_text", content: "0" }];
    }

    if (
      lastMessage.type === "input_text" &&
      lastMessage.content === "Can you give me a cat name?"
    ) {
      return [{
        type: "output_text",
        content: JSON.stringify({ name: "Bingus" }),
      }];
    }

    if (
      lastMessage.type === "input_text" &&
      lastMessage.content === "Can you tell me what cat websites there are?"
    ) {
      const searchTool = this.#tools[0];
      return [
        {
          type: "tool_use",
          tool_use_id: Math.random().toString(),
          kind: searchTool.name,
          content: '"cats"',
        },
      ];
    }

    if (lastMessage.type === "tool_result_text") {
      if (lastMessage.content === "throw") {
        const store = testingTracker.getStore();
        if (store) {
          store.failures += 1;
        }
        throw new Error("Deterministic Provider Error");
      }

      return [{
        type: "output_text",
        content: "looks like the tool call got " + lastMessage.content,
      }];
    }

    return [
      {
        type: "output_text",
        content:
          "I'm sorry, but I seem to be having issues processing your request...",
      },
    ];
  }

  // TODO: add testing here
  async *stream({}: {
    systemPrompt: string;
    history: ChatItem[];
    signal: AbortSignal;
  }): AsyncStreamItemGenerator {
  }
}
