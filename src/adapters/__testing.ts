import type z from "zod";
import type { Tool } from "../tool.ts";
import type { ChatItem } from "../types.ts";

export class TestingAdapter<zO, zI> {
  #model: string;
  #output?: z.ZodType<zO, zI>;
  #tools: Tool<unknown, unknown>[];

  constructor(
    { model, output, tools }: {
      model: string;
      output?: z.ZodType<zO, zI>;
      tools: Tool<unknown, unknown>[];
    },
  ) {
    this.#model = model;
    this.#output = output;
    this.#tools = tools;
  }

  async run({ history }: {
    systemPrompt: string;
    history: ChatItem[];
  }): Promise<ChatItem[]> {
    const lastMessage = history.slice().pop();
    if (!lastMessage) {
      return [{
        type: "output_text",
        text: "How can I assist you today?",
      }];
    }
    if (
      lastMessage.type === "input_text" &&
      lastMessage.text.toLowerCase().includes("hello")
    ) {
      return [{
        type: "output_text",
        text: "Hey! How are you doing?",
      }];
    }

    if (
      lastMessage.type === "input_text" &&
      lastMessage.text === "Can you give me a temperature estimate?"
    ) {
      return [{ type: "output_text", text: "0" }];
    }

    if (
      lastMessage.type === "input_text" &&
      lastMessage.text === "Can you give me a cat name?"
    ) {
      return [{
        type: "output_text",
        text: JSON.stringify({ name: "Bingus" }),
      }];
    }

    if (
      lastMessage.type === "input_text" &&
      lastMessage.text === "Can you tell me what cat websites there are?"
    ) {
      const searchTool = this.#tools[0];
      return [
        {
          type: "tool_use",
          id: Math.random().toString(),
          name: searchTool.name,
          input: '"cats"',
        },
      ];
    }

    if (lastMessage.type === "tool_result") {
      return [{
        type: "output_text",
        text: "looks like the tool call got " + lastMessage.content,
      }];
    }

    return [
      {
        type: "output_text",
        text:
          "I'm sorry, but I seem to be having issues processing your request...",
      },
    ];
  }
}
