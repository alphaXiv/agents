import OpenAI from "openai";
import type {
  ChatCompletionFunctionTool,
  ChatCompletionMessageParam,
} from "openai/resources/chat/completions/completions";
import z from "zod";
import { assert } from "@std/assert";

import { Tool } from "../tool.ts";
import type { ChatItem, ZodSchemaType } from "../types.ts";
import { crossPlatformEnv } from "../util.ts";
import { assertEquals } from "@std/assert/equals";

export class OpenRouterAdapter<O> {
  #client: OpenAI;
  #model: string;
  #output?: ZodSchemaType<O>;
  #normalizedTools: {
    original: Tool<unknown>;
    openrouter: ChatCompletionFunctionTool;
    /** OpenAI doesn't allow non-objects at the top level but we want to. We therefore wrap the tool input with a wrapper object which need to unwrap at the output */
    wrapperObject: boolean;
  }[];

  constructor(
    { model, output, tools }: {
      model: string;
      output?: ZodSchemaType<O>;
      tools: Tool<unknown>[];
    },
  ) {
    this.#model = model;
    this.#output = output;
    this.#normalizedTools = tools.map((tool) => {
      const name = tool.name.toLowerCase().replaceAll(" ", "_").replace(
        /[^a-zA-Z0-9_-]/g,
        "",
      ); // TODO: improve this mapping
      const wrapperObject = !(tool.parameters instanceof z.ZodObject);

      return {
        original: tool,
        openrouter: {
          type: "function",
          function: {
            name,
            parameters: z.toJSONSchema(
              wrapperObject
                ? z.object({ content: tool.parameters })
                : tool.parameters,
            ),
            description: tool.description,
            strict: false,
          },
        },
        wrapperObject,
      };
    });
    this.#client = new OpenAI({
      baseURL: "https://openrouter.ai/api/v1",
      apiKey: crossPlatformEnv("OPENROUTER_API_KEY"),
    });
  }

  async run({ history, systemPrompt }: {
    systemPrompt: string;
    history: ChatItem[];
  }): Promise<ChatItem[]> {
    const openrouterHistory: ChatCompletionMessageParam[] = [{
      role: "system", // TODO: select right role for each model
      content: systemPrompt,
    }];
    for (const historyItem of history) {
      if (historyItem.type === "input_text") {
        openrouterHistory.push({
          role: "user",
          content: historyItem.text,
        });
      } else if (historyItem.type === "output_text") {
        openrouterHistory.push({
          role: "assistant",
          content: historyItem.text,
        });
      } else if (historyItem.type === "tool_use") {
        const tool = this.#normalizedTools.find((tool) =>
          tool.original.name === historyItem.name
        );
        assert(tool);
        openrouterHistory.push({
          role: "assistant",
          tool_calls: [
            {
              id: historyItem.id,
              type: "function",
              function: {
                name: tool.openrouter.function.name,
                arguments: tool.wrapperObject
                  ? `{"content":${historyItem.input}}`
                  : historyItem.input,
              },
            },
          ],
        });
      } else if (historyItem.type === "tool_result") {
        openrouterHistory.push({
          role: "tool",
          tool_call_id: historyItem.tool_use_id,
          content: historyItem.content,
        });
      } else if (historyItem.type === "output_reasoning") {
        // no-op, don't propagate reasoning
      } else {
        historyItem satisfies never;
      }
    }

    const response = await this.#client.chat.completions.create({
      model: this.#model,
      messages: openrouterHistory,
      tools: this.#normalizedTools.map(({ openrouter }) => openrouter),
      response_format: this.#output
        ? {
          type: "json_schema",
          json_schema: z.toJSONSchema(this.#output) as any,
        }
        : { type: "text" },
    });

    const output: ChatItem[] = [];

    const choice = response.choices[0];

    if (choice.message.content) {
      output.push({
        type: "output_text",
        text: choice.message.content,
      });
    }
    for (const toolUse of choice.message.tool_calls ?? []) {
      assert(toolUse.type === "function");
      const tool = this.#normalizedTools.find((tool) =>
        tool.openrouter.function.name === toolUse.function.name
      );
      assert(tool);
      const content = JSON.parse(toolUse.function.arguments);
      output.push({
        type: "tool_use",
        id: toolUse.id,
        name: tool.original.name,
        input: tool.wrapperObject
          ? JSON.stringify(content.content)
          : toolUse.function.arguments,
      });
    }
    // TODO: figure out reasoning

    return output;
  }
}
