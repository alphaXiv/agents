import OpenAI from "openai";
import z from "zod";
import { assert } from "@std/assert";
import type { ResponseInputItem } from "openai/resources/responses/responses";
import type { Tool } from "../tool.ts";
import type { ChatItem } from "../types.ts";

export class OpenAIAdapter<zO, zI> {
  #client: OpenAI;
  #model: string;
  #output?: z.ZodType<zO, zI>;
  #normalizedTools: {
    original: Tool<unknown, unknown>;
    openai: OpenAI.Responses.FunctionTool;
    /** OpenAI doesn't allow non-objects at the top level but we want to. We therefore wrap the tool input with a wrapper object which need to unwrap at the output */
    wrapperObject: boolean;
  }[];

  constructor(
    { model, output, tools }: {
      model: string;
      output?: z.ZodType<zO, zI>;
      tools: Tool<unknown, unknown>[];
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
        openai: {
          name,
          parameters: z.toJSONSchema(
            wrapperObject
              ? z.object({ content: tool.parameters })
              : tool.parameters,
          ),
          description: tool.description,
          type: "function",
          strict: true,
        },
        wrapperObject,
      };
    });
    this.#client = new OpenAI();
  }

  async run({ history, systemPrompt }: {
    systemPrompt: string;
    history: ChatItem[];
  }): Promise<ChatItem[]> {
    const openAIHistory: ResponseInputItem[] = [];
    for (const historyItem of history) {
      if (historyItem.type === "input_text") {
        openAIHistory.push({
          type: "message",
          role: "user",
          content: historyItem.text,
        });
      } else if (historyItem.type === "output_text") {
        openAIHistory.push({
          type: "message",
          role: "assistant",
          content: historyItem.text,
        });
      } else if (historyItem.type === "tool_use") {
        const tool = this.#normalizedTools.find((tool) =>
          tool.original.name === historyItem.name
        );
        assert(tool);
        openAIHistory.push({
          type: "custom_tool_call",
          call_id: historyItem.id,
          name: tool.openai.name,
          input: tool.wrapperObject
            ? `{"content":${historyItem.input}}`
            : historyItem.input,
        });
      } else if (historyItem.type === "tool_result") {
        openAIHistory.push({
          type: "custom_tool_call_output",
          call_id: historyItem.tool_use_id,
          output: historyItem.content,
        });
      } else if (historyItem.type === "output_reasoning") {
        // no-op, don't propagate reasoning
      } else {
        historyItem satisfies never;
      }
    }

    const response = await this.#client.responses.create({
      model: this.#model,
      input: openAIHistory,
      instructions: systemPrompt,
      tools: this.#normalizedTools.map(({ openai }) => openai),
      text: {
        format: this.#output
          ? {
            type: "json_schema",
            name: "result",
            schema: z.toJSONSchema(this.#output),
          }
          : { type: "text" },
      },
      reasoning: {
        summary: "auto",
      },
    });

    const output: ChatItem[] = [];

    for (const part of response.output) {
      if (part.type === "message") {
        output.push({
          type: "output_text",
          text: part.content.map((content) =>
            content.type === "output_text" ? content.text : content.refusal
          ).join("\n"),
        });
      } else if (part.type === "function_call") {
        const tool = this.#normalizedTools.find((tool) =>
          tool.openai.name === part.name
        );
        assert(tool);
        const input = tool.wrapperObject
          ? JSON.stringify(JSON.parse(part.arguments).content)
          : part.arguments;

        output.push({
          type: "tool_use",
          id: part.call_id,
          name: tool.original.name,
          input,
        });
      } else if (part.type === "reasoning") {
        let reasoningText = "";
        if (part.content) {
          reasoningText += part.content.map((c) => c.text).join("\n\n");
        } else if (part.summary) {
          reasoningText += part.summary.map((c) => c.text).join("\n\n");
        }
        if (reasoningText) {
          output.push({
            type: "output_reasoning",
            text: reasoningText,
          });
        }
      } else {
        console.warn("Unrecognized type", part);
      }
    }

    return output;
  }
}
