import OpenAI from "openai";
import z from "zod";
import { assert } from "@std/assert";
import type { ResponseInputItem } from "openai/resources/responses/responses";
import type { Tool } from "../tool.ts";
import type { AsyncStreamItemGenerator, ChatItem } from "../types.ts";
import { encodeBase64 } from "@std/encoding";

const supportedImageMimeTypes = [
  "image/jpeg",
  "image/jpg",
  "image/png",
  "image/gif",
  "image/webp",
];

async function getOpenAIHistory(history: ChatItem[], toolMap: OpenAIToolMap[]) {
  const openAIHistory: ResponseInputItem[] = [];
  for (const historyItem of history) {
    if (historyItem.type === "input_text") {
      openAIHistory.push({
        type: "message",
        role: "user",
        content: historyItem.content,
      });
    } else if (historyItem.type === "output_text") {
      openAIHistory.push({
        type: "message",
        role: "assistant",
        content: historyItem.content,
      });
    } else if (historyItem.type === "tool_use") {
      const tool = toolMap.find((tool) =>
        tool.original.name === historyItem.kind
      );
      assert(tool);
      openAIHistory.push({
        type: "custom_tool_call",
        call_id: historyItem.tool_use_id,
        name: tool.openai.name,
        input: tool.wrapperObject
          ? `{"content":${historyItem.content}}`
          : historyItem.content,
      });
    } else if (historyItem.type === "tool_result") {
      openAIHistory.push({
        type: "custom_tool_call_output",
        call_id: historyItem.tool_use_id,
        output: historyItem.content,
      });
    } else if (historyItem.type === "input_file") {
      if (supportedImageMimeTypes.includes(historyItem.kind)) {
        openAIHistory.push({
          type: "message",
          role: "user",
          content: [{
            type: "input_image",
            image_url: historyItem.content,
            detail: "auto",
          }],
        });
      } else if (historyItem.kind === "application/pdf") {
        const req = await fetch(historyItem.content);
        const buffer = await req.arrayBuffer();
        const filename = historyItem.content.split("/").pop(); // TODO: make this heuristic better

        // TODO: check file support
        openAIHistory.push({
          type: "message",
          role: "user",
          content: [{
            type: "input_file",
            file_data: `data:application/pdf;base64,${encodeBase64(buffer)}`, // TODO: investigate openai file api
            filename,
          }],
        });
      } else if (historyItem.kind.startsWith("text/")) {
        const req = await fetch(historyItem.content);
        const text = await req.text();

        openAIHistory.push({
          type: "message",
          role: "user",
          content: [{
            type: "input_text",
            text: `<file>${text}</file>`,
          }],
        });
      } else {
        throw new Error(
          "OpenAI models don't support the following media type: " +
            historyItem.kind,
        );
      }
    } else if (historyItem.type === "output_reasoning") {
      // no-op, don't propagate reasoning
    } else {
      historyItem satisfies never;
    }
  }
  return openAIHistory;
}

type OpenAIToolMap = {
  original: Tool<unknown, unknown>;
  openai: OpenAI.Responses.FunctionTool;
  /** OpenAI doesn't allow non-objects at the top level but we want to. We therefore wrap the tool input with a wrapper object which need to unwrap at the output */
  wrapperObject: boolean;
};

export class OpenAIAdapter<zO, zI> {
  #client: OpenAI;
  #model: string;
  #output?: z.ZodType<zO, zI>;
  #normalizedTools: OpenAIToolMap[];

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
    const openAIHistory = await getOpenAIHistory(
      history,
      this.#normalizedTools,
    );

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
          content: part.content.map((content) =>
            content.type === "output_text" ? content.text : content.refusal
          ).join("\n"),
        });
      } else if (part.type === "function_call") {
        const tool = this.#normalizedTools.find((tool) =>
          tool.openai.name === part.name
        );
        assert(tool);
        const content = tool.wrapperObject
          ? JSON.stringify(JSON.parse(part.arguments).content)
          : part.arguments;

        output.push({
          type: "tool_use",
          tool_use_id: part.call_id,
          kind: tool.original.name,
          content,
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
            content: reasoningText,
          });
        }
      } else {
        console.warn("Unrecognized type", part);
      }
    }

    return output;
  }

  async *stream({ history, systemPrompt }: {
    systemPrompt: string;
    history: ChatItem[];
  }): AsyncStreamItemGenerator {
    const openAIHistory = await getOpenAIHistory(
      history,
      this.#normalizedTools,
    );

    const response = await this.#client.responses.create({
      model: this.#model,
      input: openAIHistory,
      instructions: systemPrompt,
      stream: true,
      tools: this.#normalizedTools.map(({ openai }) => openai),
      reasoning: {
        summary: "auto",
      },
    });

    let toolIndex: ChatItem[] = [];
    for await (const part of response) {
      if (part.type === "response.output_item.added") {
        const partItem = part.item;
        if (partItem.type === "message") {
          yield {
            type: "delta_output_text",
            delta: partItem.content.map((d) =>
              d.type === "refusal" ? d.refusal : d.text
            ).join("\n"),
            index: part.output_index,
          };
        } else if (partItem.type === "reasoning") {
          yield {
            type: "delta_output_reasoning",
            delta: partItem.summary.join("\n"),
            index: part.output_index,
          };
        } else if (partItem.type === "function_call") {
          const tool = this.#normalizedTools.find((tool) =>
            tool.openai.name === partItem.name
          );
          assert(tool);
          toolIndex[part.output_index] = {
            type: "tool_use",
            tool_use_id: partItem.call_id,
            kind: tool.original.name,
            content: partItem.arguments,
          };
        }
      } else if (part.type === "response.output_text.delta") {
        yield {
          type: "delta_output_text",
          delta: part.delta,
          index: part.output_index,
        };
      } else if (part.type === "response.reasoning_summary_text.delta") {
        yield {
          type: "delta_output_reasoning",
          delta: part.delta,
          index: part.output_index,
        };
      } else if (part.type === "response.function_call_arguments.done") {
        const toolUse = toolIndex[part.output_index];
        assert(toolUse.type === "tool_use");
        const tool = this.#normalizedTools.find((tool) =>
          tool.original.name === toolUse.kind
        );
        assert(tool);
        const content = tool.wrapperObject
          ? JSON.stringify(JSON.parse(part.arguments).content)
          : part.arguments;
        yield {
          type: "tool_use",
          tool_use_id: toolUse.tool_use_id,
          kind: tool.original.name,
          content,
          index: part.output_index,
        };
      }
    }
  }
}
