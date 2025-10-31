import Anthropic from "@anthropic-ai/sdk";
import type { Tool as AnthropicTool } from "@anthropic-ai/sdk/resources/messages/messages";
import z from "zod";
import type { Tool } from "../tool.ts";
import type { AsyncStreamItemGenerator, ChatItem } from "../types.ts";
import { assert } from "@std/assert/assert";

const supportedImageMimeTypes = [
  "image/jpeg",
  "image/jpg",
  "image/png",
  "image/gif",
  "image/webp",
];

// TODO: drop signature after 10 minutes or whatever
// Mapping between thinking response and signature since signature is meaningless cross-provider and we technically only need to include thinking for the one step
const signatureMap = new Map<string, string>();

async function getAnthropicHistory(
  history: ChatItem[],
  normalizedTools: AnthropicToolMap[],
  signal: AbortSignal,
) {
  const anthropicHistory: Anthropic.Messages.MessageParam[] = [];
  for (const historyItem of history) {
    if (historyItem.type === "input_text") {
      anthropicHistory.push({
        role: "user",
        content: [{ type: "text", text: historyItem.content }],
      });
    } else if (historyItem.type === "output_text") {
      anthropicHistory.push({
        role: "assistant",
        content: [{ type: "text", text: historyItem.content }],
      });
    } else if (historyItem.type === "tool_use") {
      const tool = normalizedTools.find((tool) =>
        tool.original.name === historyItem.kind
      );
      const content = historyItem.content
        ? JSON.parse(historyItem.content)
        : {};
      anthropicHistory.push({
        role: "assistant",
        content: [{
          type: "tool_use",
          id: historyItem.tool_use_id,
          name: tool?.anthropic.name ?? historyItem.kind,
          input: tool?.wrapperObject ? { content } : content,
        }],
      });
    } else if (historyItem.type === "tool_result_text") {
      anthropicHistory.push({
        role: "user",
        content: [{
          type: "tool_result",
          tool_use_id: historyItem.tool_use_id,
          content: historyItem.content,
          is_error: historyItem.content.startsWith("Error: "),
        }],
      });
    } else if (historyItem.type === "output_reasoning") {
      const signature = signatureMap.get(historyItem.content);
      if (signature) {
        anthropicHistory.push({
          role: "assistant",
          content: [{
            type: "thinking",
            thinking: historyItem.content,
            signature,
          }],
        });
      } else {
        // no-op :( nothing we can do
      }
    } else if (
      historyItem.type === "input_file" ||
      historyItem.type === "tool_result_file"
    ) {
      if (supportedImageMimeTypes.includes(historyItem.kind)) {
        anthropicHistory.push({
          role: "user",
          content: [{
            type: "image",
            source: {
              type: "url",
              url: historyItem.content,
            },
          }],
        });
      } else if (historyItem.kind === "application/pdf") {
        anthropicHistory.push({
          role: "user",
          content: [
            {
              type: "document",
              source: {
                type: "url",
                url: historyItem.content,
              },
            },
          ],
        });
      } else if (historyItem.kind.startsWith("text/")) {
        const req = await fetch(historyItem.content, { signal });
        const text = await req.text();

        anthropicHistory.push({
          role: "user",
          content: [
            {
              type: "text",
              text: `<ant-file>${text}</ant-file>`,
            },
          ],
        });
      } else {
        throw new Error(
          "Anthropic models don't support the following media type: " +
            historyItem.kind,
        );
      }
    } else {
      historyItem satisfies never;
    }
  }
  return anthropicHistory;
}

// TODO: ensure this list is complete
const nonReasoningModels = [
  "claude-3-5-haiku-20241022",
  "claude-3-5-haiku-latest",
  "claude-3-5-haiku",
  "claude-3-haiku-20240307",
  "claude-3-haiku",
];

type AnthropicToolMap = {
  original: Tool<unknown, unknown>;
  anthropic: AnthropicTool;
  /** Anthropic doesn't allow non-objects at the top level but we want to. We therefore wrap the tool input with a wrapper object which need to unwrap at the output */
  wrapperObject: boolean;
  /** No parameter specified */
  isVoid: boolean;
};

export class AnthropicAdapter<zO, zI> {
  #client: Anthropic;
  #model: string;
  #output?: z.ZodType<zO, zI>;
  #normalizedTools: AnthropicToolMap[];

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
      let name = tool.name.toLowerCase().replaceAll(" ", "_").replace(
        /[^a-zA-Z0-9_-]/g,
        "",
      );
      if (!/^[a-zA-Z_]/.test(name)) {
        name = "_" + name; // Ensure name starts with letter or underscore
      }
      name = name.slice(0, 64); // Limit to 64 characters

      const isVoid = tool.parameters instanceof z.ZodVoid;
      const wrapperObject = !isVoid &&
        !(tool.parameters instanceof z.ZodObject);

      return {
        original: tool,
        anthropic: {
          name,
          input_schema: isVoid ? { type: "object" } : z.toJSONSchema(
            wrapperObject
              ? z.object({ content: tool.parameters })
              : tool.parameters,
          ) as any,
          description: tool.description,
        },
        wrapperObject,
        isVoid,
      };
    });
    this.#client = new Anthropic();
  }

  async run({ history, systemPrompt, signal }: {
    systemPrompt: string;
    history: ChatItem[];
    signal: AbortSignal;
  }): Promise<ChatItem[]> {
    const anthropicHistory = await getAnthropicHistory(
      history,
      this.#normalizedTools,
      signal,
    );
    const isReasoningModel = !nonReasoningModels.includes(this.#model);

    // TODO: implement structured outputs properly instead of this hack
    const response = await this.#client.messages.create({
      model: this.#model,
      system: systemPrompt +
        (this.#output
          ? `\n\n<system-requirement>Your final output message must match this below JSON schema exactly. Wrap your response in a code block (i.e. \`\`\`json \`\`\`) :\n${
            z.toJSONSchema(this.#output)
          }</system-requirement>`
          : ""),
      messages: anthropicHistory,
      tools: this.#normalizedTools.map(({ anthropic }) => anthropic),
      max_tokens: 16001,
      thinking: isReasoningModel
        ? {
          type: "enabled",
          budget_tokens: 16000,
        }
        : undefined,
    }, { signal });

    const output: ChatItem[] = [];

    for (const part of response.content) {
      if (part.type === "thinking") {
        output.push({
          type: "output_reasoning",
          content: part.thinking,
        });
        signatureMap.set(part.thinking, part.signature);
      } else if (part.type === "text") {
        if (this.#output) {
          const parsedBlock = part.text.split("```json")[1].split("```")[0]
            .trim();
          output.push({
            type: "output_text",
            content: parsedBlock,
          });
        } else {
          output.push({
            type: "output_text",
            content: part.text,
          });
        }
      } else if (part.type === "tool_use") {
        const tool = this.#normalizedTools.find((tool) =>
          tool.anthropic.name === part.name
        );
        output.push({
          type: "tool_use",
          tool_use_id: part.id,
          kind: tool?.original.name ?? part.name,
          content: tool?.isVoid ? undefined : (JSON.stringify(
            tool?.wrapperObject ? (part.input as any).content : part.input,
          )),
        });
      } else {
        console.warn("Unrecognized type", part);
      }
    }

    return output;
  }

  async *stream({ history, systemPrompt, signal }: {
    systemPrompt: string;
    history: ChatItem[];
    signal: AbortSignal;
  }): AsyncStreamItemGenerator {
    const anthropicHistory = await getAnthropicHistory(
      history,
      this.#normalizedTools,
      signal,
    );
    const isReasoningModel = !nonReasoningModels.includes(this.#model);

    // TODO: implement structured outputs properly instead of this hack
    const response = this.#client.messages.stream({
      model: this.#model,
      system: systemPrompt,
      messages: anthropicHistory,
      tools: this.#normalizedTools.map(({ anthropic }) => anthropic),
      max_tokens: 16001,
      thinking: isReasoningModel
        ? {
          type: "enabled",
          budget_tokens: 16000,
        }
        : undefined,
    }, { signal });

    const parts: ChatItem[] = [];
    for await (const part of response) {
      if (part.type === "content_block_delta") {
        const { delta } = part;
        if (delta.type === "text_delta") {
          if (!parts[part.index]) {
            parts[part.index] = { type: "output_text", content: "" };
          }
          parts[part.index].content += delta.text;
          yield {
            type: "delta_output_text",
            delta: delta.text,
            index: part.index,
          };
        } else if (delta.type === "thinking_delta") {
          if (!parts[part.index]) {
            parts[part.index] = { type: "output_reasoning", content: "" };
          }
          parts[part.index].content += delta.thinking;
          yield {
            type: "delta_output_reasoning",
            delta: delta.thinking,
            index: part.index,
          };
        } else if (delta.type === "signature_delta") {
          const thinkingPart = parts[part.index];
          assert(thinkingPart.type === "output_reasoning");
          signatureMap.set(thinkingPart.content, delta.signature);
        } else if (delta.type === "input_json_delta") {
          parts[part.index].content += delta.partial_json;
        }
      } else if (part.type === "content_block_start") {
        if (part.content_block.type === "tool_use") {
          parts[part.index] = {
            type: "tool_use",
            kind: part.content_block.name,
            tool_use_id: part.content_block.id,
            content: "",
          };
        }
      } else if (part.type === "content_block_stop") {
        const endingPart = parts[part.index];
        if (endingPart.type === "tool_use") {
          const tool = this.#normalizedTools.find((tool) =>
            tool.anthropic.name === endingPart.kind
          );
          yield {
            type: "tool_use",
            index: part.index,
            kind: tool?.original.name ?? endingPart.kind,
            tool_use_id: endingPart.tool_use_id,
            content: endingPart.content
              ? (tool?.wrapperObject
                ? JSON.stringify(JSON.parse(endingPart.content).content)
                : endingPart.content)
              : undefined,
          };
        }
      }
    }
  }
}
