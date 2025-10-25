import Anthropic from "@anthropic-ai/sdk";
import type { Tool as AnthropicTool } from "@anthropic-ai/sdk/resources/messages/messages";
import z from "zod";
import { assert } from "@std/assert";
import type { Tool } from "../tool.ts";
import type { ChatItem } from "../types.ts";

// TODO: drop signature after 10 minutes or whatever
// Mapping between thinking response and signature since signature is meaningless cross-provider and we technically only need to include thinking for the one step
const signatureMap = new Map<string, string>();

export class AnthropicAdapter<zO, zI> {
  #client: Anthropic;
  #model: string;
  #output?: z.ZodType<zO, zI>;
  #normalizedTools: {
    original: Tool<unknown, unknown>;
    anthropic: AnthropicTool;
    /** Anthropic doesn't allow non-objects at the top level but we want to. We therefore wrap the tool input with a wrapper object which need to unwrap at the output */
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
      let name = tool.name.toLowerCase().replaceAll(" ", "_").replace(
        /[^a-zA-Z0-9_-]/g,
        "",
      );
      if (!/^[a-zA-Z_]/.test(name)) {
        name = "_" + name; // Ensure name starts with letter or underscore
      }
      name = name.slice(0, 64); // Limit to 64 characters

      const wrapperObject = !(tool.parameters instanceof z.ZodObject);

      return {
        original: tool,
        anthropic: {
          name,
          input_schema: z.toJSONSchema(
            wrapperObject
              ? z.object({ content: tool.parameters })
              : tool.parameters,
          ) as any,
          description: tool.description,
        },
        wrapperObject,
      };
    });
    this.#client = new Anthropic();
  }

  async run({ history, systemPrompt }: {
    systemPrompt: string;
    history: ChatItem[];
  }): Promise<ChatItem[]> {
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
        const tool = this.#normalizedTools.find((tool) =>
          tool.original.name === historyItem.name
        );
        assert(tool);
        const content = JSON.parse(historyItem.content);
        anthropicHistory.push({
          role: "assistant",
          content: [{
            type: "tool_use",
            id: historyItem.tool_use_id,
            name: tool.anthropic.name,
            input: tool.wrapperObject ? { content } : content,
          }],
        });
      } else if (historyItem.type === "tool_result") {
        anthropicHistory.push({
          role: "user",
          content: [{
            type: "tool_result",
            tool_use_id: historyItem.tool_use_id,
            content: historyItem.content,
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
      } else {
        historyItem satisfies never;
      }
    }

    const isReasoningModel = true; // TODO: implement

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
    });

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
        assert(tool);
        output.push({
          type: "tool_use",
          tool_use_id: part.id,
          name: tool.original.name,
          content: JSON.stringify(
            tool.wrapperObject ? (part.input as any).content : part.input,
          ),
        });
      } else {
        console.warn("Unrecognized type", part);
      }
    }

    return output;
  }
}
