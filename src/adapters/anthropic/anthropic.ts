/**
 * Adapter implementation for Anthropic-style APIs using `@anthropic-ai/sdk`.
 * ```ts
 * const adapter = anthropicAdapter({
 *   name: "anthropic", // display name according to who is running the API
 *   url: "https://api.anthropic.com",
 *   apiKey: process.env.ANTHROPIC_API_KEY,
 * });
 *
 * const agent = new Agent({
 *   adapter,
 *   model: "claude-4.6-sonnet"
 * });
 * ```
 * If you are using the default provider and API key environment variable, you
 * can omit the `adapter` property and use a unified model string like
 * `model: "anthropic:claude-4.6-sonnet"`.
 * @module
 */
import Anthropic from "@anthropic-ai/sdk";
import z from "zod";
import type {
  Adapter,
  AdapterStreamOptions,
  AdapterTypeOptions,
} from "../../adapters.ts";
import type {
  AdapterStreamIterator,
  ChatItem,
  ReasoningEffort,
} from "../../types.ts";
import { assert } from "@std/assert/assert";
import {
  getAnthropicHistory,
  normalizeAnthropicTools,
  signatureMap,
} from "./utils.ts";

const maxTokensMap: Record<string, number> = {
  "claude-3-haiku-20240307": 4000,
  "claude-3-5-haiku-20241022": 8000,
  "claude-3-5-haiku-latest": 8000,
};

// TODO: ensure this list is complete
const nonReasoningModels = [
  "claude-3-5-haiku-20241022",
  "claude-3-5-haiku-latest",
  "claude-3-5-haiku",
  "claude-3-haiku-20240307",
  "claude-3-haiku",
];

function getAnthropicThinking(
  model: string,
  reasoningEffort: ReasoningEffort,
) {
  const isReasoningModel = !nonReasoningModels.includes(model);
  return isReasoningModel && reasoningEffort === "normal"
    ? {
      type: "enabled" as const,
      budget_tokens: 16000,
    }
    : undefined;
}

function getAnthropicSystemPrompt<zO, zI>(
  systemPrompt: string,
  output?: z.ZodType<zO, zI>,
) {
  if (!output) return systemPrompt;

  return systemPrompt +
    `\n\n<system-requirement>Your final output message must match this below JSON schema exactly. Wrap your response in a code block (i.e. \`\`\`json \`\`\`) :\n${
      z.toJSONSchema(output)
    }</system-requirement>`;
}

export interface AnthropicAdapterOptions {
  name: string;
  /** example: "https://api.anthropic.com" */
  url: string;
  apiKey: string;
}

export function anthropicAdapter<Models extends string>(
  options: AnthropicAdapterOptions & AdapterTypeOptions<Models>,
): Adapter<Models> {
  const anthropic = new Anthropic({
    apiKey: options.apiKey,
    baseURL: options.url,
  });

  async function* stream<zO, zI>({
    model,
    output,
    tools,
    reasoningEffort,
    systemPrompt,
    history,
    signal,
  }: AdapterStreamOptions<zO, zI, Models>): AdapterStreamIterator {
    const normalizedTools = normalizeAnthropicTools(tools);
    const anthropicHistory = await getAnthropicHistory(
      history,
      normalizedTools,
      signal,
    );

    const response = anthropic.beta.messages.stream({
      model,
      system: getAnthropicSystemPrompt(systemPrompt, output),
      messages: anthropicHistory,
      tools: normalizedTools.map(({ anthropic }) => anthropic),
      max_tokens: maxTokensMap[model] ?? 16001,
      betas: ["context-1m-2025-08-07"],
      thinking: getAnthropicThinking(model, reasoningEffort),
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
          const block = part.content_block;
          const tool = normalizedTools.find((tool) =>
            tool.anthropic.name === block.name
          );
          parts[part.index] = {
            type: "tool_use",
            kind: block.name,
            tool_use_id: block.id,
            content: "",
          };
          yield {
            type: "tool_use_start",
            index: part.index,
            kind: tool?.original.name ?? block.name,
            tool_use_id: block.id,
          };
        }
      } else if (part.type === "content_block_stop") {
        const endingPart = parts[part.index];
        if (endingPart.type === "tool_use") {
          const tool = normalizedTools.find((tool) =>
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

    if (output) {
      const part = parts.at(-1);
      if (part && part.type === "output_text") {
        const parsedBlock = part.content.split("```json")[1].split("```")[0]
          .trim();
        yield {
          type: "delta_output_text",
          delta: parsedBlock,
          index: parts.length,
        };
      }
    }

    const final = await response.finalMessage();
    return {
      outputTokens: final.usage.output_tokens,
      inputTokens: final.usage.input_tokens,
    };
  }

  return { name: "anthropic", stream };
}
