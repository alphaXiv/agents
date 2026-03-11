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
import type { Tool as AnthropicTool } from "@anthropic-ai/sdk/resources/messages/messages";
import z from "zod";
import type {
  Adapter,
  AdapterStreamOptions,
  AdapterTypeOptions,
} from "../adapters.ts";
import type { Tool } from "../tool.ts";
import type {
  AdapterStreamIterator,
  ChatItem,
  ReasoningEffort,
} from "../types.ts";
import { assert } from "@std/assert/assert";

const supportedImageMimeTypes = [
  "image/jpeg",
  "image/jpg",
  "image/png",
  "image/gif",
  "image/webp",
];

const maxTokensMap: Record<string, number> = {
  "claude-3-haiku-20240307": 4000,
  "claude-3-5-haiku-20241022": 8000,
  "claude-3-5-haiku-latest": 8000,
};

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
  original: Tool<unknown, unknown, unknown>;
  anthropic: AnthropicTool;
  /** Anthropic doesn't allow non-objects at the top level but we want to. We therefore wrap the tool input with a wrapper object which need to unwrap at the output */
  wrapperObject: boolean;
  /** No parameter specified */
  isVoid: boolean;
};

function normalizeAnthropicTools(
  tools: Tool<unknown, unknown, unknown>[],
): AnthropicToolMap[] {
  return tools.map((tool) => {
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
          // deno-lint-ignore no-explicit-any
        ) as any,
        description: tool.description,
      },
      wrapperObject,
      isVoid,
    };
  });
}

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

function extractStructuredOutput(text: string) {
  const codeBlockMatch = text.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  return (codeBlockMatch?.[1] ?? text).trim();
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
          if (!output) {
            yield {
              type: "delta_output_text",
              delta: delta.text,
              index: part.index,
            };
          }
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
          const toolPart = parts[part.index];
          assert(toolPart.type === "tool_use");
          toolPart.content = (toolPart.content ?? "") + delta.partial_json;
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
        if (!endingPart) continue;

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
        } else if (endingPart.type === "output_text" && output) {
          yield {
            type: "delta_output_text",
            delta: extractStructuredOutput(endingPart.content),
            index: part.index,
          };
        }
      }
    }
  }

  return { name: "anthropic", stream };
}
