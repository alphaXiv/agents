import Anthropic from "@anthropic-ai/sdk";
import { assert } from "@std/assert";
import { type ClassifiedError, createClassifiedError } from "../../errors.ts";
import { isStructuredOutputRetryFeedback } from "../../constants.ts";
import type { AdapterStreamIterator, ChatItem } from "../../types.ts";
import { Adapter, type AdapterStreamOptions } from "../adapter.ts";
import {
  type AnthropicMessagesStreamConfig,
  type AnthropicModels,
  anthropicModelStructuredOutputSupport,
} from "./models.ts";
import {
  type AnthropicCompatibleSchema,
  type AnthropicToolMap,
  createAnthropicCompatibleSchema,
  normalizeAnthropicTools,
} from "./utils.ts";

// TODO: drop signature after 10 minutes or whatever
// Mapping between thinking response and signature since signature is meaningless cross-provider and we technically only need to include thinking for the one step
const signatureMap = new Map<string, string>();

const supportedImageMimeTypes = ["image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp"];

/**
 * Extracts JSON from model output that may contain prose before/after a code block.
 * Precedence:
 *   1. First ```json ... ``` block found anywhere in the text.
 *   2. First ``` ... ``` block found anywhere in the text.
 *   3. The full text trimmed (no code block present).
 */
function extractJson(text: string): string {
  const jsonBlock = text.match(/```json\s*\n?([\s\S]*?)\n?```/);
  if (jsonBlock) return jsonBlock[1].trim();
  const anyBlock = text.match(/```\s*\n?([\s\S]*?)\n?```/);
  if (anyBlock) return anyBlock[1].trim();
  return text.trim();
}

export interface AnthropicAdapterOptions<TModel extends AnthropicModels> {
  model: TModel;
  client: Anthropic;
  streamConfig: AnthropicMessagesStreamConfig;
}

export class AnthropicAdapter<TModel extends AnthropicModels> extends Adapter<TModel> {
  name = "Anthropic";
  #client: Anthropic;
  #streamConfig: AnthropicMessagesStreamConfig;

  constructor(options: AnthropicAdapterOptions<TModel>) {
    super(options);
    this.#client = options.client;
    this.#streamConfig = options.streamConfig;
  }

  get supportsNativeStructuredOutput(): boolean {
    return anthropicModelStructuredOutputSupport[this.model];
  }

  override classifyError(error: unknown): ClassifiedError | null {
    if (error instanceof Anthropic.APIConnectionTimeoutError) {
      return createClassifiedError("timeout", error, error.status);
    } else if (error instanceof Anthropic.APIConnectionError) {
      return createClassifiedError("network", error, error.status);
    } else if (error instanceof Anthropic.RateLimitError) {
      return createClassifiedError("rate_limit", error, error.status);
    } else if (error instanceof Anthropic.AuthenticationError) {
      return createClassifiedError("auth", error, error.status);
    } else if (error instanceof Anthropic.PermissionDeniedError) {
      return createClassifiedError("auth", error, error.status);
    } else if (error instanceof Anthropic.InternalServerError) {
      return createClassifiedError("server", error, error.status);
    } else if (error instanceof Anthropic.APIUserAbortError) {
      return createClassifiedError("aborted", error, error.status);
    }
    return null;
  }

  async getHistory(
    history: ChatItem[],
    normalizedTools: AnthropicToolMap[],
    signal: AbortSignal,
  ): Promise<Anthropic.Messages.MessageParam[]> {
    const anthropicHistory: Anthropic.Messages.MessageParam[] = [];
    let anthropicToolFileBuffer: Anthropic.Messages.MessageParam[] = [];

    // Put all of the history in place
    for (const historyItem of history) {
      switch (historyItem.type) {
        case "input_text": {
          // first, flush tool buffer
          anthropicHistory.push(...anthropicToolFileBuffer);
          anthropicToolFileBuffer = [];

          // next, append message
          anthropicHistory.push({
            role: "user",
            content: [{ type: "text", text: historyItem.content }],
          });
          break;
        }
        case "output_text": {
          // first, flush tool buffer
          anthropicHistory.push(...anthropicToolFileBuffer);
          anthropicToolFileBuffer = [];

          // next, append message
          anthropicHistory.push({
            role: isStructuredOutputRetryFeedback(historyItem.content) ? "user" : "assistant",
            content: [{ type: "text", text: historyItem.content }],
          });
          break;
        }
        case "context_summary": {
          anthropicHistory.push(...anthropicToolFileBuffer);
          anthropicToolFileBuffer = [];

          anthropicHistory.push({
            role: "user",
            content: [{ type: "text", text: historyItem.content }],
          });
          break;
        }
        case "tool_use": {
          const tool = normalizedTools.find((tool) => tool.original.normalizedName === historyItem.kind);
          const content = historyItem.content ? JSON.parse(historyItem.content) : {};
          anthropicHistory.push({
            role: "assistant",
            content: [{
              type: "tool_use",
              id: historyItem.tool_use_id,
              name: tool?.anthropic.name ?? historyItem.kind,
              input: tool?.compatibility ? tool.compatibility.toAnthropic(content) : content,
            }],
          });
          break;
        }
        case "tool_result_text": {
          anthropicHistory.push({
            role: "user",
            content: [{
              type: "tool_result",
              tool_use_id: historyItem.tool_use_id,
              content: historyItem.content,
              is_error: historyItem.content.startsWith("Error: "),
            }],
          });
          break;
        }
        case "output_reasoning": {
          // first, flush tool buffer
          anthropicHistory.push(...anthropicToolFileBuffer);
          anthropicToolFileBuffer = [];

          // next append reasoning
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
          break;
        }
        case "input_file":
        case "tool_result_file": {
          const pushBuffer = historyItem.type === "input_file" ? anthropicHistory : anthropicToolFileBuffer;
          if (supportedImageMimeTypes.includes(historyItem.kind)) {
            pushBuffer.push({
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
            pushBuffer.push({
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

            pushBuffer.push({
              role: "user",
              content: [
                {
                  type: "text",
                  text: `<ant-file>${text}</ant-file>`,
                },
              ],
            });
          } else {
            throw new Error(`Anthropic models don't support the following media type: ${historyItem.kind}`);
          }
          break;
        }
        default:
          historyItem satisfies never;
      }
    }

    // Flush remaining toolFileBuffer
    anthropicHistory.push(...anthropicToolFileBuffer);
    anthropicToolFileBuffer = [];

    return anthropicHistory;
  }

  getSystemPrompt(instructions: string, structuredOutput?: AnthropicCompatibleSchema): string {
    if (!structuredOutput) {
      return instructions;
    }

    if (this.supportsNativeStructuredOutput) {
      if (structuredOutput.instructions) {
        return `${structuredOutput.instructions}\n\n${instructions}`;
      }
      return instructions;
    }

    const SCHEMA_ADHERENCE_PROMPT = `\
<output_requirements>
You MUST respond with a JSON object that strictly adheres to provided JSON Schema.
You MUST NOT include any text that is not part of the JSON object.
You MUST wrap the JSON in a \`\`\`json ... \`\`\` code block and you MUST ensure the content of the code block is valid JSON that adheres to the schema.
The JSON provided by you does not have to be formatted in any particular way, but it MUST be parseable by a standard JSON parser.

Here is the JSON Schema you must adhere to:
\`\`\`json_schema
${JSON.stringify(structuredOutput.originalJsonSchema, null, 2)}
\`\`\`
</output_requirements>`;

    return `${instructions}\n\n${SCHEMA_ADHERENCE_PROMPT}`;
  }

  async *stream<zO, zI>(
    { history, instructions, tools, signal, output }: AdapterStreamOptions<zO, zI>,
  ): AdapterStreamIterator {
    const normalizedTools = normalizeAnthropicTools(tools);
    const anthropicHistory = await this.getHistory(history, normalizedTools, signal);

    const structuredOutput = output && createAnthropicCompatibleSchema(output, {
      kind: "output",
      rootPath: "output",
    });

    const systemPrompt = this.getSystemPrompt(instructions, structuredOutput);

    const streamConfig = this.#streamConfig;
    const response = this.#client.beta.messages.stream({
      model: this.model,
      system: systemPrompt,
      messages: anthropicHistory,
      tools: normalizedTools.map(({ anthropic }) => anthropic),

      max_tokens: 16001,

      betas: streamConfig.betas,
      thinking: streamConfig.thinking,
      output_config: {
        ...streamConfig.output_config,
        format: this.supportsNativeStructuredOutput && structuredOutput
          ? { type: "json_schema", schema: structuredOutput.jsonSchema }
          : undefined,
      },
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
          if (!structuredOutput) {
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
          parts[part.index].content += delta.partial_json;
        }
      } else if (part.type === "content_block_start") {
        if (part.content_block.type === "tool_use") {
          const block = part.content_block;
          const tool = normalizedTools.find((tool) => tool.anthropic.name === block.name);
          parts[part.index] = {
            type: "tool_use",
            kind: block.name,
            tool_use_id: block.id,
            content: "",
          };
          yield {
            type: "tool_use_start",
            index: part.index,
            kind: tool?.original.normalizedName ?? block.name,
            tool_use_id: block.id,
          };
        }
      } else if (part.type === "content_block_stop") {
        const endingPart = parts[part.index];
        if (endingPart.type === "tool_use") {
          const tool = normalizedTools.find((tool) => tool.anthropic.name === endingPart.kind);
          const restoredContent = endingPart.content
            ? JSON.stringify(
              tool?.compatibility
                ? tool.compatibility.fromAnthropic(JSON.parse(endingPart.content))
                : JSON.parse(endingPart.content),
            )
            : undefined;
          yield {
            type: "tool_use",
            index: part.index,
            kind: tool?.original.normalizedName ?? endingPart.kind,
            tool_use_id: endingPart.tool_use_id,
            content: restoredContent,
          };
        } else if (endingPart.type === "output_text" && structuredOutput) {
          const rawJson = this.supportsNativeStructuredOutput ? endingPart.content : extractJson(endingPart.content);

          let structuredContent: string;
          try {
            structuredContent = JSON.stringify(
              structuredOutput.fromAnthropic(JSON.parse(rawJson)),
            );
          } catch {
            structuredContent = rawJson;
          }

          endingPart.content = structuredContent;
          yield {
            type: "delta_output_text",
            delta: structuredContent,
            index: part.index,
          };
        }
      }
    }

    const final = await response.finalMessage();
    return {
      outputTokens: final.usage.output_tokens,
      inputTokens: final.usage.input_tokens,
    };
  }
}
