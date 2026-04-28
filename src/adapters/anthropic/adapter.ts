import Anthropic from "@anthropic-ai/sdk";
import { type ClassifiedError, createClassifiedError } from "../../errors.ts";
import type { AdapterStreamIterator, ChatItem } from "../../types.ts";
import type { Adapter, AdapterStreamOptions } from "../adapter.ts";
import type { SchemaCompatibility } from "../shared/schema_compatibility.ts";
import { getAnthropicHistory, rememberAnthropicReasoningSignature } from "./history.ts";
import {
  type AnthropicModels,
  anthropicModelStructuredOutputSupport,
  anthropicModelThinkingSupport,
  type EffortLevel,
  getAnthropicMessagesStreamConfig,
  type SupportedEffortLevel,
  type SupportedThinkingLevel,
  type SupportsInterleaved,
  type ThinkingDisplay,
  type ThinkingLevel,
} from "./models.ts";
import { createAnthropicCompatibleSchema, normalizeAnthropicTools } from "./utils.ts";
import { crossPlatformEnv, requireEnv } from "../../util.ts";

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

function supportsNativeStructuredOutput<TModel extends AnthropicModels>(model: TModel) {
  return anthropicModelStructuredOutputSupport[model];
}

// Only include thinkingLevel for models with extended thinking (adaptive-only models don't use budget_tokens).
type ThinkingLevelOption<TModel extends AnthropicModels> = SupportedThinkingLevel<TModel> extends never ? undefined
  : SupportedThinkingLevel<TModel>;

// Only include effort for models that support it (extended-only models don't have an effort API).
type EffortOption<TModel extends AnthropicModels> = SupportedEffortLevel<TModel> extends never ? undefined
  : SupportedEffortLevel<TModel>;

// Only include interleaved for models with extended thinking; adaptive models handle it automatically.
type InterleavedOption<TModel extends AnthropicModels> = SupportsInterleaved<TModel> extends true ? boolean
  : undefined;

export function anthropicModel<zO, zI, TModel extends AnthropicModels>(options: {
  model: TModel;
  effort?: EffortOption<TModel>;
  thinkingLevel?: ThinkingLevelOption<TModel>;
  interleaved?: InterleavedOption<TModel>;
  thinkingDisplay?: ThinkingDisplay;
  baseUrl?: string;
  apiKey?: string;
  client?: Anthropic;
}): Adapter<zO, zI> {
  const modelConfig = anthropicModelThinkingSupport[options.model];
  // I know, terrifying, someone should fix this tbh it's super super scary
  const thinkingLevel =
    ("schema" in modelConfig ? (options.thinkingLevel ?? modelConfig.schema.parse(undefined)) : undefined) as
      | SupportedThinkingLevel<TModel>
      | undefined;
  const effort =
    ("effortSchema" in modelConfig ? (options.effort ?? modelConfig.effortSchema.parse(undefined)) : undefined) as
      | SupportedEffortLevel<TModel>
      | undefined;
  const thinkingDisplay = options.thinkingDisplay ?? "summarized";
  const interleaved = options.interleaved;
  const streamConfig = getAnthropicMessagesStreamConfig({
    model: options.model,
    thinkingLevel: thinkingLevel as ThinkingLevel | undefined,
    effort: effort as EffortLevel | undefined,
    thinkingDisplay: thinkingDisplay,
    interleaved: interleaved,
  });
  // scaryness over
  const client = options.client ??
    new Anthropic({
      apiKey: options.apiKey ?? requireEnv("ANTHROPIC_API_KEY"),
      baseURL: options.baseUrl ?? crossPlatformEnv("ANTHROPIC_BASE_URL") ?? "https://api.anthropic.com",
    });

  function getSystemPrompt(instructions: string, structuredOutput?: SchemaCompatibility): string {
    if (!structuredOutput) {
      return instructions;
    }

    if (supportsNativeStructuredOutput(options.model)) {
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

  return {
    provider: "Anthropic",
    model: options.model,
    stream: async function* stream<zO, zI>(
      { history, instructions, tools, signal, output }: AdapterStreamOptions<zO, zI>,
    ): AdapterStreamIterator {
      const normalizedTools = normalizeAnthropicTools(tools);
      const anthropicHistory = await getAnthropicHistory({ history, normalizedTools, signal });

      const structuredOutput = output && createAnthropicCompatibleSchema(output, {
        kind: "output",
        rootPath: "output",
      });

      const systemPrompt = getSystemPrompt(instructions, structuredOutput);

      const response = client.beta.messages.stream({
        model: options.model,
        system: systemPrompt,
        messages: anthropicHistory,
        tools: normalizedTools.map(({ anthropic }) => anthropic),

        max_tokens: 16001,

        betas: streamConfig.betas,
        thinking: streamConfig.thinking,
        output_config: {
          ...streamConfig.output_config,
          format: supportsNativeStructuredOutput(options.model) && structuredOutput
            ? { type: "json_schema", schema: structuredOutput.jsonSchema }
            : undefined,
        },
      }, { signal });

      const parts: ChatItem[] = [];
      const reasoningSignatures = new Map<number, string>();
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
            if (!parts[part.index]) {
              parts[part.index] = { type: "output_reasoning", content: "" };
            }
            reasoningSignatures.set(part.index, delta.signature);
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
              kind: tool?.original.name ?? block.name,
              tool_use_id: block.id,
            };
          } else if (part.content_block.type === "thinking") {
            parts[part.index] = {
              type: "output_reasoning",
              content: part.content_block.thinking,
            };
            if (part.content_block.signature) {
              reasoningSignatures.set(part.index, part.content_block.signature);
            }
            if (part.content_block.thinking) {
              yield {
                type: "delta_output_reasoning",
                delta: part.content_block.thinking,
                index: part.index,
              };
            }
          }
        } else if (part.type === "content_block_stop") {
          const endingPart = parts[part.index];
          if (endingPart?.type === "output_reasoning") {
            const signature = reasoningSignatures.get(part.index);
            if (signature && endingPart.content) {
              rememberAnthropicReasoningSignature(endingPart.content, signature);
            }
            reasoningSignatures.delete(part.index);
          } else if (endingPart.type === "tool_use") {
            const tool = normalizedTools.find((tool) => tool.anthropic.name === endingPart.kind);
            const restoredContent = endingPart.content
              ? JSON.stringify(
                tool?.compatibility
                  ? tool.compatibility.fromProvider(JSON.parse(endingPart.content))
                  : JSON.parse(endingPart.content),
              )
              : undefined;
            yield {
              type: "tool_use",
              index: part.index,
              kind: tool?.original.name ?? endingPart.kind,
              tool_use_id: endingPart.tool_use_id,
              content: restoredContent,
            };
          } else if (endingPart.type === "output_text" && structuredOutput) {
            const rawJson = supportsNativeStructuredOutput(options.model)
              ? endingPart.content
              : extractJson(endingPart.content);

            let structuredContent: string;
            try {
              structuredContent = JSON.stringify(
                structuredOutput.fromProvider(JSON.parse(rawJson)),
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
    },
    classifyError(error: unknown): ClassifiedError | null {
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
    },
  };
}
