/**
 * Adapter implementation for OpenAI's Responses API using the `openai` package.
 * ```ts
 * const adapter = openAiResponsesAdapter({
 *   name: "openai", // display name according to who is running the API
 *   url: "https://api.openai.com/v1",
 *   apiKey: process.env.OPENAI_API_KEY,
 * });
 *
 * const agent = new Agent({
 *   adapter,
 *   model: "gpt-5.4"
 * });
 * ```
 * If you are using the default provider and API key environment variable, you
 * can omit the `adapter` property and use a unified model string like
 * `model: "openai:gpt-5.4"`.
 * @module
 */
import OpenAI from "openai";
import z from "zod";
import { assert } from "@std/assert";
import type {
  ResponseInputFile,
  ResponseInputImage,
  ResponseInputItem,
  ResponseInputText,
} from "openai/resources/responses/responses";
import type { Tool } from "../tool.ts";
import type {
  AdapterStreamIterator,
  ChatItem,
  ChatItemInputFile,
  ChatItemToolResultFile,
  ReasoningEffort,
} from "../types.ts";
import type {
  Adapter,
  AdapterStreamOptions,
  AdapterTypeOptions,
} from "../adapters.ts";
import { encodeBase64 } from "@std/encoding";
import { RETRY_RESUMABILITY_PROMPT } from "../constants.ts";

const supportedImageMimeTypes = [
  "image/jpeg",
  "image/jpg",
  "image/png",
  "image/gif",
  "image/webp",
];

// TODO: ensure this list is complete
const nonReasoningModels = [
  "gpt-4.1",
];

export interface OpenaiResponsesAdapterOptions {
  name: string;

  /** example: "http://localhost:1234/v1" */
  url: string;
  apiKey: string;

  /**
   * @deprecated please set this to "function"
   * @default "tool"
   */
  toolCallFlavor?: "tool" | "function";
}

export function openAiResponsesAdapter<Models extends string>(
  options: OpenaiResponsesAdapterOptions & AdapterTypeOptions<Models>,
): Adapter<Models> {
  const openai = new OpenAI({
    baseURL: options.url,
    apiKey: options.apiKey,
  });

  async function* stream<zO, zI>({
    model,
    output,
    tools,
    reasoningEffort,
    systemPrompt,
    history: history,
    signal,
  }: AdapterStreamOptions<zO, zI, Models>): AdapterStreamIterator {
    const normalizedTools: OpenAIToolMap[] = tools.map((tool) => {
      // TODO: improve this mapping
      const name = tool.name.toLowerCase().replaceAll(" ", "_").replace(
        /[^a-zA-Z0-9_-]/g,
        "",
      );

      const isVoid = tool.parameters instanceof z.ZodVoid;
      const wrapperObject = !(tool.parameters instanceof z.ZodObject);

      return {
        original: tool,
        openai: {
          name,
          parameters: isVoid
            ? { type: "object", properties: {}, additionalProperties: false }
            : z.toJSONSchema(
              wrapperObject
                ? z.object({ content: tool.parameters })
                : tool.parameters,
            ),
          description: tool.description,
          type: "function",
          strict: false,
        },
        wrapperObject,
      };
    });

    const input = await getOpenAIHistory(
      history,
      normalizedTools,
      signal,
      options.toolCallFlavor ?? "tool",
    );

    const request: OpenAI.Responses.ResponseCreateParamsStreaming = {
      model,
      input,
      instructions: systemPrompt,
      tools: normalizedTools.map(({ openai }) => openai),
      text: {
        format: output
          ? {
            type: "json_schema",
            name: "result",
            schema: z.toJSONSchema(output),
          }
          : { type: "text" },
      },
      reasoning: {
        summary: "auto",
        effort: getModelReasoning(model, reasoningEffort),
      },
      stream: true,
    };
    const response = openai.responses.stream(request, { signal });

    const toolIndex: ChatItem[] = [];
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
          const tool = normalizedTools
            .find((tool) => tool.openai.name === partItem.name);
          toolIndex[part.output_index] = {
            type: "tool_use",
            tool_use_id: partItem.call_id,
            kind: tool?.original.name ?? partItem.name,
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
      } else if (part.type === "response.reasoning_text.delta") {
        yield {
          type: "delta_output_reasoning",
          delta: part.delta,
          index: part.output_index,
        };
      } else if (part.type === "response.function_call_arguments.done") {
        const toolUse = toolIndex[part.output_index];
        assert(toolUse.type === "tool_use");
        const tool = normalizedTools
          .find((tool) => tool.original.name === toolUse.kind);
        const content = tool?.wrapperObject
          ? JSON.stringify(JSON.parse(part.arguments).content)
          : part.arguments;
        yield {
          type: "tool_use",
          tool_use_id: toolUse.tool_use_id,
          kind: tool?.original.name ?? toolUse.kind,
          content,
          index: part.output_index,
        };
      }
    }
  }

  return { name: options.name, stream };
}

function getModelReasoning(
  model: string,
  idealReasoning: ReasoningEffort,
): OpenAI.ReasoningEffort | undefined {
  if (nonReasoningModels.includes(model)) {
    return undefined;
  }
  if (idealReasoning === "normal") {
    return undefined; // "just pick the default"
  } else if (idealReasoning === "minimal") {
    if (model === "gpt-5.4") { // gpt-5.4 doesn't support minimal reasoning
      return "none";
    }
    return "minimal";
  }
  idealReasoning satisfies never;
}

async function getOpenAIFile(
  historyItem: ChatItemToolResultFile | ChatItemInputFile,
  signal: AbortSignal,
): Promise<ResponseInputText | ResponseInputFile | ResponseInputImage> {
  if (supportedImageMimeTypes.includes(historyItem.kind)) {
    return {
      type: "input_image",
      image_url: historyItem.content,
      detail: "auto",
    };
  } else if (historyItem.kind === "application/pdf") {
    const req = await fetch(historyItem.content, { signal });
    const buffer = await req.arrayBuffer();
    const filename = historyItem.content.split("/").pop(); // TODO: make this heuristic better

    // TODO: check file support
    return {
      type: "input_file",
      file_data: `data:application/pdf;base64,${encodeBase64(buffer)}`, // TODO: investigate openai file api
      filename,
    };
  } else if (historyItem.kind.startsWith("text/")) {
    const req = await fetch(historyItem.content, { signal });
    const text = await req.text();

    return {
      type: "input_text",
      text: `<file>${text}</file>`,
    };
  } else {
    throw new Error(
      "OpenAI models don't support the following media type: " +
        historyItem.kind,
    );
  }
}

async function getOpenAIHistory(
  history: ChatItem[],
  toolMap: OpenAIToolMap[],
  signal: AbortSignal,
  flavor: "function" | "tool",
) {
  const toolOutputType = flavor === "function"
    ? "function_call_output"
    : "custom_tool_call_output";
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
      openAIHistory.push(
        flavor === "tool"
          ? {
            type: "custom_tool_call",
            call_id: historyItem.tool_use_id,
            name: tool?.openai.name ?? historyItem.kind,
            input: (tool?.wrapperObject
              ? `{"content":${historyItem.content}}`
              : historyItem.content) ?? "{}",
          }
          : {
            type: "function_call",
            name: tool?.openai.name ?? historyItem.kind,
            arguments: (tool?.wrapperObject
              ? `{"content":${historyItem.content}}`
              : historyItem.content) ?? "{}",
            call_id: historyItem.tool_use_id,
          },
      );
    } else if (historyItem.type === "tool_result_text") {
      const previousToolCallResult = openAIHistory.find((call) =>
        call.type === toolOutputType &&
        call.call_id === historyItem.tool_use_id
      );
      if (previousToolCallResult) {
        assert(previousToolCallResult.type === toolOutputType);
        assert(typeof previousToolCallResult.output !== "string");
        previousToolCallResult.output.push({
          type: "input_text",
          text: historyItem.content,
        });
      } else {
        openAIHistory.push(
          flavor === "tool"
            ? {
              type: "custom_tool_call_output",
              call_id: historyItem.tool_use_id,
              output: [{ type: "input_text", text: historyItem.content }],
            }
            : {
              type: "function_call_output",
              call_id: historyItem.tool_use_id,
              output: historyItem.content,
            },
        );
      }
    } else if (historyItem.type === "tool_result_file") {
      assert(
        flavor === "tool",
        "TODO: Provider doesn't support tool call file outputs",
      );
      const previousToolCallResult = openAIHistory.find((call) =>
        call.type === toolOutputType &&
        call.call_id === historyItem.tool_use_id
      );
      if (previousToolCallResult) {
        assert(previousToolCallResult.type === toolOutputType);
        assert(typeof previousToolCallResult.output !== "string");
        previousToolCallResult.output.push(
          await getOpenAIFile(historyItem, signal),
        );
      } else {
        openAIHistory.push({
          type: toolOutputType,
          call_id: historyItem.tool_use_id,
          output: [await getOpenAIFile(historyItem, signal)],
        });
      }
    } else if (historyItem.type === "input_file") {
      openAIHistory.push({
        type: "message",
        role: "user",
        content: [await getOpenAIFile(historyItem, signal)],
      });
    } else if (historyItem.type === "output_reasoning") {
      // no-op, don't propagate reasoning
    } else {
      historyItem satisfies never;
    }
  }

  // Check if this is a resumability request
  if (openAIHistory.length > 0) {
    const lastMessage = openAIHistory[openAIHistory.length - 1];
    if (lastMessage.type === "message" && lastMessage.role === "assistant") {
      openAIHistory.push({
        type: "message",
        role: "developer",
        content: RETRY_RESUMABILITY_PROMPT,
      });
    }
  }
  return openAIHistory;
}

type OpenAIToolMap = {
  original: Tool<unknown, unknown, unknown>;
  openai: OpenAI.Responses.FunctionTool;
  /** OpenAI doesn't allow non-objects at the top level but we want to. We therefore wrap the tool input with a wrapper object which need to unwrap at the output */
  wrapperObject: boolean;
};
