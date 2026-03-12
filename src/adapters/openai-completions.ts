/**
 * Adapter implementation for OpenAI-compatible chat completions APIs using the `openai` package.
 * ```ts
 * const adapter = openAiCompletionsAdapter({
 *   name: "my-provider",
 *   url: "https://api.example.com/v1",
 *   apiKey: process.env.MY_PROVIDER_API_KEY,
 * });
 *
 * const agent = new Agent({
 *   adapter,
 *   model: "gpt-4.1"
 * });
 * ```
 * @module
 */
import OpenAI from "openai";
import type {
  ChatCompletionFunctionTool,
  ChatCompletionMessageParam,
  ChatCompletionStreamParams,
} from "openai/resources/chat/completions/completions";
import z from "zod";
import type {
  Adapter,
  AdapterStreamOptions,
  AdapterTypeOptions,
} from "../adapters.ts";
import { RETRY_RESUMABILITY_PROMPT } from "../constants.ts";
import type { Tool } from "../tool.ts";
import type {
  AdapterStreamIterator,
  ChatItem,
  ChatItemToolUse,
} from "../types.ts";
import type { ReasoningEffort } from "../types.ts";

const TEXTLIKE_MIME_TYPES = [
  "text/*",
  "application/json",
  "application/*+json",
  "application/xml",
  "application/*+xml",
  "application/yaml",
  "application/*+yaml",
  "application/x-yaml",
  "application/toml",
  "application/javascript",
  "application/ecmascript",
  "application/typescript",
  "application/x-typescript",
];

const DEFAULT_SUPPORTED_MIME_TYPES = [
  "image/jpeg",
  "image/jpg",
  "image/png",
  "image/gif",
  "image/webp",
  "application/pdf",
  ...TEXTLIKE_MIME_TYPES,
];

type ToolMap = {
  original: Tool<unknown, unknown, unknown>;
  openai: ChatCompletionFunctionTool;
  wrapperObject: boolean;
  isVoid: boolean;
};

type PdfSupportConfig =
  | null
  | false
  | {
    mode: "native" | "text";
    maxSize?: number;
  };

type PdfSupport<Model extends string> =
  | PdfSupportConfig
  | ((model: Model) => PdfSupportConfig);

type FileHistoryItem = Extract<
  ChatItem,
  { type: "input_file" } | { type: "tool_result_file" }
>;

export interface OpenAiCompletionsAdapterOptions<Models extends string>
  extends AdapterTypeOptions<Models> {
  name: string;
  url: string;
  apiKey: string;
  supportedMimeTypes?: string[];
  pdfSupport?: PdfSupport<Models>;
  extraRequestBody?<zO, zI>(args: {
    model: Models;
    output?: z.ZodType<zO, zI>;
    normalizedTools: ToolMap[];
    reasoningEffort: ReasoningEffort;
  }): Record<string, unknown>;
}

function normalizeTools(
  tools: Tool<unknown, unknown, unknown>[],
): ToolMap[] {
  return tools.map((tool) => {
    const name = tool.name.toLowerCase().replaceAll(" ", "_").replace(
      /[^a-zA-Z0-9_-]/g,
      "",
    );

    const isVoid = tool.parameters instanceof z.ZodVoid;
    const wrapperObject = !isVoid &&
      !(tool.parameters instanceof z.ZodObject);

    return {
      original: tool,
      openai: {
        type: "function",
        function: {
          name,
          parameters: isVoid ? undefined : z.toJSONSchema(
            wrapperObject
              ? z.object({ content: tool.parameters })
              : tool.parameters,
          ),
          description: tool.description,
          strict: false,
        },
      },
      isVoid,
      wrapperObject,
    };
  });
}

function unsupportedMediaTypeError(model: string, mimeType: string) {
  return new Error(
    `Model ${JSON.stringify(model)} does not support media type ${
      JSON.stringify(mimeType)
    }`,
  );
}

function mimeTypeMatches(pattern: string, mimeType: string) {
  if (pattern === mimeType) return true;
  const escapedPattern = pattern.replace(
    /[|\\{}()[\]^$+?.]/g,
    "\\$&",
  ).replaceAll("*", ".*");
  return new RegExp(`^${escapedPattern}$`).test(mimeType);
}

function supportsMimeType(mimeType: string, supportedMimeTypes: string[]) {
  return supportedMimeTypes.some((pattern) =>
    mimeTypeMatches(pattern, mimeType)
  );
}

function isTextLikeMimeType(mimeType: string) {
  return supportsMimeType(mimeType, TEXTLIKE_MIME_TYPES);
}

function getPdfSupport<Model extends string>(
  pdfSupport: PdfSupport<Model> | undefined,
  model: Model,
): Exclude<PdfSupportConfig, null | false> | false {
  const resolved = typeof pdfSupport === "function"
    ? pdfSupport(model)
    : (pdfSupport ?? { mode: "native" as const });
  return resolved || false;
}

async function getContentLength(url: string, signal: AbortSignal) {
  const headResponse = await fetch(url, {
    method: "HEAD",
    signal,
  });
  const contentLength = headResponse.headers.get("Content-Length");
  if (contentLength) {
    return parseInt(contentLength, 10);
  }

  const response = await fetch(url, { signal });
  return (await response.arrayBuffer()).byteLength;
}

async function fetchTextFile(
  url: string,
  signal: AbortSignal,
) {
  const response = await fetch(url, { signal });
  const text = await response.text();
  return `<file>${text}</file>`;
}

async function fetchPdfAsText(url: string, signal: AbortSignal) {
  const response = await fetch(url, { signal });
  const { default: parsePdf } = await import("@lino/pdf-parse");
  const pdfText = await parsePdf(await response.arrayBuffer());
  return pdfText.text.join("\n");
}

async function getFileHistoryMessages<Models extends string>(
  options: OpenAiCompletionsAdapterOptions<Models>,
  model: Models,
  item: FileHistoryItem,
  signal: AbortSignal,
): Promise<ChatCompletionMessageParam[]> {
  const supportedMimeTypes = options.supportedMimeTypes ??
    DEFAULT_SUPPORTED_MIME_TYPES;
  if (!supportsMimeType(item.kind, supportedMimeTypes)) {
    throw unsupportedMediaTypeError(model, item.kind);
  }

  if (item.kind.startsWith("image/")) {
    return [{
      role: "user",
      content: [{
        type: "image_url",
        image_url: { url: item.content },
      }],
    }];
  }

  if (item.kind === "application/pdf") {
    const pdfSupport = getPdfSupport(options.pdfSupport, model);
    if (!pdfSupport) {
      throw unsupportedMediaTypeError(model, item.kind);
    }

    if (pdfSupport.mode === "native" && pdfSupport.maxSize !== undefined) {
      const contentLength = await getContentLength(item.content, signal);
      if (contentLength > pdfSupport.maxSize) {
        return [{
          role: "user",
          content: [{
            type: "text",
            text: await fetchPdfAsText(item.content, signal),
          }],
        }];
      }
    }

    if (pdfSupport.mode === "text") {
      return [{
        role: "user",
        content: [{
          type: "text",
          text: await fetchPdfAsText(item.content, signal),
        }],
      }];
    }

    return [{
      role: "user",
      content: [{
        type: "file",
        file: {
          file_data: item.content,
        },
      }],
    }];
  }

  if (isTextLikeMimeType(item.kind)) {
    return [{
      role: "user",
      content: [{
        type: "text",
        text: await fetchTextFile(
          item.content,
          signal,
        ),
      }],
    }];
  }

  return [{
    role: "user",
    content: [{
      type: "file",
      file: {
        file_data: item.content,
      },
    }],
  }];
}

async function getOpenAiCompletionsHistory<Models extends string>(
  options: OpenAiCompletionsAdapterOptions<Models>,
  model: Models,
  history: ChatItem[],
  systemPrompt: string,
  toolMap: ToolMap[],
  signal: AbortSignal,
) {
  const messages: ChatCompletionMessageParam[] = [{
    role: "system", // TODO: select right role for each model
    content: systemPrompt,
  }];

  for (const historyItem of history) {
    if (!historyItem) continue;

    if (historyItem.type === "input_text") {
      messages.push({
        role: "user",
        content: historyItem.content,
      });
    } else if (historyItem.type === "output_text") {
      messages.push({
        role: "assistant",
        content: historyItem.content,
      });
    } else if (historyItem.type === "tool_use") {
      const tool = toolMap.find((tool) =>
        tool.original.name === historyItem.kind
      );
      messages.push({
        role: "assistant",
        tool_calls: [
          {
            id: historyItem.tool_use_id,
            type: "function",
            function: {
              name: tool?.openai.function.name ?? historyItem.kind,
              arguments: (tool?.wrapperObject
                ? `{"content":${historyItem.content}}`
                : historyItem.content) ?? "{}",
            },
          },
        ],
      });
    } else if (historyItem.type === "tool_result_text") {
      messages.push({
        role: "tool",
        tool_call_id: historyItem.tool_use_id,
        content: historyItem.content,
      });
    } else if (
      historyItem.type === "input_file" ||
      historyItem.type === "tool_result_file"
    ) {
      messages.push(
        ...await getFileHistoryMessages(options, model, historyItem, signal),
      );
    } else if (historyItem.type === "output_reasoning") {
      // no-op, don't propagate reasoning
    } else {
      historyItem satisfies never;
    }
  }

  const lastMessage = messages[messages.length - 1];
  if (lastMessage?.role === "assistant") {
    messages.push({
      role: "developer",
      content: [{
        type: "text",
        text: RETRY_RESUMABILITY_PROMPT,
      }],
    });
  }

  return messages;
}

function getResponseFormat<zO, zI>(output?: z.ZodType<zO, zI>) {
  return output
    ? {
      type: "json_schema" as const,
      json_schema: {
        name: "result",
        // deno-lint-ignore no-explicit-any
        schema: z.toJSONSchema(output) as any,
      },
    }
    : { type: "text" as const };
}

function getToolUseContent(tool: ToolMap | undefined, rawArguments: string) {
  try {
    if (!rawArguments) {
      return tool?.isVoid ? undefined : rawArguments;
    }

    const parsed = JSON.parse(rawArguments);
    return tool?.isVoid
      ? undefined
      : (tool?.wrapperObject ? JSON.stringify(parsed.content) : rawArguments);
  } catch {
    return JSON.stringify(rawArguments);
  }
}

type PendingToolUse = {
  streamIndex: number;
  toolUse: ChatItemToolUse;
};

export function openAiCompletionsAdapter<Models extends string>(
  options: OpenAiCompletionsAdapterOptions<Models>,
): Adapter<Models> {
  if (
    options.pdfSupport !== undefined &&
    options.pdfSupport !== null &&
    options.pdfSupport !== false &&
    !supportsMimeType(
      "application/pdf",
      options.supportedMimeTypes ?? DEFAULT_SUPPORTED_MIME_TYPES,
    )
  ) {
    throw new Error(
      "pdfSupport requires application/pdf to be included in supportedMimeTypes",
    );
  }

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
    history,
    signal,
  }: AdapterStreamOptions<zO, zI, Models>): AdapterStreamIterator {
    const normalizedTools = normalizeTools(tools);
    const messages = await getOpenAiCompletionsHistory(
      options,
      model,
      history,
      systemPrompt,
      normalizedTools,
      signal,
    );

    const extraRequestBody = options.extraRequestBody?.({
      model,
      output,
      normalizedTools,
      reasoningEffort,
    }) ?? {};

    const request: ChatCompletionStreamParams = {
      model,
      messages,
      tools: normalizedTools.map(({ openai }) => openai),
      response_format: getResponseFormat(output),
      ...extraRequestBody,
    };

    const response = openai.chat.completions.stream(
      request,
      { signal },
    );

    const pendingToolUses = new Map<number, PendingToolUse>();
    let lastType = "";
    let lastIndex = -1;

    for await (const part of response) {
      const choice = part.choices[0];
      if (!choice) continue;
      const { delta } = choice;

      // @ts-expect-error Handle reasoning content, this is a openrouter-specific extension
      const reasoningDelta = delta.reasoning as string | undefined;
      if (reasoningDelta) {
        if (lastType !== "reasoning") {
          lastType = "reasoning";
          lastIndex++;
        }
        yield {
          type: "delta_output_reasoning",
          index: lastIndex,
          delta: reasoningDelta,
        };
      }

      if (delta.content) {
        if (lastType !== "text") {
          lastType = "text";
          lastIndex++;
        }
        yield {
          type: "delta_output_text",
          index: lastIndex,
          delta: delta.content,
        };
      }

      for (const call of delta.tool_calls ?? []) {
        const callIndex = call.index ?? 0;
        let pending = pendingToolUses.get(callIndex);

        if (!pending) {
          if (!call.function?.name || !call.id) {
            continue;
          }

          const tool = normalizedTools.find((tool) =>
            tool.openai.function.name === call.function?.name
          );
          const kind = tool?.original.name ?? call.function.name;
          lastType = "tool_use";
          lastIndex++;
          pending = {
            streamIndex: lastIndex,
            toolUse: {
              type: "tool_use",
              kind,
              tool_use_id: call.id,
              content: call.function.arguments ?? "",
            },
          };
          pendingToolUses.set(callIndex, pending);
          yield {
            type: "tool_use_start",
            index: lastIndex,
            kind,
            tool_use_id: call.id,
          };
        } else if (call.function?.arguments) {
          pending.toolUse.content = (pending.toolUse.content ?? "") +
            call.function.arguments;
        }
      }
    }

    for (
      const { streamIndex, toolUse } of [...pendingToolUses.values()].sort((
        a,
        b,
      ) => a.streamIndex - b.streamIndex)
    ) {
      const tool = normalizedTools.find((tool) =>
        tool.original.name === toolUse.kind
      );

      yield {
        type: "tool_use",
        index: streamIndex,
        tool_use_id: toolUse.tool_use_id,
        kind: toolUse.kind,
        content: tool?.isVoid
          ? undefined
          : getToolUseContent(tool, toolUse.content ?? ""),
      };
    }
  }

  return { name: options.name, stream };
}
