import OpenAI, { type ClientOptions } from "openai";
import type { ChatCompletionCreateParamsStreaming } from "openai/resources/chat/completions";
import type z from "zod";
import type { AdapterStreamIterator } from "../../types.ts";
import type { Adapter, AdapterStreamOptions } from "../adapter.ts";
import { classifyOpenAIError } from "../shared/classify_error.ts";
import { DEFAULT_SUPPORTED_MIME_TYPES, supportsMimeType } from "../shared/media.ts";
import { createOpenAICompatibleSchema } from "../shared/openai_compatibility.ts";
import { restoreWrappedToolArguments } from "../shared/tools.ts";

import { getOpenAICompletionsHistory, type OpenAICompletionsPdfSupport } from "./history.ts";
import { normalizeOpenAICompletionsTools, type OpenAICompletionsToolMap } from "./tools.ts";

interface PendingToolUse {
  streamIndex: number | null;
  callId?: string;
  name?: string;
  content: string;
}

export type OpenAICompletionsClient = Pick<OpenAI, "chat">;

export type { OpenAICompletionsPdfSupport, OpenAICompletionsPdfSupportConfig } from "./history.ts";

export interface OpenAICompletionsExtraRequestBodyArgs<TModel extends string, zO, zI> {
  model: TModel;
  output?: z.ZodType<zO, zI>;
  normalizedTools: OpenAICompletionsToolMap[];
}

export type OpenAICompletionsExtraRequestBody<TModel extends string> =
  | Record<string, unknown>
  | (<zO, zI>(args: OpenAICompletionsExtraRequestBodyArgs<TModel, zO, zI>) => Record<string, unknown>);

export function resolveOpenAICompletionsExtraRequestBody<TModel extends string, zO, zI>(
  extraRequestBody: OpenAICompletionsExtraRequestBody<TModel> | undefined,
  args: OpenAICompletionsExtraRequestBodyArgs<TModel, zO, zI>,
): Record<string, unknown> {
  if (!extraRequestBody) return {};
  return typeof extraRequestBody === "function" ? extraRequestBody(args) : extraRequestBody;
}

/** Generic adapter over an OpenAI Chat Completions compatible API */
export function openAICompletionsModel<zO, zI, TModel extends string>(options: {
  model: TModel;
  supportedMimeTypes?: string[];
  pdfSupport?: OpenAICompletionsPdfSupport<TModel>;
  parallelToolCalls?: boolean;
  provider?: string;
  extraRequestBody?: OpenAICompletionsExtraRequestBody<TModel>;
  openAIOptions?: ClientOptions;
  client?: OpenAICompletionsClient;
}): Adapter<zO, zI> {
  const supportedMimeTypes = options.supportedMimeTypes ?? DEFAULT_SUPPORTED_MIME_TYPES;
  if (options.pdfSupport && !supportsMimeType("application/pdf", supportedMimeTypes)) {
    throw new Error("pdfSupport requires application/pdf to be included in supportedMimeTypes");
  }

  const client = options.client ?? new OpenAI(options.openAIOptions);
  const parallelToolCalls = options.parallelToolCalls ?? true;

  return {
    provider: options.provider ?? "OpenAICompletions",
    model: options.model,
    stream: async function* stream<zO, zI>(
      { history, instructions, tools, signal, output }: AdapterStreamOptions<zO, zI>,
    ): AdapterStreamIterator {
      const normalizedTools = normalizeOpenAICompletionsTools(tools);
      const structuredOutput = output && createOpenAICompatibleSchema(output, {
        kind: "output",
        rootPath: "output",
      });
      const shouldRestoreStructuredOutput = structuredOutput?.requiresValueTransformation ?? false;
      const fullInstructions = structuredOutput?.instructions
        ? `${structuredOutput.instructions}\n\n${instructions}`
        : instructions;
      const messages = await getOpenAICompletionsHistory({
        model: options.model,
        history,
        instructions: fullInstructions,
        normalizedTools,
        signal,
        supportedMimeTypes,
        pdfSupport: options.pdfSupport,
      });

      const extraRequestBody = resolveOpenAICompletionsExtraRequestBody(options.extraRequestBody, {
        model: options.model,
        output,
        normalizedTools,
      });

      const request: ChatCompletionCreateParamsStreaming = {
        model: options.model,
        messages,
        parallel_tool_calls: normalizedTools.length > 0 ? parallelToolCalls : undefined,
        tools: normalizedTools.length > 0 ? normalizedTools.map((tool) => tool.openAI) : undefined,
        response_format: structuredOutput
          ? {
            type: "json_schema",
            json_schema: {
              name: "output",
              strict: true,
              schema: structuredOutput.jsonSchema,
            },
          }
          : { type: "text" },
        stream: true,
        ...extraRequestBody,
        stream_options: {
          ...(extraRequestBody.stream_options ?? {}),
          include_usage: true,
        },
      };

      const response = client.chat.completions.stream(request, { signal });

      const pendingToolUses: PendingToolUse[] = [];
      const startedToolUses: PendingToolUse[] = [];
      const pendingStructuredOutput: string[] = [];

      let lastType = "";
      let lastIndex = -1;

      for await (const part of response) {
        const choice = part.choices?.[0];
        if (!choice?.delta) continue;
        const delta = choice.delta as {
          content?: string | null;
          reasoning?: string | null;
          tool_calls?: Array<{
            index?: number;
            id?: string;
            function?: { name?: string; arguments?: string };
          }>;
        };

        if (delta.reasoning) {
          if (lastType !== "reasoning") {
            lastType = "reasoning";
            lastIndex++;
          }

          yield {
            type: "delta_output_reasoning",
            index: lastIndex,
            delta: delta.reasoning,
          };
        }

        if (delta.content) {
          if (lastType !== "text") {
            lastType = "text";
            lastIndex++;
          }

          if (shouldRestoreStructuredOutput) {
            pendingStructuredOutput[lastIndex] ??= "";
            pendingStructuredOutput[lastIndex] += delta.content;
          } else {
            yield {
              type: "delta_output_text",
              index: lastIndex,
              delta: delta.content,
            };
          }
        }

        for (const call of delta.tool_calls ?? []) {
          const callIndex = call.index ?? 0;
          const pending = pendingToolUses[callIndex] ?? { streamIndex: null, content: "" };

          if (call.id) pending.callId = call.id;
          if (call.function?.name) pending.name = call.function.name;
          if (call.function?.arguments) pending.content += call.function.arguments;

          if (pending.streamIndex === null && pending.callId && pending.name) {
            const tool = normalizedTools.find((candidate) => candidate.openAI.function.name === pending.name);
            pending.streamIndex = ++lastIndex;
            lastType = "tool_use";
            startedToolUses.push(pending);

            yield {
              type: "tool_use_start",
              index: pending.streamIndex,
              tool_use_id: pending.callId,
              kind: tool?.original.name ?? pending.name,
            };
          }

          pendingToolUses[callIndex] = pending;
        }
      }

      for (const pending of startedToolUses) {
        if (pending.streamIndex === null || !pending.callId || !pending.name) continue;
        const tool = normalizedTools.find((candidate) => candidate.openAI.function.name === pending.name);

        yield {
          type: "tool_use",
          index: pending.streamIndex,
          tool_use_id: pending.callId,
          kind: tool?.original.name ?? pending.name,
          content: restoreWrappedToolArguments(pending.content, tool),
        };
      }

      if (shouldRestoreStructuredOutput) {
        for (let index = 0; index < pendingStructuredOutput.length; index++) {
          const rawText = pendingStructuredOutput[index];
          if (rawText === undefined) continue;

          let restoredText = rawText;
          try {
            restoredText = JSON.stringify(structuredOutput!.fromProvider(JSON.parse(rawText)));
          } catch {
            restoredText = rawText;
          }

          yield {
            type: "delta_output_text",
            index,
            delta: restoredText,
          };
        }
      }

      const usage = await response.totalUsage();
      return {
        inputTokens: usage?.prompt_tokens ?? null,
        outputTokens: usage?.completion_tokens ?? null,
      };
    },
    classifyError: classifyOpenAIError,
  };
}
