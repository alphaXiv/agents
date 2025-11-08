import OpenAI from "openai";
import type {
  ChatCompletionFunctionTool,
  ChatCompletionMessageParam,
} from "openai/resources/chat/completions/completions";
import z from "zod";
import { assert } from "@std/assert";

import type { Tool } from "../tool.ts";
import type {
  AsyncStreamItemGenerator,
  ChatItem,
  ReasoningEffort,
} from "../types.ts";
import { crossPlatformEnv } from "../util.ts";

const supportedImageMimeTypes = [
  "image/jpeg",
  "image/jpg",
  "image/png",
  "image/gif",
  "image/webp",
];

async function getOpenrouterHistory(
  history: ChatItem[],
  systemPrompt: string,
  toolMap: OpenrouterToolMap[],
  signal: AbortSignal,
) {
  const openrouterHistory: ChatCompletionMessageParam[] = [{
    role: "system", // TODO: select right role for each model
    content: systemPrompt,
  }];
  for (const historyItem of history) {
    if (historyItem.type === "input_text") {
      openrouterHistory.push({
        role: "user",
        content: historyItem.content,
      });
    } else if (historyItem.type === "output_text") {
      openrouterHistory.push({
        role: "assistant",
        content: historyItem.content,
      });
    } else if (historyItem.type === "tool_use") {
      const tool = toolMap.find((tool) =>
        tool.original.name === historyItem.kind
      );
      openrouterHistory.push({
        role: "assistant",
        tool_calls: [
          {
            id: historyItem.tool_use_id,
            type: "function",
            function: {
              name: tool?.openrouter.function.name ?? historyItem.kind,
              arguments: (tool?.wrapperObject
                ? `{"content":${historyItem.content}}`
                : historyItem.content) ?? "{}",
            },
          },
        ],
      });
    } else if (historyItem.type === "tool_result_text") {
      openrouterHistory.push({
        role: "tool",
        tool_call_id: historyItem.tool_use_id,
        content: historyItem.content,
      });
    } else if (
      historyItem.type === "input_file" ||
      historyItem.type === "tool_result_file"
    ) {
      if (supportedImageMimeTypes.includes(historyItem.kind)) {
        openrouterHistory.push({
          role: "user",
          content: [{
            type: "image_url",
            image_url: { url: historyItem.content },
          }],
        });
      } else if (historyItem.kind === "application/pdf") {
        const req = await fetch(historyItem.content, {
          method: "HEAD",
        });
        const fileSize = parseInt(
          req.headers.get("Content-Length") ?? "9999999999999",
        ); // If the service doesn't currently return content length, assume it's too big
        const MAX_MEGABYTES = 4;
        if (fileSize > MAX_MEGABYTES * 1024 * 1024) { // Openrouter seems to have an undocumented 5MB size limit on pdfs :) - 4 to be safe here
          const req = await fetch(historyItem.content);
          const { default: parsePdf } = await import("@lino/pdf-parse");
          const pdfText = await parsePdf(await req.arrayBuffer());
          openrouterHistory.push({
            role: "user",
            content: [{
              type: "text",
              text: pdfText.text.join("\n"),
            }],
          });
        } else {
          openrouterHistory.push({
            role: "user",
            content: [{
              type: "file",
              file: {
                file_data: historyItem.content,
              },
            }],
          });
        }
      } else if (historyItem.kind.startsWith("text/")) {
        const req = await fetch(historyItem.content, { signal });
        const text = await req.text();

        openrouterHistory.push({
          role: "user",
          content: [{
            type: "text",
            text: `<file>${text}</file>`,
          }],
        });
      } else {
        throw new Error(
          "OpenRouter models don't support the following media type: " +
            historyItem.kind,
        );
      }
    } else if (historyItem.type === "output_reasoning") {
      // no-op, don't propagate reasoning
    } else {
      historyItem satisfies never;
    }
  }

  return openrouterHistory;
}

// TODO: keep this updated (pulled from https://openrouter.ai/models?fmt=cards&input_modalities=file)
const nativePdfSupport = [
  "openai/gpt-5-image-mini",
  "openai/gpt-5-image",
  "openai/o3-deep-research",
  "openai/o4-mini-deep-research",
  "openai/gpt-5-pro",
  "anthropic/claude-sonnet-4.5",
  "google/gemini-2.5-flash-preview-09-2025",
  "google/gemini-2.5-flash-lite-preview-09-2025",
  "openai/gpt-5-chat",
  "openai/gpt-5",
  "openai/gpt-5-mini",
  "openai/gpt-5-nano",
  "anthropic/claude-opus-4.1",
  "google/gemini-2.5-flash-lite",
  "google/gemini-2.5-flash-lite-preview-06-17",
  "google/gemini-2.5-flash",
  "google/gemini-2.5-pro",
  "openai/o3-pro",
  "google/gemini-2.5-pro-preview",
  "anthropic/claude-opus-4",
  "anthropic/claude-sonnet-4",
  "google/gemini-2.5-pro-preview-05-06",
  "openai/o4-mini-high",
  "openai/o3",
  "openai/o4-mini",
  "openai/gpt-4.1",
  "openai/gpt-4.1-mini",
  "openai/gpt-4.1-nano",
  "openai/o1-pro",
  "google/gemini-2.0-flash-lite-001",
  "anthropic/claude-3.7-sonnet",
  "anthropic/claude-3.7-sonnet",
  "openai/o3-mini-high",
  "google/gemini-2.0-flash-001",
  "openai/o3-mini",
  "openai/o1",
  "openai/gpt-4o-2024-11-20",
  "anthropic/claude-3.5-haiku-20241022",
  "anthropic/claude-3.5-sonnet",
  "openai/gpt-4o-2024-08-06",
  "openai/gpt-4o-mini",
  "openai/gpt-4o-mini-2024-07-18",
  "anthropic/claude-3.5-sonnet-20240620",
  "openai/gpt-4o",
  "openai/gpt-4o",
  "openai/gpt-4o-2024-05-13",
  "google/gemini-2.5-flash-preview-05-20",
  "google/gemini-2.5-flash-preview",
  "google/gemini-2.5-pro-exp-03-25",
];

// TODO: ensure this list is complete
const alwaysReasoningModels = [
  "openai/gpt-oss-20b",
  "openai/gpt-oss-120b",
];

type OpenrouterToolMap = {
  original: Tool<unknown, unknown>;
  openrouter: ChatCompletionFunctionTool;
  /** Openrouter doesn't allow non-objects at the top level but we want to. We therefore wrap the tool input with a wrapper object which need to unwrap at the output */
  wrapperObject: boolean;
  /** No parameter specified */
  isVoid: boolean;
};

export class OpenRouterAdapter<zO, zI> {
  #client: OpenAI;
  #model: string;
  #output?: z.ZodType<zO, zI>;
  #normalizedTools: OpenrouterToolMap[];
  #reasoningEffort: ReasoningEffort;

  constructor(
    { model, output, tools, reasoningEffort }: {
      model: string;
      output?: z.ZodType<zO, zI>;
      tools: Tool<unknown, unknown>[];
      reasoningEffort: ReasoningEffort;
    },
  ) {
    this.#model = model;
    this.#output = output;
    this.#reasoningEffort = reasoningEffort;
    this.#normalizedTools = tools.map((tool) => {
      // TODO: improve this mapping
      const name = tool.name.toLowerCase().replaceAll(" ", "_").replace(
        /[^a-zA-Z0-9_-]/g,
        "",
      );

      const isVoid = tool.parameters instanceof z.ZodVoid;
      const wrapperObject = !isVoid &&
        !(tool.parameters instanceof z.ZodObject);

      return {
        original: tool,
        openrouter: {
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
    this.#client = new OpenAI({
      baseURL: "https://openrouter.ai/api/v1",
      apiKey: crossPlatformEnv("OPENROUTER_API_KEY"),
    });
  }

  async run({ history, systemPrompt, signal }: {
    systemPrompt: string;
    history: ChatItem[];
    signal: AbortSignal;
  }): Promise<ChatItem[]> {
    const openrouterHistory = await getOpenrouterHistory(
      history,
      systemPrompt,
      this.#normalizedTools,
      signal,
    );

    const response = await this.#client.chat.completions.create({
      model: this.#model,
      messages: openrouterHistory,
      tools: this.#normalizedTools.map(({ openrouter }) => openrouter),
      response_format: this.#output
        ? {
          type: "json_schema",
          json_schema: z.toJSONSchema(this.#output) as any,
        }
        : { type: "text" },
      // @ts-expect-error openrouter isn't type safe :(
      reasoning: {
        enabled: this.#reasoningEffort === "normal" ||
          alwaysReasoningModels.includes(this.#model),
      },
      plugins: nativePdfSupport.includes(this.#model) ? undefined : [
        {
          id: "file-parser",
          pdf: {
            engine: "pdf-text",
          },
        },
      ],
    }, { signal });

    const output: ChatItem[] = [];

    const choice = response.choices[0];

    if (choice.message.content) {
      const reasoning =
        (choice.message as unknown as { reasoning: string | undefined })
          .reasoning;
      if (reasoning) {
        output.push({
          type: "output_reasoning",
          content: reasoning,
        });
      }
      output.push({
        type: "output_text",
        content: choice.message.content,
      });
    }
    for (const toolUse of choice.message.tool_calls ?? []) {
      assert(toolUse.type === "function");
      const tool = this.#normalizedTools.find((tool) =>
        tool.openrouter.function.name === toolUse.function.name
      );
      const content = JSON.parse(toolUse.function.arguments);
      output.push({
        type: "tool_use",
        tool_use_id: toolUse.id,
        kind: tool?.original.name ?? toolUse.function.name,
        content: tool?.isVoid
          ? undefined
          : (tool?.wrapperObject
            ? JSON.stringify(content.content)
            : toolUse.function.arguments),
      });
    }

    return output;
  }

  async *stream({ history, systemPrompt, signal }: {
    systemPrompt: string;
    history: ChatItem[];
    signal: AbortSignal;
  }): AsyncStreamItemGenerator {
    const openrouterHistory = await getOpenrouterHistory(
      history,
      systemPrompt,
      this.#normalizedTools,
      signal,
    );

    const response = this.#client.chat.completions.stream({
      model: this.#model,
      messages: openrouterHistory,
      tools: this.#normalizedTools.map(({ openrouter }) => openrouter),
      // openrouter-specific extensions
      reasoning: {
        enabled: this.#reasoningEffort === "normal" ||
          alwaysReasoningModels.includes(this.#model),
      },
      plugins: nativePdfSupport.includes(this.#model) ? undefined : [
        {
          id: "file-parser",
          pdf: {
            engine: "pdf-text",
          },
        },
      ],
    }, { signal });

    const toolMap: ChatItem[] = [];

    const deltas = [];

    let lastType = "";
    let lastIndex = -1;
    for await (const part of response) {
      const choice = part.choices[0];
      if (!choice) continue; // Skip empty choices
      const { delta } = choice;
      deltas.push(delta);

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
        const callFunction = call.function;
        if (callFunction?.name) {
          lastType = "tool_use";
          lastIndex++;

          const tool = this.#normalizedTools.find((tool) =>
            tool.openrouter.function.name === callFunction.name
          );
          assert(call.id);

          toolMap[lastIndex] = {
            type: "tool_use",
            kind: tool?.original.name ?? callFunction.name,
            tool_use_id: call.id,
            content: callFunction.arguments ?? "",
          };
        } else if (callFunction?.arguments) {
          const toolUse = toolMap[lastIndex];
          assert(toolUse.type === "tool_use");
          toolUse.content += callFunction.arguments;
        }
      }
    }
    for (const msg of toolMap) {
      if (msg?.type === "tool_use") {
        const toolUse = msg;
        assert(toolUse.type === "tool_use");
        const tool = this.#normalizedTools.find((tool) =>
          tool.original.name === toolUse.kind
        );

        try {
          const parsedContent = toolUse.content
            ? JSON.parse(toolUse.content)
            : undefined;
          yield {
            type: "tool_use",
            index: lastIndex,
            tool_use_id: toolUse.tool_use_id,
            kind: toolUse.kind,
            content: tool?.isVoid
              ? undefined
              : (tool?.wrapperObject
                ? JSON.stringify(parsedContent.content)
                : toolUse.content),
          };
        } catch {
          // the function call isn't done yet
        }
      }
    }
    // console.log(deltas);
  }
}
