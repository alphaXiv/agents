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
import { RETRY_RESUMABILITY_PROMPT } from "../constants.ts";

const supportedImageMimeTypes = [
  "image/jpeg",
  "image/jpg",
  "image/png",
  "image/gif",
  "image/webp",
];

async function getTributaryHistory(
  history: ChatItem[],
  systemPrompt: string,
  toolMap: TributaryToolMap[],
  signal: AbortSignal,
) {
  const tributaryHistory: ChatCompletionMessageParam[] = [{
    role: "system", // TODO: select right role for each model
    content: systemPrompt,
  }];
  for (const historyItem of history) {
    if (historyItem.type === "input_text") {
      tributaryHistory.push({
        role: "user",
        content: historyItem.content,
      });
    } else if (historyItem.type === "output_text") {
      tributaryHistory.push({
        role: "assistant",
        content: historyItem.content,
      });
    } else if (historyItem.type === "tool_use") {
      const tool = toolMap.find((tool) =>
        tool.original.name === historyItem.kind
      );
      tributaryHistory.push({
        role: "assistant",
        tool_calls: [
          {
            id: historyItem.tool_use_id,
            type: "function",
            function: {
              name: tool?.tributary.function.name ?? historyItem.kind,
              arguments: (tool?.wrapperObject
                ? `{"content":${historyItem.content}}`
                : historyItem.content) ?? "{}",
            },
          },
        ],
      });
    } else if (historyItem.type === "tool_result_text") {
      tributaryHistory.push({
        role: "tool",
        tool_call_id: historyItem.tool_use_id,
        content: historyItem.content,
      });
    } else if (
      historyItem.type === "input_file" ||
      historyItem.type === "tool_result_file"
    ) {
      if (supportedImageMimeTypes.includes(historyItem.kind)) {
        tributaryHistory.push({
          role: "user",
          content: [{
            type: "image_url",
            image_url: { url: historyItem.content },
          }],
        });
      } else if (historyItem.kind === "application/pdf") {
        const req = await fetch(historyItem.content);
        const { default: parsePdf } = await import("@lino/pdf-parse");
        const pdfText = await parsePdf(await req.arrayBuffer());
        tributaryHistory.push({
          role: "user",
          content: [{
            type: "text",
            text: pdfText.text.join("\n"),
          }],
        });
      } else if (historyItem.kind.startsWith("text/")) {
        const req = await fetch(historyItem.content, { signal });
        const text = await req.text();

        tributaryHistory.push({
          role: "user",
          content: [{
            type: "text",
            text: `<file>${text}</file>`,
          }],
        });
      } else {
        throw new Error(
          "Tributary models don't support the following media type: " +
            historyItem.kind,
        );
      }
    } else if (historyItem.type === "output_reasoning") {
      // no-op, don't propagate reasoning
    } else {
      historyItem satisfies never;
    }
  }

  // Check if this is a resumability request
  if (tributaryHistory.length > 0) {
    const lastMessage = tributaryHistory[tributaryHistory.length - 1];
    if (lastMessage.role === "assistant") {
      tributaryHistory.push({
        role: "developer",
        content: [{
          type: "text",
          text: RETRY_RESUMABILITY_PROMPT,
        }],
      });
    }
  }

  return tributaryHistory;
}

type TributaryToolMap = {
  original: Tool<unknown, unknown, unknown>;
  tributary: ChatCompletionFunctionTool;
  /** Tributary doesn't allow non-objects at the top level but we want to. We therefore wrap the tool input with a wrapper object which need to unwrap at the output */
  wrapperObject: boolean;
  /** No parameter specified */
  isVoid: boolean;
};

export class TributaryAdapter<zO, zI> {
  #client: OpenAI;
  #model: string;
  #output?: z.ZodType<zO, zI>;
  #normalizedTools: TributaryToolMap[];
  #reasoningEffort: ReasoningEffort;

  constructor(
    { model, output, tools, reasoningEffort }: {
      model: string;
      output?: z.ZodType<zO, zI>;
      tools: Tool<unknown, unknown, unknown>[];
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
        tributary: {
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
      baseURL: "https://api.tributary.cc/openai/v1",
      apiKey: crossPlatformEnv("TRIBUTARY_API_KEY"),
    });
  }

  async run({ history, systemPrompt, signal }: {
    systemPrompt: string;
    history: ChatItem[];
    signal: AbortSignal;
  }): Promise<ChatItem[]> {
    const tributaryHistory = await getTributaryHistory(
      history,
      systemPrompt,
      this.#normalizedTools,
      signal,
    );

    const response = await this.#client.chat.completions.create({
      model: this.#model,
      messages: tributaryHistory,
      tools: this.#normalizedTools.map(({ tributary }) => tributary),
      response_format: this.#output
        ? {
          type: "json_schema",
          // deno-lint-ignore no-explicit-any
          json_schema: z.toJSONSchema(this.#output) as any,
        }
        : { type: "text" },
      // reasoning: alwaysReasoningModels.includes(this.#model) ? undefined : {
      //   enabled: this.#reasoningEffort === "normal",
      // },
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
        tool.tributary.function.name === toolUse.function.name
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
    const tributaryHistory = await getTributaryHistory(
      history,
      systemPrompt,
      this.#normalizedTools,
      signal,
    );

    const response = this.#client.chat.completions.stream({
      model: this.#model,
      messages: tributaryHistory,
      tools: this.#normalizedTools.map(({ tributary }) => tributary),
      // reasoning: alwaysReasoningModels.includes(this.#model) ? undefined : {
      //   enabled: this.#reasoningEffort === "normal",
      // },
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

      // @ts-expect-error Handle reasoning content, this is a tributary-specific extension
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
            tool.tributary.function.name === callFunction.name
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
