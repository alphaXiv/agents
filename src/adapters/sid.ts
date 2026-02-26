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

function getSidHistory(
  history: ChatItem[],
  systemPrompt: string,
  toolMap: SidToolMap[],
) {
  const sidHistory: ChatCompletionMessageParam[] = [{
    role: "system",
    content: systemPrompt,
  }];
  for (const historyItem of history) {
    if (!historyItem) continue;
    if (historyItem.type === "input_text") {
      sidHistory.push({
        role: "user",
        content: historyItem.content,
      });
    } else if (historyItem.type === "output_text") {
      sidHistory.push({
        role: "assistant",
        content: historyItem.content,
      });
    } else if (historyItem.type === "tool_use") {
      const tool = toolMap.find((tool) =>
        tool.original.name === historyItem.kind
      );
      sidHistory.push({
        role: "assistant",
        tool_calls: [
          {
            id: historyItem.tool_use_id,
            type: "function",
            function: {
              name: tool?.sid.function.name ?? historyItem.kind,
              arguments: (tool?.wrapperObject
                ? `{"content":${historyItem.content}}`
                : historyItem.content) ?? "{}",
            },
          },
        ],
      });
    } else if (historyItem.type === "tool_result_text") {
      sidHistory.push({
        role: "tool",
        tool_call_id: historyItem.tool_use_id,
        content: historyItem.content,
      });
    } else if (
      historyItem.type === "input_file" ||
      historyItem.type === "tool_result_file"
    ) {
      throw new Error(
        "SID models don't support the following media type: " +
          historyItem.kind,
      );
    }
  }

  if (sidHistory.length > 0) {
    const lastMessage = sidHistory[sidHistory.length - 1];
    if (lastMessage.role === "assistant") {
      sidHistory.push({
        role: "developer",
        content: [{
          type: "text",
          text: RETRY_RESUMABILITY_PROMPT,
        }],
      });
    }
  }

  return sidHistory;
}

type SidToolMap = {
  original: Tool<unknown, unknown>;
  sid: ChatCompletionFunctionTool;
  wrapperObject: boolean;
  isVoid: boolean;
};

export class SidAdapter<zO, zI> {
  #client: OpenAI;
  #model: string;
  #output?: z.ZodType<zO, zI>;
  #normalizedTools: SidToolMap[];

  constructor(
    { model, output, tools }: {
      model: string;
      output?: z.ZodType<zO, zI>;
      tools: Tool<unknown, unknown>[];
      reasoningEffort: ReasoningEffort;
    },
  ) {
    this.#model = model;
    this.#output = output;
    this.#normalizedTools = tools.map((tool) => {
      const name = tool.name.toLowerCase().replaceAll(" ", "_").replace(
        /[^a-zA-Z0-9_-]/g,
        "",
      );

      const isVoid = tool.parameters instanceof z.ZodVoid;
      const wrapperObject = !isVoid &&
        !(tool.parameters instanceof z.ZodObject);

      return {
        original: tool,
        sid: {
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
      baseURL: "https://api.sid-1.com/v1",
      apiKey: crossPlatformEnv("SID_API_KEY"),
    });
  }

  async run({ history, systemPrompt, signal }: {
    systemPrompt: string;
    history: ChatItem[];
    signal: AbortSignal;
  }): Promise<ChatItem[]> {
    const sidHistory = getSidHistory(
      history,
      systemPrompt,
      this.#normalizedTools,
    );

    const response = await this.#client.chat.completions.create({
      model: this.#model,
      messages: sidHistory,
      tools: this.#normalizedTools.map(({ sid }) => sid),
      response_format: this.#output
        ? {
          type: "json_schema",
          // deno-lint-ignore no-explicit-any
          json_schema: z.toJSONSchema(this.#output) as any,
        }
        : { type: "text" },
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
        tool.sid.function.name === toolUse.function.name
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
    const sidHistory = getSidHistory(
      history,
      systemPrompt,
      this.#normalizedTools,
    );

    const response = this.#client.chat.completions.stream({
      model: this.#model,
      messages: sidHistory,
      tools: this.#normalizedTools.map(({ sid }) => sid),
    }, { signal });

    const toolMap: ChatItem[] = [];

    let lastType = "";
    let lastIndex = -1;
    for await (const part of response) {
      const choice = part.choices[0];
      if (!choice) continue;
      const { delta } = choice;

      // TODO: rewrite this because they don't parse reasoning, we would have to parse the reasonign ourselves (which we should do!)
      const reasoningDelta =
        (delta as unknown as { reasoning: string | undefined }).reasoning;

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
            tool.sid.function.name === callFunction.name
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
  }
}
