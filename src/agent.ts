import type z from "zod";
import { ADAPTERS } from "./adapters.ts";
import type { Tool } from "./tool.ts";
import type { ChatItem, ChatLike, StreamItem } from "./types.ts";
import {
  convertChatLikeToChatItem,
  crossPlatformLog,
  runWithRetries,
} from "./util.ts";
import { addStreamItem } from "./client.ts";

const MAX_TURNS = 100;

export type ModelString =
  | "__testing:deterministic"
  | "openai:gpt-5-pro"
  | "openai:gpt-5"
  | "openai:gpt-5-mini"
  | "openai:gpt-5-nano"
  | "openai:gpt-4.1"
  | "google:gemini-2.5-pro"
  | "google:gemini-2.5-flash"
  | "google:gemini-2.5-flash-image"
  | "google:gemini-2.5-flash-lite"
  | "google:gemini-2.0-flash"
  | "google:gemini-2.0-flash-lite"
  | "anthropic:claude-sonnet-4-5"
  | "anthropic:claude-haiku-4-5"
  | "anthropic:claude-opus-4-1"
  | "openrouter:openai/gpt-oss-20b"
  | "openrouter:openai/gpt-oss-120b"
  | "openrouter:qwen/qwen3-235b-a22b-thinking-2507"
  | "openrouter:qwen/qwen3-235b-a22b-2507"
  | "openrouter:qwen/qwen3-next-80b-a3b-instruct"
  | "openrouter:qwen/qwen3-next-80b-a3b-thinking"
  | "openrouter:x-ai/grok-4-fast"
  | "openrouter:x-ai/grok-4"
  | "openrouter:x-ai/grok-code-fast-1"
  | (string & {});

export type NoToolCallModels = "google:gemini-2.5-flash-image";

export type AgentOptions<zO, zI, M extends ModelString = ModelString> =
  M extends NoToolCallModels ? {
      model: M;
      instructions: string;
      output?: z.ZodType<zO, zI>;
      tools?: never;
    }
    : {
      model: M;
      instructions: string;
      output?: z.ZodType<zO, zI>;
      tools?: Tool<any, any>[];
    };

type AgentRunResultOutput<zO> = unknown extends zO ? undefined : zO;
export interface AgentRunResult<zO> {
  history: ChatItem[];
  output: AgentRunResultOutput<zO>;
  outputText: string;
}

export class Agent<zO, zI, M extends ModelString> {
  #provider: string;
  #model: ModelString;
  #instructions: string;
  #output?: z.ZodType<zO, zI>;
  #tools: Tool<any, any>[];

  constructor(options: AgentOptions<zO, zI, M>) {
    const [provider, ...modelParts] = options.model.split(":");
    this.#provider = provider;
    this.#model = modelParts.join(":");
    this.#instructions = options.instructions;
    this.#output = options.output;
    this.#tools = options.tools ?? [];
  }

  /** Run agent without streaming */
  async run(
    chatLike: ChatLike,
  ): Promise<AgentRunResult<zO>> {
    const history = convertChatLikeToChatItem(chatLike, "input_text");
    const adapterClass = ADAPTERS[this.#provider];
    if (!adapterClass) throw new Error("Could not resolve provider");
    const adapter = new adapterClass({
      model: this.#model,
      output: this.#output,
      tools: this.#tools,
    });

    const newHistory: ChatItem[] = [];
    for (let turn = 0; turn < MAX_TURNS; turn++) {
      const result = await runWithRetries(() =>
        adapter.run({
          systemPrompt: this.#instructions,
          history: [...history, ...newHistory],
        }), 5);

      newHistory.push(...result);

      const toolUses = result.filter((chatItem) =>
        chatItem.type === "tool_use"
      );
      if (toolUses.length > 0) {
        // TODO: verify that it's using all real tools

        // tool subloop
        const toolResults = await Promise.all(
          toolUses.map(async (toolUse) => {
            const tool = this.#tools.find((tool) => tool.name === toolUse.kind);
            if (!tool) throw new Error("wtfrick model tried to use fake tool"); // TODO: handle this better

            const result = await tool.execute({
              param: JSON.parse(toolUse.content),
              toolUseId: toolUse.tool_use_id,
            });

            return convertChatLikeToChatItem(result, "tool_result", {
              tool_use_id: toolUse.tool_use_id,
            });
          }),
        );
        newHistory.push(...toolResults.flat());
        continue;
      }

      const finalItem = newHistory[newHistory.length - 1];
      if (this.#output) {
        if (finalItem.type !== "output_text") {
          // TODO: add error
          continue;
        }

        try {
          this.#output.parse(JSON.parse(finalItem.content));
        } catch (err) {
          console.log("parsing failed", finalItem.content);
          const errStr = err instanceof Error
            ? err.message
            : (err as string).toString();
          newHistory.push({
            type: "output_text",
            content: "Sorry, my output has an error: " + errStr +
              "\n I will try again.",
          });
          continue;
        }
      }

      return {
        history: newHistory,
        output: (this.#output
          ? JSON.parse(
            finalItem.type === "output_text" ? finalItem.content : "",
          )
          : undefined) as AgentRunResultOutput<zO>,
        outputText: newHistory.filter((history) =>
          history.type === "output_text"
        ).map((history) => history.content).join("\n"),
      };
    }

    throw new Error("MAX TURNS EXCEEDED");
  }

  /** Run agent with streaming */
  async *stream(chatLike: ChatLike): AsyncGenerator<StreamItem, void, unknown> {
    const initialHistory = convertChatLikeToChatItem(chatLike, "input_text");
    const adapterClass = ADAPTERS[this.#provider];
    if (!adapterClass) throw new Error("Could not resolve provider");
    const adapter = new adapterClass({
      model: this.#model,
      output: this.#output,
      tools: this.#tools,
    });

    const history: ChatItem[] = [];
    for (let turn = 0; turn < MAX_TURNS; turn++) {
      const stream = adapter.stream({
        systemPrompt: this.#instructions,
        history: [...initialHistory, ...history],
      });
      const newHistory: ChatItem[] = [];
      for await (const part of stream) {
        const reIndexedPart = {
          ...part,
          index: part.index + history.length,
        };
        addStreamItem(newHistory, part);
        yield reIndexedPart;
      }

      const toolUses = newHistory.filter((chatItem) =>
        chatItem.type === "tool_use"
      );
      if (toolUses.length > 0) {
        // execute and stream tools
        const toolResults = await Promise.all(
          toolUses.map(async (toolUse) => {
            const tool = this.#tools.find((tool) => tool.name === toolUse.kind);
            if (!tool) throw new Error("wtfrick model tried to use fake tool"); // TODO: handle this better

            const result = await tool.execute({
              param: JSON.parse(toolUse.content),
              toolUseId: toolUse.tool_use_id,
            });

            return convertChatLikeToChatItem(result, "tool_result", {
              tool_use_id: toolUse.tool_use_id,
            });
          }),
        );
        for (const toolResult of toolResults.flat()) {
          if (toolResult.type === "tool_result") {
            yield {
              type: "tool_result",
              index: newHistory.length,
              tool_use_id: toolResult.tool_use_id,
              content: toolResult.content,
            };
            newHistory.push(toolResult);
          }
        }

        // continue loop
        history.push(...newHistory);
        continue;
      }
      break;
    }
  }

  async cli() {
    const history: ChatItem[] = [];
    while (true) {
      const content = prompt(">");
      if (!content) break;
      history.push({ type: "input_text", content });

      const stream = this.stream(history);
      const newHistory: ChatItem[] = [];
      for await (const part of stream) {
        if (part.index + 1 > newHistory.length) {
          if (newHistory.length > 0) {
            crossPlatformLog("\n");
          }
          if (part.type === "delta_output_text") {
            crossPlatformLog("\x1b[0m");
          } else if (part.type === "delta_output_reasoning") {
            crossPlatformLog("\x1b[3m");
          } else if (part.type === "tool_use") {
            crossPlatformLog(
              `[${part.tool_use_id}] Calling '${part.kind}' with parameters '${part.content}'`,
            );
          } else if (part.type === "tool_result") {
            crossPlatformLog(
              `[${part.tool_use_id}] Got result '${part.content}'`,
            );
          }
        }

        if (part.type === "delta_output_text") {
          crossPlatformLog(part.delta);
        } else if (part.type === "delta_output_reasoning") {
          crossPlatformLog(part.delta);
        }
        addStreamItem(newHistory, part);
      }
      crossPlatformLog("\n");

      history.push(...newHistory);
    }
  }
}
