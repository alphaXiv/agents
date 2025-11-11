import type z from "zod";
import { abortable } from "@std/async/abortable";
import { ADAPTERS } from "./adapters.ts";
import type { Tool } from "./tool.ts";
import type {
  AsyncStreamItemGenerator,
  ChatItem,
  ChatItemToolUse,
  ChatLike,
} from "./types.ts";
import {
  convertChatLikeToChatItem,
  convertToolResultLikeToChatItem,
  crossPlatformHandleSigInt,
  crossPlatformLog,
  crossPlatformRemoveHandleSigInt,
  runWithRetries,
} from "./util.ts";
import { addStreamItem } from "./client.ts";
import { signalAsyncLocalStorage } from "./storage.ts";
import { ZodVoid } from "zod";
import { assert } from "@std/assert/assert";
import { ReasoningEffort } from "@alphaxiv/agents";

const MAX_TURNS = 100;
const MAX_PROVIDER_ERRORS = 10;

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
  | "anthropic:claude-3-7-sonnet-latest"
  | "anthropic:claude-sonnet-4-0"
  | "anthropic:claude-sonnet-4-5"
  | "anthropic:claude-3-haiku-20240307"
  | "anthropic:claude-3-5-haiku-latest"
  | "anthropic:claude-haiku-4-5"
  | "anthropic:claude-opus-4-0"
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

export type AgentOptions<zO, zI, M extends ModelString = ModelString> = {
  model: M;
  instructions: string;
  output?: z.ZodType<zO, zI>;
  tools?: M extends NoToolCallModels ? never : Tool<any, any>[];
  reasoningEffort?: ReasoningEffort;
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
  #reasoningEffort: ReasoningEffort;

  constructor(options: AgentOptions<zO, zI, M>) {
    const [provider, ...modelParts] = options.model.split(":");
    this.#provider = provider;
    this.#model = modelParts.join(":");
    this.#instructions = options.instructions;
    this.#output = options.output;
    this.#tools = options.tools ?? [];
    this.#reasoningEffort = options.reasoningEffort ?? "normal";
  }

  async #runToolUses(toolUses: ChatItemToolUse[], signal: AbortSignal) {
    return await Promise.all(
      toolUses.map(async (toolUse) => {
        try {
          const tool = this.#tools.find((tool) => tool.name === toolUse.kind);
          if (!tool) {
            throw new Error(`Tool does not exist: ${toolUse.kind}`);
          }

          try {
            if (!(tool.parameters instanceof ZodVoid)) {
              assert(toolUse.content);
              tool.parameters.parse(JSON.parse(toolUse.content));
            }
          } catch (err) {
            throw new Error(
              `Invalid parameters for tool: ${
                err instanceof Error ? err.message : err
              }`,
            );
          }

          const result = await abortable(
            signalAsyncLocalStorage.run(signal, async () => {
              return await tool.execute({
                param: toolUse.content
                  ? JSON.parse(toolUse.content)
                  : undefined,
              });
            }),
            signal,
          );

          return convertToolResultLikeToChatItem(result, toolUse.tool_use_id);
        } catch (err) {
          if (signal.aborted) {
            throw err;
          }
          return [{
            type: "tool_result_text" as const,
            tool_use_id: toolUse.tool_use_id,
            content: "Error: " +
              (err instanceof Error ? err.message : (err as string).toString()),
          }];
        }
      }),
    );
  }

  /** Run agent without streaming */
  async run(
    chatLike: ChatLike,
    options?: {
      signal: AbortSignal;
    },
  ): Promise<AgentRunResult<zO>> {
    const signal = options?.signal ?? signalAsyncLocalStorage.getStore() ??
      new AbortController().signal;
    const history = convertChatLikeToChatItem(chatLike, "input_text");
    const adapterClass = ADAPTERS[this.#provider];
    if (!adapterClass) throw new Error("Could not resolve provider");
    const adapter = new adapterClass({
      model: this.#model,
      output: this.#output,
      tools: this.#tools,
      reasoningEffort: this.#reasoningEffort,
    });

    const newHistory: ChatItem[] = [];
    for (let turn = 0; turn < MAX_TURNS; turn++) {
      const result = await runWithRetries(() =>
        adapter.run({
          systemPrompt: this.#instructions,
          history: [...history, ...newHistory],
          signal,
        }), 5);

      newHistory.push(...result);

      const toolUses = result.filter((chatItem) =>
        chatItem.type === "tool_use"
      );
      if (toolUses.length > 0) {
        const toolResults = await this.#runToolUses(toolUses, signal);
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
  async *stream(chatLike: ChatLike, options?: {
    signal: AbortSignal;
  }): AsyncStreamItemGenerator {
    const signal = options?.signal ?? new AbortController().signal;
    const initialHistory = convertChatLikeToChatItem(chatLike, "input_text");
    const adapterClass = ADAPTERS[this.#provider];
    if (!adapterClass) throw new Error("Could not resolve provider");
    const adapter = new adapterClass({
      model: this.#model,
      output: this.#output,
      tools: this.#tools,
      reasoningEffort: this.#reasoningEffort,
    });

    let providerErrors = 0;
    const history: ChatItem[] = [];
    for (let turn = 0; turn < MAX_TURNS; turn++) {
      const newHistory: ChatItem[] = [];
      const stream = adapter.stream({
        systemPrompt: this.#instructions,
        history: [...initialHistory, ...history],
        signal,
      });

      try {
        for await (const part of stream) {
          const reIndexedPart = {
            ...part,
            index: part.index + history.length,
          };
          addStreamItem(newHistory, part);
          yield reIndexedPart;
        }
      } catch (err) {
        if (providerErrors < MAX_PROVIDER_ERRORS) {
          providerErrors++;
          // continue loop
          history.push(...newHistory);
          continue;
        }
        throw err;
      }

      const toolUses = newHistory.filter((chatItem) =>
        chatItem.type === "tool_use"
      );
      if (toolUses.length > 0) {
        // execute and stream tools
        const toolResults = await this.#runToolUses(toolUses, signal);
        for (const toolResult of toolResults.flat()) {
          if (toolResult.type === "tool_result_text") {
            yield {
              type: "tool_result_text",
              index: newHistory.length + history.length,
              tool_use_id: toolResult.tool_use_id,
              content: toolResult.content,
            };
            newHistory.push(toolResult);
          } else {
            console.log(toolResult);
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

      const abortController = new AbortController();
      const handler = () => {
        abortController.abort();
      };
      crossPlatformHandleSigInt(handler);

      const newHistory: ChatItem[] = [];
      try {
        const stream = this.stream(history, { signal: abortController.signal });
        for await (const part of stream) {
          if (part.index + 1 > newHistory.length) {
            if (
              newHistory.length > 0 &&
              !(newHistory[newHistory.length - 1].type === "output_text" &&
                part.type === "delta_output_text")
            ) {
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
            } else if (part.type === "tool_result_text") {
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
      } catch (err) {
        if (!abortController.signal.aborted) {
          throw err;
        }
      }
      history.push(...newHistory);
      crossPlatformLog("\x1b[0m\n");

      crossPlatformRemoveHandleSigInt(handler);
    }
  }
}
