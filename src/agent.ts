import type z from "zod";
import { ADAPTERS } from "./adapters.ts";
import type { Tool } from "./tool.ts";
import type { ChatItem, ChatLike } from "./types.ts";
import { convertChatLikeToChatItem, runWithRetries } from "./util.ts";

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

type AgentRunResultOutput<zO, zI> = unknown extends zO ? undefined : zO;
export interface AgentRunResult<zO, zI> {
  history: ChatItem[];
  output: AgentRunResultOutput<zO, zI>;
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
  ): Promise<AgentRunResult<zO, zI>> {
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
          : undefined) as AgentRunResultOutput<zO, zI>,
        outputText: newHistory.filter((history) =>
          history.type === "output_text"
        ).map((history) => history.content).join("\n"),
      };
    }

    throw new Error("MAX TURNS EXCEEDED");
  }

  async cli() {
    const history: ChatItem[] = [];
    while (true) {
      const content = prompt(">");
      if (!content) break;
      history.push({ type: "input_text", content });

      const newResult = await this.run(history);
      for (const item of newResult.history) {
        if (item.type === "tool_use") {
          console.log(
            `[${item.tool_use_id}]`,
            "Calling",
            item.kind,
            "with parameters",
            item.content,
          );
        }
        if (item.type === "tool_result") {
          console.log(`[${item.tool_use_id}]`, "Got tool result", item.content);
        }
        if (item.type === "output_reasoning") {
          console.log(`\x1b[3m${item.content}\x1b[0m`);
        }
        if (item.type === "output_text") {
          console.log(item.content);
        }
      }
      history.push(...newResult.history);
    }
  }
}
