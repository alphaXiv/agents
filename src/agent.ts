import { ADAPTERS } from "./adapters.ts";
import { Tool } from "./tool.ts";
import type { ChatItem, ChatLike, ZodSchemaType } from "./types.ts";
import { convertChatLikeToChatItem } from "./util.ts";

const MAX_TURNS = 100;

export type ModelString = "__testing:deterministic" | (string & {});

export interface AgentOptions<O> {
  model: ModelString;
  instructions: string;
  output?: ZodSchemaType<O>;
  tools?: Tool<any>[];
}

type AgentRunResultOutput<O> = O extends unknown ? undefined : O;
export interface AgentRunResult<O> {
  history: ChatItem[];
  output: AgentRunResultOutput<O>;
  outputText: string;
}

export class Agent<O> {
  #provider: string;
  #model: ModelString;
  #instructions: string;
  #output?: ZodSchemaType<O>;
  #tools: Tool<any>[];

  constructor(options: AgentOptions<O>) {
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
  ): Promise<AgentRunResult<O>> {
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
      const result = await adapter.run({ // TODO: add retry here
        systemPrompt: this.#instructions,
        history: [...history, ...newHistory],
      });
      newHistory.push(...result);

      const toolUses = result.filter((chatItem) =>
        chatItem.type === "tool_use"
      );
      if (toolUses.length > 0) {
        // TODO: verify that it's using all real tools

        // tool subloop
        const toolResults = await Promise.all(
          toolUses.map(async (toolUse) => {
            const tool = this.#tools.find((tool) => tool.name === toolUse.name);
            if (!tool) throw new Error("wtfrick model tried to use fake tool"); // TODO: handle this better

            const result = await tool.execute({
              param: JSON.parse(toolUse.input),
              toolUseId: toolUse.id,
            });

            return convertChatLikeToChatItem(result, "tool_result", {
              tool_use_id: toolUse.id,
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
          this.#output.parse(JSON.parse(finalItem.text));
        } catch {
          console.log("parsing failed", finalItem.text);
          // TODO: add error here
          continue;
        }
      }

      return {
        history: newHistory,
        output: (this.#output
          ? JSON.parse(finalItem.type === "output_text" ? finalItem.text : "")
          : undefined) as AgentRunResultOutput<O>,
        outputText: newHistory.filter((history) =>
          history.type === "output_text"
        ).map((history) =>
          history.text
        ).join("\n"),
      };
    }

    throw new Error("MAX TURNS EXCEEDED");
  }

  async cli() {
    let history: ChatItem[] = [];
    while (true) {
      let text = prompt(">");
      if (!text) break;
      history.push({ type: "input_text", text });

      const newResult = await this.run(history);
      for (const item of newResult.history) {
        if (item.type === "tool_use") {
          console.log(
            `[${item.id}]`,
            "Calling",
            item.name,
            "with parameters",
            item.input,
          );
        }
        if (item.type === "tool_result") {
          console.log(`[${item.tool_use_id}]`, "Got tool result", item.content);
        }
        if (item.type === "output_text") {
          console.log(item.text);
        }
      }
      history.push(...newResult.history);
    }
  }
}
