import type { ReasoningEffort } from "@alphaxiv/agents";
import { assert } from "@std/assert/assert";
import { abortable } from "@std/async/abortable";
import type z from "zod";
import { ZodVoid } from "zod";
import { type Adapter, ADAPTERS } from "./adapters.ts";
import { addStreamItem } from "./client.ts";
import { DEBUG_MODE } from "./constants.ts";
import { signalAsyncLocalStorage } from "./storage.ts";
import { ModelOutput, type Tool } from "./tool.ts";
import type {
  AgentStreamIterator,
  ChatItem,
  ChatItemToolResult,
  ChatItemToolUse,
  ChatLike,
} from "./types.ts";
import {
  convertChatLikeToChatItem,
  convertToolResultLikeToChatItem,
  crossPlatformHandleSigInt,
  crossPlatformLog,
  crossPlatformRemoveHandleSigInt,
  crossPlatformStdin,
  errMessage,
  iteratePromiseArray,
} from "./util.ts";

const MAX_TURNS = 100;
const MAX_PROVIDER_ERRORS = 5;

export type ModelString =
  | "openai:gpt-5.4"
  | "openai:gpt-5.2"
  | "openai:gpt-5.1"
  | "openai:gpt-5-pro"
  | "openai:gpt-5"
  | "openai:gpt-5-mini"
  | "openai:gpt-5-nano"
  | "openai:gpt-4.1"
  | "google:gemini-3-flash-preview"
  | "google:gemini-3-pro-image-preview"
  | "google:gemini-3-pro-preview"
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
  | "anthropic:claude-opus-4-5"
  | "openrouter:openai/gpt-oss-20b"
  | "openrouter:openai/gpt-oss-120b"
  | "openrouter:qwen/qwen3-235b-a22b-thinking-2507"
  | "openrouter:qwen/qwen3-235b-a22b-2507"
  | "openrouter:qwen/qwen3-next-80b-a3b-instruct"
  | "openrouter:qwen/qwen3-next-80b-a3b-thinking"
  | "openrouter:x-ai/grok-4-fast"
  | "openrouter:x-ai/grok-4"
  | "openrouter:x-ai/grok-code-fast-1"
  | "sid:sid-1"
  // deno-lint-ignore ban-types
  | (string & {});

export type NoToolCallModels = "google:gemini-2.5-flash-image";

// deno-fmt-ignore
// deno-lint-ignore no-explicit-any
type ExtractModelOutput<Tools extends readonly Tool<any, any, any>[]> = Tools[number] extends Tool<any, any, infer MO> ? MO : never;

// deno-lint-ignore no-explicit-any
type ResolveAgentOutput<zO, Tools extends readonly Tool<any, any, any>[]> =
  // If the model has structured output
  unknown extends zO ? (
      // It doesn't, extract out the model output from tools and return that + undefined
      [ExtractModelOutput<Tools>] extends [never] ? undefined
        : ExtractModelOutput<Tools> | undefined
    )
    : (
      // It does, extract out the model output from tools and return that + the structured output
      [ExtractModelOutput<Tools>] extends [never] ? zO
        : zO | ExtractModelOutput<Tools>
    );

export type AgentOptions<
  zO,
  zI,
  M extends ModelString = ModelString,
  // deno-lint-ignore no-explicit-any
  Tools extends readonly Tool<any, any, any>[] = Tool<any, any, never>[],
  A extends Adapter<M> = Adapter<M>,
> = {
  /**
   * By default, the adapter is derived from the model string, and you don't
   * need to worry about setting one yourself. But if you need to hit a custom
   * API, that's what this is for. For example, you can call a custom
   * OpenAI-compatible endpoint with `adapter: openAiAdapter({ url: '...' })`.
   */
  adapter?: A;
  /**
   * By default, the `ModelString` is parsed for an adapter prefix like
   * `openai:` or `anthropic:`, which will call that adapter with the given
   * model. Any string value is allowed, even if it isn't listed in the enum,
   * but it is up to the provider to actually support it.
   *
   * When passing a custom adapter, this type depends on the adapter's type.
   */
  model: M;
  /** What this is agent intended to do. Equivilent to a "system prompt". */
  instructions: string;
  output?: z.ZodType<zO, zI>;
  tools?: M extends NoToolCallModels ? never : [...Tools];
  reasoningEffort?: ReasoningEffort;
  /**
   * APIs which are not finalized and are subject to change.
   */
  unstable?: UnstableAgentOptions;
};

export interface UnstableAgentOptions {
  /**
   * Set to `false` to disable retrying.
   * @default true
   */
  retries?: boolean;
}

export interface AgentRunResult<zO> {
  history: ChatItem[];
  output: zO;
  outputText: string;
}

export class Agent<
  zO,
  zI,
  M extends ModelString,
  // deno-lint-ignore no-explicit-any
  const Tools extends readonly Tool<any, any, any>[] = Tool<any, any, never>[],
  A extends Adapter<M> = Adapter<M>,
> {
  #adapter: A | Promise<A>;
  #model: M;
  #instructions: string;
  #output?: z.ZodType<zO, zI>;
  // deno-lint-ignore no-explicit-any
  #tools: Tool<any, any, any>[];
  #reasoningEffort: ReasoningEffort;

  #noRetries = false;

  constructor(options: AgentOptions<zO, zI, M, Tools, A>) {
    if (options.adapter) {
      this.#adapter = options.adapter;
      this.#model = options.model;
    } else {
      const [provider, ...modelParts] = options.model.split(":");
      const factory = ADAPTERS[provider];

      if (!factory) {
        throw new Error(
          "Could not resolve provider " + JSON.stringify(provider),
        );
      }
      this.#adapter = factory() as A | Promise<A>;
      this.#model = modelParts.join(":") as M;
    }

    this.#instructions = options.instructions;
    this.#output = options.output;
    this.#tools = options.tools?.slice() ?? [];
    this.#reasoningEffort = options.reasoningEffort ?? "normal";

    if (options.unstable) {
      const { retries = true, ...unknown } = options.unstable;
      const unknownKeys = Object.keys(unknown);
      if (unknownKeys.length > 0) {
        throw new Error(
          `Unknown unstable options passed: ${
            unknownKeys.join(", ")
          }. These options may have been removed in an update.`,
        );
      }
      this.#noRetries = !retries;
    }
  }

  async #runTool(
    use: ChatItemToolUse,
    signal: AbortSignal,
  ): Promise<{ id: string; items: ChatItemToolResult[] }> {
    try {
      const tool = this.#tools.find((tool) => tool.name === use.kind);
      if (!tool) {
        throw new Error(`Tool does not exist: ${use.kind}`);
      }

      try {
        if (!(tool.parameters instanceof ZodVoid)) {
          assert(use.content);
          tool.parameters.parse(JSON.parse(use.content));
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
            param: use.content ? JSON.parse(use.content) : undefined,
            signal,
          });
        }),
        signal,
      );

      if (result instanceof ModelOutput) throw result;

      return {
        id: use.tool_use_id,
        items: convertToolResultLikeToChatItem(result, use.tool_use_id),
      };
    } catch (err) {
      if (err instanceof ModelOutput || signal.aborted) throw err;
      return {
        id: use.tool_use_id,
        items: [{
          type: "tool_result_text" as const,
          tool_use_id: use.tool_use_id,
          content: "Error: " +
            (err instanceof Error ? err.message : (err as string).toString()),
        }],
      };
    }
  }

  /**
   * Run the agent without streaming. Internally this just runs a stream and
   * collects the output for you.
   */
  async run(
    chatLike: ChatLike,
    options?: { signal: AbortSignal },
  ): Promise<AgentRunResult<ResolveAgentOutput<zO, Tools>>> {
    const result = this.stream(chatLike, options);
    while (true) {
      const next = await result.next();
      if (next.done) return next.value;
    }
  }

  /** Run the agent, streaming the response and tool calls. */
  async *stream(chatLike: ChatLike, options?: {
    signal: AbortSignal;
  }): AgentStreamIterator<ResolveAgentOutput<zO, Tools>> {
    options?.signal.throwIfAborted();
    const signal = options?.signal ?? new AbortController().signal;
    const initialHistory = convertChatLikeToChatItem(chatLike, "input_text");
    const adapter = await this.#adapter;

    // prepare a separate signal for tools that can be cancelled if a provider
    // fails too much. note that we do persist tool calls between assistant runs,
    // but in the failure case we have to abort.
    const toolController = new AbortController();
    signal.addEventListener("abort", () => toolController.abort(signal.reason));

    // tool use id -> result from tool
    const pendingTools = new Map<
      string,
      Promise<{ id: string; items: ChatItemToolResult[] }>
    >();

    let providerErrors = 0;
    const history: ChatItem[] = [];
    for (let turn = 0; turn < MAX_TURNS; turn++) {
      const newHistory: ChatItem[] = [];

      let failed = false;
      let err: unknown;
      try {
        const stream = await adapter.stream({
          model: this.#model,
          output: this.#output,
          tools: this.#tools,
          reasoningEffort: this.#reasoningEffort,
          systemPrompt: this.#instructions,
          history: [...initialHistory, ...history],
          signal,
        });

        while (true) {
          const { value: part, done } = await stream.next();
          if (done) {
            break;
          }

          const reIndexedPart = {
            ...part,
            index: part.index + history.length,
          };
          addStreamItem(newHistory, part);
          yield reIndexedPart;

          // execute tools immediately
          if (part.type === "tool_use") {
            assert(
              !pendingTools.has(part.tool_use_id),
              `Provider ${adapter.name} did not use unique tool use id: ${part.tool_use_id}`,
            );
            pendingTools.set(
              part.tool_use_id,
              this.#runTool(part, toolController.signal),
            );
          }
        }
      } catch (error) {
        err = error;
        failed = true;
      }

      // process tool outputs before re-looping
      if (pendingTools.size > 0) {
        try {
          for await (
            const result of iteratePromiseArray(pendingTools.values())
          ) {
            pendingTools.delete(result.id);

            for (const toolResult of result.items) {
              if (toolResult.type === "tool_result_text") {
                yield {
                  type: "tool_result_text",
                  index: newHistory.length + history.length,
                  tool_use_id: toolResult.tool_use_id,
                  content: toolResult.content,
                };
                newHistory.push(toolResult);
              } else if (toolResult.type === "tool_result_file") {
                yield {
                  type: "tool_result_file",
                  index: newHistory.length + history.length,
                  tool_use_id: toolResult.tool_use_id,
                  kind: toolResult.kind,
                  content: toolResult.content,
                };
                newHistory.push(toolResult);
              } else toolResult satisfies never;
            }
          }
        } catch (err) {
          let index = newHistory.length + history.length;
          for (const tool_use_id of pendingTools.keys()) {
            yield {
              type: "tool_result_text",
              index,
              tool_use_id,
              content: "Error: Tool call was cancelled",
            };
            index += 1;
          }

          if (err instanceof ModelOutput) {
            const { output } = err;
            toolController.abort(
              new Error(
                "Tool calls cancelled due to one returning ModelOutput",
              ),
            );
            return {
              history: newHistory,
              output,
              outputText: typeof output === "string"
                ? output
                : JSON.stringify(output),
            };
          }

          throw err;
        }

        // continue loop
        history.push(...newHistory);
        continue;
      }

      // If streaming fails halfway through a message, retry
      if (failed) {
        if (this.#noRetries) throw err;
        providerErrors++;
        if (providerErrors < MAX_PROVIDER_ERRORS) {
          // continue loop
          history.push(...newHistory);
          continue;
        } else {
          throw err;
        }
      }

      history.push(...newHistory);

      const finalItem = history.at(-1);
      let output: ResolveAgentOutput<zO, Tools>;
      if (this.#output) {
        if (!finalItem || finalItem.type !== "output_text") {
          throw new Error("LLM did not output");
        }

        try {
          output = this.#output.parse(
            JSON.parse(finalItem.content),
          ) as ResolveAgentOutput<zO, Tools>;
        } catch (err) {
          if (DEBUG_MODE) console.error("parsing failed", finalItem.content);
          const content = "Sorry, my output has an error: " +
            errMessage(err) +
            "\nI will try again to produce a JSON response.";
          history.push({
            type: "output_text",
            content,
          });
          yield {
            type: "delta_output_text",
            index: history.length - 1,
            delta: content,
          };
          continue;
        }
      } else {
        output = undefined as ResolveAgentOutput<zO, Tools>;
      }

      return {
        history,
        output,
        outputText: history
          .filter((history) => history.type === "output_text")
          .map((history) => history.content).join("\n"),
      };
    }
    throw new Error("MAX TURNS EXCEEDED");
  }

  async cli() {
    const decoder = new TextDecoder();
    const history: ChatItem[] = [];
    const prompt = "> ";
    crossPlatformLog(prompt);
    for await (const chunk of crossPlatformStdin()) {
      const content = decoder.decode(chunk).trim();
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
      crossPlatformLog(prompt);
    }
  }
}
