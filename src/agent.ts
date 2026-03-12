import { assert } from "@std/assert/assert";
import { abortable } from "@std/async/abortable";
import { generate } from "@std/uuid/v7";
import type z from "zod";
import { ZodVoid } from "zod";
import { type Adapter, ADAPTERS } from "./adapters.ts";
import { addStreamItem } from "./client.ts";
import { signalAsyncLocalStorage } from "./storage.ts";
import { ModelOutput, type Tool } from "./tool.ts";
import {
  type ActiveTrace,
  newTrace,
  type Tracer,
  tracerAsyncLocalStorage,
  type TraceRef,
} from "./tracing.ts";
import type {
  AgentStreamIterator,
  ChatItem,
  ChatItemToolResult,
  ChatItemToolUse,
  ChatLike,
  ReasoningEffort,
  WithTraceId,
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
import { AssertionError } from "@std/assert";

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
  /** Optionally name this agent for tracing and debugging purposes */
  name?: string;
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
  /** What this is agent intended to do. Equivalent to a "system prompt". */
  instructions: string;
  /** Enable structured output */
  output?: z.ZodType<zO, zI>;
  /** Enable tool calls, which are automatically executed */
  tools?: M extends NoToolCallModels ? never : [...Tools];
  reasoningEffort?: ReasoningEffort;
  /** APIs which are not finalized and are subject to change. */
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
  history: WithTraceId<ChatItem>[];
  output: zO;
  get outputText(): string;
  inputTokens: number;
  outputTokens: number;
}

interface RunToolResult {
  id: string;
  items: WithTraceId<ChatItemToolResult>[];
}

export class Agent<
  zO,
  zI,
  M extends ModelString,
  // deno-lint-ignore no-explicit-any
  const Tools extends readonly Tool<any, any, any>[] = Tool<any, any, never>[],
  A extends Adapter<M> = Adapter<M>,
> {
  #name: string | null;
  #adapter: A | Promise<A>;
  #model: M;
  #instructions: string;
  #output?: z.ZodType<zO, zI>;
  // deno-lint-ignore no-explicit-any
  #tools: Tool<any, any, any>[];
  #reasoningEffort: ReasoningEffort;

  #noRetries = false;

  constructor(options: AgentOptions<zO, zI, M, Tools, A>) {
    this.#name = options.name ?? null;
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
    traceId: string,
    parent: TraceRef,
  ): Promise<RunToolResult> {
    using t = newTrace({
      id: traceId,
      type: "tool",
      parent,
      content: { name: use.kind },
    });
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
        signalAsyncLocalStorage.run(
          signal,
          () =>
            tracerAsyncLocalStorage.run(t, async () => {
              return await tool.execute({
                param: use.content ? JSON.parse(use.content) : undefined,
                signal,
              });
            }),
        ),
        signal,
      );

      if (result instanceof ModelOutput) throw result;

      return {
        id: use.tool_use_id,
        items: convertToolResultLikeToChatItem(result, use.tool_use_id, t.id),
      };
    } catch (err) {
      if (err instanceof ModelOutput) throw err;
      t.error(err);
      if (signal.aborted) throw err;
      return {
        id: use.tool_use_id,
        items: [{
          type: "tool_result_text" as const,
          tool_use_id: use.tool_use_id,
          content: "Error: " +
            (err instanceof Error ? err.message : (err as string).toString()),
          trace: t.id,
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
    options?: { signal?: AbortSignal; tracers?: Tracer[] },
  ): Promise<AgentRunResult<ResolveAgentOutput<zO, Tools>>> {
    const result = this.stream(chatLike, options);
    while (true) {
      const next = await result.next();
      if (next.done) return next.value;
    }
  }

  /** Run the agent, streaming the response and tool calls. */
  async *stream(
    chatLike: ChatLike,
    options?: { signal?: AbortSignal; tracers?: Tracer[] },
  ): AgentStreamIterator<ResolveAgentOutput<zO, Tools>> {
    options?.signal?.throwIfAborted();
    const signal = options?.signal ?? new AbortController().signal;
    const initialHistory = convertChatLikeToChatItem(chatLike, "input_text");
    const adapter = await this.#adapter;

    using tAgent = newTrace({
      type: "agent",
      content: this.#name ? { name: this.#name } : {},
      tracers: options?.tracers,
    });

    let totalInputTokens = 0;
    let totalOutputTokens = 0;

    try {
      // prepare a separate signal for tools that can be cancelled if a provider
      // fails too much. note that we do persist tool calls between assistant runs,
      // but in the failure case we have to abort.
      const toolController = new AbortController();
      signal.addEventListener(
        "abort",
        () => toolController.abort(signal.reason),
      );

      // tool use id -> result from tool
      const pendingTools = new Map<
        string,
        Promise<{ id: string; items: WithTraceId<ChatItemToolResult>[] }>
      >();

      let providerErrors = 0;
      const history: WithTraceId<ChatItem>[] = [];
      let modelCallReason: string = "init";
      for (let turn = 0; turn < MAX_TURNS; turn++) {
        const newHistory: WithTraceId<ChatItem>[] = [];

        using tModel = newTrace({
          type: "model",
          parent: tAgent,
          content: {
            reason: modelCallReason,
            provider: adapter.name,
            model: this.#model,
            inputTokens: null,
            outputTokens: null,
          },
        });
        using messageTracer = new MessageTracer(tModel);

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
              messageTracer.endMessageTraceIfStarted();
              if (part.inputTokens) totalInputTokens += part.inputTokens;
              if (part.outputTokens) totalOutputTokens += part.outputTokens;
              tModel.success({
                inputTokens: part.inputTokens,
                outputTokens: part.outputTokens,
              });
              break;
            }

            // execute tools immediately
            let trace: string | null = null;
            if (part.type === "tool_use") {
              assert(
                !pendingTools.has(part.tool_use_id),
                `Provider ${adapter.name} did not use unique tool use id: ${part.tool_use_id}`,
              );
              trace = generate();
              pendingTools.set(
                part.tool_use_id,
                this.#runTool(part, toolController.signal, trace, tAgent),
              );
            }

            // add the item to the stream first
            addStreamItem(newHistory, part);
            const chatItemType = newHistory[part.index].type;

            // automatically extract message start and end traces
            if (
              part.type === "delta_output_text" ||
              part.type === "delta_output_reasoning" ||
              part.type === "tool_use_start"
            ) {
              trace ??= messageTracer.startOrContinue({
                index: part.index,
                type: chatItemType,
              });
            } else {
              // other message types always force stopping the current trace
              messageTracer.endMessageTraceIfStarted();
              trace ??= tModel.id;
            }

            const reIndexedPart = {
              ...part,
              index: part.index + history.length,
              trace,
            };
            newHistory[part.index].trace ??= trace;
            yield reIndexedPart;
          }
        } catch (error) {
          messageTracer.cancel();
          tModel.error(error);
          err = error;
          failed = true;

          if (err instanceof AssertionError) throw err;
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
                    trace: toolResult.trace,
                  };
                  newHistory.push(toolResult);
                } else if (toolResult.type === "tool_result_file") {
                  yield {
                    type: "tool_result_file",
                    index: newHistory.length + history.length,
                    tool_use_id: toolResult.tool_use_id,
                    kind: toolResult.kind,
                    content: toolResult.content,
                    trace: toolResult.trace,
                  };
                  newHistory.push(toolResult);
                } else toolResult satisfies never;
              }
            }
          } catch (err) {
            // catching here is only for returning ModelOutput OR abort signal, which should cancel all tools so that all tools have an output.
            let index = newHistory.length + history.length;
            for (const tool_use_id of pendingTools.keys()) {
              yield {
                type: "tool_result_text",
                index,
                tool_use_id,
                content: "Error: Tool call was cancelled",
                trace: tAgent.id,
              };
              index += 1;
            }

            signal.throwIfAborted(); // throw if the abort signal fired
            assert(err instanceof ModelOutput); // if it didn't, it better fricking be a ModelOutput otherwise there's an agents sdk bug

            const { output } = err;
            toolController.abort(
              new Error(
                "Tool calls cancelled due to one returning ModelOutput",
              ),
            );
            return createRunResult({
              history: [...history, ...newHistory],
              output,
              totalInputTokens,
              totalOutputTokens,
            });
          }

          // continue loop
          history.push(...newHistory);
          modelCallReason = "tool";
          continue;
        }

        // If streaming fails halfway through a message, retry
        if (failed) {
          if (this.#noRetries) throw err;
          providerErrors++;
          if (providerErrors < MAX_PROVIDER_ERRORS) {
            // continue loop
            history.push(...newHistory);
            modelCallReason = "retry-provider-error";
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
            tAgent.log(
              "Retrying to due provider missing a final output_text for structured output",
            );
            modelCallReason = "retry-missing-output";
            // TODO: we should think hard about what this should really do
            continue;
          }

          try {
            output = this.#output.parse(
              JSON.parse(finalItem.content),
            ) as ResolveAgentOutput<zO, Tools>;
          } catch (err) {
            tAgent.log(
              "Retrying due to failed structured output parse from " +
                JSON.stringify(finalItem.content),
              err,
            );
            const content = "Sorry, my output has an error: " +
              errMessage(err) +
              "\nI will try again to produce a JSON response.";
            history.push({
              type: "output_text",
              content,
              trace: tAgent.id,
            });
            yield {
              type: "delta_output_text",
              index: history.length - 1,
              delta: content,
              trace: tAgent.id,
            };
            modelCallReason = "retry-malformed-output";
            continue;
          }
        } else {
          output = undefined as ResolveAgentOutput<zO, Tools>;
        }

        return createRunResult({
          history: history.filter(Boolean),
          totalInputTokens,
          totalOutputTokens,
          output,
        });
      }

      throw new Error("Exceeded maximum turns (" + MAX_TURNS + ")");
    } catch (err) {
      tAgent.error(err);
      throw err;
    }
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

/**
 * Stateful tracking of message trace objects.
 *
 * This use-case is why the lower
 * level `newTrace` function does not enforce callback wrapping.
 */
class MessageTracer {
  current: {
    trace: ActiveTrace<"message">;
    index: number;
  } | null = null;
  modelTrace: TraceRef;

  constructor(modelTrace: TraceRef) {
    this.modelTrace = modelTrace;
  }

  startOrContinue({ index, type }: { index: number; type: ChatItem["type"] }) {
    // if the index is different than the one we're locked in on tracing, replace it
    if (!this.current || this.current.index !== index) {
      this.endMessageTraceIfStarted();
      const trace = newTrace({
        type: "message",
        parent: this.modelTrace,
        content: { type },
      });
      this.current = { trace, index };
    }
    return this.current.trace.id;
  }

  endMessageTraceIfStarted() {
    if (this.current) this.current.trace.success();
    this.current = null;
  }

  cancel() {
    if (this.current) {
      this.current.trace.error(new Error("Cancelled by provider error"));
    }
    this.current = null;
  }

  [Symbol.dispose]() {
    this.endMessageTraceIfStarted();
  }
}

function createRunResult<T>(completion: {
  history: WithTraceId<ChatItem>[];
  totalInputTokens: number;
  totalOutputTokens: number;
  output: T;
}): AgentRunResult<T> {
  const { output, history } = completion;
  return {
    history,
    output,
    inputTokens: completion.totalInputTokens,
    outputTokens: completion.totalOutputTokens,
    // using a getter so that if you console.log this object, you
    // don't see all the data twice.
    get outputText() {
      return output != null
        ? (typeof output === "string" ? output : JSON.stringify(output))
        : history.filter((x) => x.type === "output_text")
          .map((x) => x.content)
          .join("\n\n").trim();
    },
  };
}
