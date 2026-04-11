import { assert } from "@std/assert/assert";
import { generate } from "@std/uuid/v7";
import z from "zod";
import type { Model } from "./adapters/model.ts";
import { type ModelLike, resolveModel } from "./adapters/model_resolver.ts";
import { addStreamItem } from "./client.ts";
import { createStructuredOutputRetryFeedback } from "./constants.ts";
import { signalAsyncLocalStorage } from "./storage.ts";
import { type AnyTool, ModelOutput, type Tool } from "./tool.ts";
import {
  type ActiveTrace,
  MessageTracer,
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
  StreamItem,
  WithTraceId,
} from "./types.ts";
import { convertChatLikeToChatItem, convertToolResultLikeToChatItem, errMessage, iteratePromiseArray } from "./util.ts";

const DEFAULT_MAX_TURNS = 100;
const DEFAULT_MAX_RETRIES = 3;

type ExtractModelOutput<Tools extends Tool[]> = [Tools[number]] extends [never] ? never
  : Tools[number] extends Tool<infer _, infer _, infer MO> ? MO
  : never;

type ResolveAgentOutput<zO, Tools extends Tool[]> = unknown extends zO
  ? ([ExtractModelOutput<Tools>] extends [never] ? undefined : ExtractModelOutput<Tools> | undefined)
  : ([ExtractModelOutput<Tools>] extends [never] ? zO : zO | ExtractModelOutput<Tools>);

export interface AgentRunResult<zO> {
  output: zO;
  history: WithTraceId<ChatItem>[];
  inputTokens: number;
  outputTokens: number;
  get outputText(): string;
}

interface RunToolResult {
  id: string;
  items: WithTraceId<ChatItemToolResult>[];
}

export interface AgentRunOptions {
  /** Additional tracers for this run */
  tracers?: Tracer[];
  /** Cancellation signal */
  signal?: AbortSignal;
}

export interface AgentOptions<zO, zI, Tools extends AnyTool[]> {
  /** Optionally name this agent for tracing and debugging purposes */
  name?: string;
  /**
   * One or more models that this agent can use to think and generate responses.\
   * If multiple models are provided, the agent will use them in order, falling
   * back to the next model if the previous one fails or is unavailable.
   */
  model: ModelLike | [ModelLike, ...ModelLike[]];
  /** What this is agent intended to do. Equivalent to a "system prompt". */
  instructions: string;
  /** Enable tool calls, which are automatically executed */
  tools?: Tools;
  /** Optionally specify a schema for the final output. If not provided, the agent will return the final model response as a string. */
  output?: z.ZodType<zO, zI>;
  /**
   * Maximum number of agentic turns (model call -> tool execution cycles).
   * @experimental - might be removed or have its behaviour modified without any notice
   * @default 100
   */
  maxTurns?: number;
  /**
   * Maximum number of consecutive retries per model before moving to the next
   * model in the fallback list. After exhausting all models, the last error is
   * thrown. Models are tried in round-robin when retries are available.
   * @experimental - might be removed or have its behaviour modified without any notice
   * @default 3
   */
  maxRetries?: number;
}

export class Agent<zO = unknown, zI = unknown, const Tools extends AnyTool[] = []> {
  #name?: string;
  #models: Model[];
  #instructions: string;
  // deno-lint-ignore no-explicit-any
  #tools: Tool<any, any, any>[];
  #output?: z.ZodType<zO, zI>;
  #maxTurns: number;
  #maxRetries: number;

  constructor(options: AgentOptions<zO, zI, Tools>) {
    this.#name = options.name;
    this.#models = (Array.isArray(options.model) ? options.model : [options.model]).map(resolveModel);
    this.#instructions = options.instructions;
    this.#tools = options.tools?.slice() ?? [];
    this.#output = options.output;
    this.#maxTurns = options.maxTurns ?? DEFAULT_MAX_TURNS;
    this.#maxRetries = options.maxRetries ?? DEFAULT_MAX_RETRIES;
  }

  /**
   * Run the agent without streaming. Internally runs a stream and collects
   * the output for you.
   */
  async run(
    chatLike: ChatLike,
    options?: AgentRunOptions,
  ): Promise<AgentRunResult<ResolveAgentOutput<zO, Tools>>> {
    const stream = this.stream(chatLike, options);
    while (true) {
      const next = await stream.next();
      if (next.done) return next.value;
    }
  }

  /**
   * Run the agent, streaming the response and tool calls.
   *
   * The main loop has three phases per turn:
   * 1. {@linkcode #invokeModel} — stream from a model (with round-robin retries)
   * 2. {@linkcode #collectToolResults} — await eagerly-dispatched tool results
   * 3. {@linkcode #tryParseOutput} — parse structured output (with retry feedback)
   */
  async *stream(
    chatLike: ChatLike,
    options?: AgentRunOptions,
  ): AgentStreamIterator<ResolveAgentOutput<zO, Tools>> {
    const signal = options?.signal ?? new AbortController().signal;
    signal.throwIfAborted();

    const initialHistory = convertChatLikeToChatItem(chatLike, "input_text");

    using agentTrace = newTrace({
      type: "agent",
      parent: tracerAsyncLocalStorage.getStore(),
      tracers: options?.tracers,
      content: { name: this.#name },
    });

    let totalInputTokens = 0;
    let totalOutputTokens = 0;

    try {
      const toolController = new AbortController();
      signal.addEventListener("abort", () => toolController.abort(signal.reason));

      const pendingTools = new Map<string, Promise<RunToolResult>>();
      const history: WithTraceId<ChatItem>[] = [];
      let modelCallReason = "init";

      for (let turn = 0; turn < this.#maxTurns; turn++) {
        signal.throwIfAborted();

        // Phase 1: Stream from a model (dispatches tool calls eagerly)
        const { newHistory, inputTokens, outputTokens } = yield* this.#invokeModel({
          signal,
          initialHistory,
          history,
          pendingTools,
          toolSignal: toolController.signal,
          agentTrace,
          modelCallReason,
        });
        totalInputTokens += inputTokens;
        totalOutputTokens += outputTokens;

        // Phase 2: Collect tool results (if any were dispatched during streaming)
        if (pendingTools.size > 0) {
          const modelOutput = yield* this.#collectToolResults({
            signal,
            history,
            newHistory,
            pendingTools,
            toolController,
            agentTrace,
          });
          if (modelOutput) {
            return createRunResult({
              history: [...history, ...newHistory],
              output: modelOutput.output as ResolveAgentOutput<zO, Tools>,
              totalInputTokens,
              totalOutputTokens,
            });
          }
          history.push(...newHistory);
          modelCallReason = "tool";
          continue;
        }

        history.push(...newHistory);

        // Phase 3: Parse final output (retries on structured output errors)
        const result = this.#tryParseOutput(history, agentTrace);
        if (!result.ok) {
          if (result.feedback) {
            history.push({ type: "output_text", content: result.feedback, trace: agentTrace.id });
            yield {
              type: "delta_output_text",
              index: history.length - 1,
              delta: result.feedback,
              trace: agentTrace.id,
            };
          }
          modelCallReason = result.reason;
          continue;
        }

        return createRunResult({
          history,
          output: result.output,
          totalInputTokens,
          totalOutputTokens,
        });
      }

      throw new Error("Exceeded maximum turns (" + this.#maxTurns + ")");
    } catch (error) {
      agentTrace.error(error);
      throw error;
    }
  }

  /**
   * Try each model in round-robin, up to `maxRetries` full cycles.
   * Yields traced stream items. Tools are dispatched eagerly into `pendingTools`.
   */
  async *#invokeModel(options: {
    signal: AbortSignal;
    initialHistory: ChatItem[];
    history: WithTraceId<ChatItem>[];
    pendingTools: Map<string, Promise<RunToolResult>>;
    toolSignal: AbortSignal;
    agentTrace: ActiveTrace<"agent">;
    modelCallReason: string;
  }): AsyncGenerator<
    WithTraceId<StreamItem>,
    { newHistory: WithTraceId<ChatItem>[]; inputTokens: number; outputTokens: number }
  > {
    const { signal, initialHistory, history, pendingTools, toolSignal, agentTrace } = options;
    let { modelCallReason } = options;

    const newHistory: WithTraceId<ChatItem>[] = [];
    let lastError: unknown;

    for (let retry = 0; retry < this.#maxRetries; retry++) {
      for (const model of this.#models) {
        const adapter = model.adapter;

        using modelTrace = newTrace({
          type: "model",
          parent: agentTrace,
          content: {
            reason: modelCallReason,
            provider: adapter.name,
            model: adapter.model,
            inputTokens: null,
            outputTokens: null,
          },
        });
        using messageTracer = new MessageTracer(modelTrace);

        try {
          const adapterStream = adapter.stream({
            instructions: this.#instructions,
            tools: this.#tools,
            output: this.#output,
            history: [...initialHistory, ...history],
            signal,
          });

          while (true) {
            const { value: part, done } = await adapterStream.next();
            if (done) {
              messageTracer.endMessageTraceIfStarted();
              modelTrace.success({
                inputTokens: part.inputTokens,
                outputTokens: part.outputTokens,
              });
              return {
                newHistory,
                inputTokens: part.inputTokens ?? 0,
                outputTokens: part.outputTokens ?? 0,
              };
            }

            // Eager tool dispatch: start tool execution while the model is still streaming
            let trace: string | null = null;
            if (part.type === "tool_use") {
              assert(
                !pendingTools.has(part.tool_use_id),
                `Provider ${adapter.name} did not use unique tool use id: ${part.tool_use_id}`,
              );
              trace = generate();
              pendingTools.set(
                part.tool_use_id,
                this.#runTool(part, signal, toolSignal, trace, agentTrace),
              );
            }

            // Message tracing
            if (
              part.type === "delta_output_text" ||
              part.type === "delta_output_reasoning" ||
              part.type === "tool_use_start"
            ) {
              trace ??= messageTracer.startOrContinue({
                index: part.index,
                type: ({
                  delta_output_text: "output_text",
                  delta_output_reasoning: "output_reasoning",
                  tool_use_start: "tool_use",
                } as const)[part.type],
              });
            } else {
              messageTracer.endMessageTraceIfStarted();
              trace ??= modelTrace.id;
            }

            addStreamItem(newHistory, { ...part, trace });
            yield { ...part, index: part.index + history.length, trace };
          }
        } catch (error) {
          messageTracer.cancel();
          modelTrace.error(error);
          lastError = error;
          agentTrace.log(`Model ${adapter.name} failed: ${errMessage(error)}`, error);
          modelCallReason = "retry-provider-error";
        }
      }
    }

    throw lastError;
  }

  /**
   * Collect results from eagerly-dispatched tools. Yields tool result stream
   * items as they complete. Returns a `ModelOutput` if a tool short-circuited,
   * or `null` to continue the agent loop normally.
   */
  async *#collectToolResults(options: {
    signal: AbortSignal;
    history: WithTraceId<ChatItem>[];
    newHistory: WithTraceId<ChatItem>[];
    pendingTools: Map<string, Promise<RunToolResult>>;
    toolController: AbortController;
    agentTrace: ActiveTrace<"agent">;
  }): AsyncGenerator<WithTraceId<StreamItem>, ModelOutput<unknown> | null> {
    const { signal, history, newHistory, pendingTools, toolController, agentTrace } = options;

    try {
      signal.throwIfAborted();
      for await (const result of iteratePromiseArray(pendingTools.values())) {
        pendingTools.delete(result.id);
        for (const toolResult of result.items) {
          const streamItem: WithTraceId<StreamItem> = toolResult.type === "tool_result_text"
            ? {
              type: "tool_result_text",
              index: newHistory.length + history.length,
              tool_use_id: toolResult.tool_use_id,
              content: toolResult.content,
              trace: toolResult.trace,
            }
            : {
              type: "tool_result_file",
              index: newHistory.length + history.length,
              tool_use_id: toolResult.tool_use_id,
              kind: toolResult.kind,
              content: toolResult.content,
              trace: toolResult.trace,
            };
          yield streamItem;
          newHistory.push(toolResult);
        }
      }
      return null;
    } catch (error) {
      // Emit error results for any remaining pending tools
      let index = newHistory.length + history.length;
      for (const tool_use_id of pendingTools.keys()) {
        yield {
          type: "tool_result_text",
          index,
          tool_use_id,
          content: "Error: Tool call was cancelled",
          trace: agentTrace.id,
        };
        index += 1;
      }

      signal.throwIfAborted();
      assert(error instanceof ModelOutput);

      toolController.abort(new Error("Tool calls cancelled due to one returning ModelOutput"));
      return error;
    }
  }

  /**
   * Try to parse the final model output. When structured output is configured,
   * validates the last `output_text` against the schema. Returns retry info
   * with feedback on parse failure so the model can self-correct.
   */
  #tryParseOutput(
    history: WithTraceId<ChatItem>[],
    agentTrace: ActiveTrace<"agent">,
  ):
    | { ok: true; output: ResolveAgentOutput<zO, Tools> }
    | { ok: false; reason: string; feedback?: string } {
    if (!this.#output) {
      return { ok: true, output: undefined as ResolveAgentOutput<zO, Tools> };
    }

    const finalItem = history.at(-1);
    if (!finalItem || finalItem.type !== "output_text") {
      agentTrace.log("Retrying due to provider missing a final output_text for structured output");
      return { ok: false, reason: "retry-missing-output" };
    }

    try {
      const output = this.#output.parse(
        JSON.parse(finalItem.content),
      ) as ResolveAgentOutput<zO, Tools>;
      return { ok: true, output };
    } catch (error) {
      agentTrace.log("Retrying due to failed structured output parse from " + JSON.stringify(finalItem.content), error);
      return {
        ok: false,
        reason: "retry-malformed-output",
        feedback: createStructuredOutputRetryFeedback(errMessage(error)),
      };
    }
  }

  /** Execute a single tool call with tracing and error handling. */
  async #runTool(
    use: ChatItemToolUse,
    agentSignal: AbortSignal,
    toolSignal: AbortSignal,
    traceId: string,
    parent: TraceRef,
  ): Promise<RunToolResult> {
    using trace = newTrace({
      id: traceId,
      type: "tool",
      parent,
      content: { name: use.kind },
    });
    try {
      const tool = this.#tools.find((t) => t.normalizedName === use.kind);
      if (!tool) {
        throw new Error(`Tool does not exist: ${use.kind}`);
      }

      let param = use.content ? JSON.parse(use.content) : undefined;
      try {
        // When it is meant to be a void, it doesn't matter if the provider sent something - we ignore it and pass undefined to the tool
        // This sometimes happens for providers that don't support providing "nothing" as input, e.g. OpenAI's function calls
        param = tool.parameters instanceof z.ZodVoid ? undefined : tool.parameters.parse(param);
      } catch (error) {
        throw new Error(`Invalid parameters for tool: ${errMessage(error)}`);
      }

      const result = await signalAsyncLocalStorage.run(
        toolSignal,
        () => tracerAsyncLocalStorage.run(trace, () => tool.execute(param, { signal: toolSignal })),
      );

      if (result instanceof ModelOutput) {
        throw result;
      }

      return {
        id: use.tool_use_id,
        items: convertToolResultLikeToChatItem(result, use.tool_use_id, trace.id),
      };
    } catch (error) {
      if (error instanceof ModelOutput) throw error;

      trace.error(error);
      if (agentSignal.aborted) throw error;

      return {
        id: use.tool_use_id,
        items: [{
          type: "tool_result_text" as const,
          tool_use_id: use.tool_use_id,
          content: `Error: ${errMessage(error)}`,
          trace: trace.id,
        }],
      };
    }
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
