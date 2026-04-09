import { delay } from "@std/async/delay";
import type z from "zod";
import type { ToolResultLike } from "./types.ts";

export class ModelOutput<T> {
  output: T;
  constructor(output: T) {
    this.output = output;
  }
}

export type ExecuteResult<MO = never> =
  | ToolResultLike
  | ModelOutput<MO>
  | Promise<ToolResultLike | ModelOutput<MO>>;

export type ExecuteFuncInput<O> = O;

export interface ExecuteContext {
  signal: AbortSignal;
}

export type ExecuteFunc<O, MO = never> = (input: ExecuteFuncInput<O>, context: ExecuteContext) => ExecuteResult<MO>;

interface ToolOptions<zO, zI, TModelOutput> {
  /**
   * Display name of the tool.
   * Can be humanly readable, will be normalized depending on the model requirements.
   * @example
   * "Validate LaTeX Tool!" might get normalized to "validate_latex_tool".
   */
  name: string;
  /** Description of what the tool does, exposed to the model. */
  description: string;
  /** Zod schema defining the tool's input parameters. */
  parameters: z.ZodType<zO, zI>;
  /**
   * Function called when the model invokes this tool.
   * You should not invoke it directly yourself.
   */
  execute: ExecuteFunc<zO, TModelOutput>;
  /**
   * Number of times to retry on failure before giving up.
   * @default 0
   */
  retries?: number;
  /** Signal that can cancel tool execution from the outside. */
  signal?: AbortSignal;
  /**
   * Total timeout in milliseconds for each tool execution.
   * Does not reset on {@linkcode retries} attempts.
   * @experimental - might be removed or have its behaviour modified without any notice
   */
  timeout?: number;
}

/**
 * Make sure tool names are normalized to a consistent format (lowercase, underscores, max length) to avoid issues with different LLMs having different requirements for tool naming.
 * @example "Validate LaTeX Tool!" -> "validate_latex_tool"
 */
function normalizeToolName(name: string): string {
  let normalized = name
    .toLowerCase()
    .replaceAll(" ", "_")
    .replace(/[^a-zA-Z0-9_-]/g, "");

  // Ensure name starts with letter or underscore
  if (!/^[a-zA-Z_]/.test(normalized)) {
    normalized = "_" + normalized;
  }

  return normalized.slice(0, 64);
}

// deno-lint-ignore no-explicit-any
export type AnyTool = Tool<any, any, any>;

export class Tool<zO = unknown, zI = unknown, TModelOutput = unknown> {
  #name: string;
  #normalizedName: string;
  #description: string;
  #parameters: z.ZodType<zO, zI>;
  #execute: ExecuteFunc<zO, TModelOutput>;
  #retries: number;
  #signal?: AbortSignal;
  #timeout?: number;

  constructor({
    name,
    description,
    parameters,
    execute,
    retries,
    signal,
    timeout,
  }: ToolOptions<zO, zI, TModelOutput>) {
    this.#name = name;
    this.#normalizedName = normalizeToolName(name);
    this.#description = description;
    this.#parameters = parameters;
    this.#execute = execute;
    this.#retries = retries ?? 0;
    this.#signal = signal;
    this.#timeout = timeout;
  }

  get name(): string {
    return this.#name;
  }

  get normalizedName(): string {
    return this.#normalizedName;
  }

  get description(): string {
    return this.#description;
  }

  get parameters(): z.ZodType<zO, zI> {
    return this.#parameters;
  }

  async execute(
    input: ExecuteFuncInput<zO>,
    context: ExecuteContext,
  ): Promise<ToolResultLike | ModelOutput<TModelOutput>> {
    const signals: AbortSignal[] = [context.signal];
    if (this.#signal) signals.push(this.#signal);
    if (this.#timeout !== undefined) signals.push(AbortSignal.timeout(this.#timeout));

    const combinedSignal = AbortSignal.any(signals);

    let lastError: unknown;
    for (let i = 0; i < this.#retries + 1; i++) {
      combinedSignal.throwIfAborted();
      try {
        return await this.#execute(input, { signal: combinedSignal });
      } catch (err) {
        lastError = err;
        if (combinedSignal.aborted || i === this.#retries) {
          break;
        }
        await delay(500 * (i ** 2), { signal: combinedSignal });
      }
    }

    throw lastError;
  }
}
