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

export type ExecuteFunc<O, MO = never> = (input: ExecuteFuncInput<O>, signal: AbortSignal) => ExecuteResult<MO>;

function createAbortableDelay(ms: number, signal: AbortSignal): Promise<void> {
  if (ms <= 0) {
    signal.throwIfAborted();
    return Promise.resolve();
  }

  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, ms);

    function onAbort() {
      clearTimeout(timeout);
      reject(signal.reason);
    }

    if (signal.aborted) {
      onAbort();
      return;
    }

    signal.addEventListener("abort", onAbort, { once: true });
  });
}

interface ToolOptions<zO, zI, TModelOutput> {
  name: string;
  description: string;
  parameters: z.ZodType<zO, zI>;
  execute: ExecuteFunc<zO, TModelOutput>;
  retries?: number;
  signal?: AbortSignal;
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

  constructor({
    name,
    description,
    parameters,
    execute,
    retries,
    signal,
  }: ToolOptions<zO, zI, TModelOutput>) {
    this.#name = name;
    this.#normalizedName = normalizeToolName(name);
    this.#description = description;
    this.#parameters = parameters;
    this.#execute = execute;
    this.#retries = retries ?? 0;
    this.#signal = signal;
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

  async execute(input: ExecuteFuncInput<zO>, signal: AbortSignal): Promise<ToolResultLike | ModelOutput<TModelOutput>> {
    const combinedSignal = this.#signal ? AbortSignal.any([signal, this.#signal]) : signal;

    let lastError: unknown;
    for (let i = 0; i < this.#retries + 1; i++) {
      combinedSignal.throwIfAborted();
      try {
        return await this.#execute(input, combinedSignal);
      } catch (err) {
        lastError = err;
        if (combinedSignal.aborted || i === this.#retries) {
          break;
        }
        await createAbortableDelay(500 * (i ** 2), combinedSignal);
      }
    }

    throw lastError;
  }
}
