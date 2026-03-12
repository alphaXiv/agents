import type z from "zod";
import type { ToolResultLike } from "./types.ts";
import { delay } from "@std/async/delay";

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

export type ExecuteFuncInput<O> = {
  param: O;
  signal: AbortSignal;
};
export type ExecuteFunc<O, MO = never> = (
  input: ExecuteFuncInput<O>,
) => ExecuteResult<MO>;

export class Tool<zO, zI, TModelOutput = never> {
  #name: string;
  #description: string;
  #parameters: z.ZodType<zO, zI>;
  #execute: ExecuteFunc<zO, TModelOutput>;
  #retries: number;

  constructor({
    name,
    description,
    parameters,
    execute,
    retries,
  }: {
    name: string;
    description: string;
    parameters: z.ZodType<zO, zI>;
    execute: ExecuteFunc<zO, TModelOutput>;
    retries?: number;
  }) {
    this.#name = name;
    this.#description = description;
    this.#parameters = parameters;
    this.#execute = execute;
    this.#retries = retries ?? 0;
  }

  get name(): string {
    return this.#name;
  }

  get description(): string {
    return this.#description;
  }

  get parameters(): z.ZodType<zO, zI> {
    return this.#parameters;
  }

  async execute(
    input: ExecuteFuncInput<zO>,
  ): Promise<ToolResultLike | ModelOutput<TModelOutput>> {
    let lastError: unknown;
    for (let i = 0; i < this.#retries + 1; i++) {
      try {
        return await this.#execute(input);
      } catch (err) {
        await delay(500 * (i ** 2));
        lastError = err;
      }
    }
    throw lastError;
  }
}
