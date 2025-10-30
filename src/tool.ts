import type z from "zod";
import type { ToolResultLike } from "./types.ts";
import { delay } from "@std/async/delay";

export type ExecuteResult = ToolResultLike | Promise<ToolResultLike>;

export type ExecuteFuncInput<O> = {
  param: O;
};
export type ExecuteFunc<O> = (input: ExecuteFuncInput<O>) => ExecuteResult;

export class Tool<zO, zI> {
  #name: string;
  #description: string;
  #parameters: z.ZodType<zO, zI>;
  #execute: ExecuteFunc<zO>;
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
    execute: ExecuteFunc<zO>;
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

  async execute(input: ExecuteFuncInput<zO>) {
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
