import type z from "zod";
import type { ToolResultLike } from "./types.ts";

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
  constructor({
    name,
    description,
    parameters,
    execute,
  }: {
    name: string;
    description: string;
    parameters: z.ZodType<zO, zI>;
    execute: ExecuteFunc<zO>;
  }) {
    this.#name = name;
    this.#description = description;
    this.#parameters = parameters;
    this.#execute = execute;
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

  get execute(): ExecuteFunc<zO> {
    return this.#execute;
  }
}
