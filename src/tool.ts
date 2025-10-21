import { ChatLike, ZodSchemaType } from "./types.ts";

export type ExecuteResult = ChatLike | Promise<ChatLike>;

export type ExecuteFuncInput<O> = {
  param: O;
  toolUseId: string;
};
export type ExecuteFunc<O> = (input: ExecuteFuncInput<O>) => ExecuteResult;

export class Tool<O> {
  #name: string;
  #description: string;
  #parameters: ZodSchemaType<O>;
  #execute: ExecuteFunc<O>;
  constructor({
    name,
    description,
    parameters,
    execute,
  }: {
    name: string;
    description: string;
    parameters: ZodSchemaType<O>;
    execute: ExecuteFunc<O>;
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

  get parameters(): ZodSchemaType<O> {
    return this.#parameters;
  }

  get execute(): ExecuteFunc<O> {
    return this.#execute;
  }
}
