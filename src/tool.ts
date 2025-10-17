import { ChatLike, ZodSchemaType } from "./types.ts";

export type ExecuteMetadata = { toolUseId: string };
export type ExecuteResult = ChatLike | Promise<ChatLike>;
export type ExecuteFunc<O> =
  | ((param: O) => ExecuteResult)
  | ((param: O, metadata: ExecuteMetadata) => ExecuteResult);

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

  get name() {
    return this.#name;
  }

  get description() {
    return this.#description;
  }

  get parameters() {
    return this.#parameters;
  }

  get execute() {
    return this.#execute;
  }
}
