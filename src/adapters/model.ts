import type { Adapter } from "./adapter.ts";

export interface ModelOptions<SupportedModels extends string> {
  model: SupportedModels;
}

export abstract class Model<SupportedModels extends string = string> {
  abstract adapter: Adapter<SupportedModels>;
  model: SupportedModels;

  constructor(options: ModelOptions<SupportedModels>) {
    this.model = options.model;
  }
}
