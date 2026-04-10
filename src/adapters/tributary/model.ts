import { Model } from "../model.ts";
import { TributaryAdapter, type TributaryAdapterOptions } from "./adapter.ts";
import type { TributaryModels } from "./models.ts";
import type { ReasoningEffort } from "openai/resources/shared";

export type TributaryModelOptions<TModel extends TributaryModels> =
  & Omit<TributaryAdapterOptions<TModel>, "client" | "reasoningEffort">
  & { effort?: ReasoningEffort };

export class TributaryModel<TModel extends TributaryModels = TributaryModels> extends Model<TModel> {
  adapter: TributaryAdapter<TModel>;
  readonly effort: ReasoningEffort | undefined;

  constructor(options: TributaryModelOptions<TModel>) {
    super(options);
    this.effort = options.effort;
    this.adapter = new TributaryAdapter({
      ...options,
      reasoningEffort: options.effort,
    });
  }
}
