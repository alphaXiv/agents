import { Model } from "../model.ts";
import { OpenRouterAdapter, type OpenRouterAdapterOptions, type OpenRouterReasoningConfig } from "./adapter.ts";
import type { OpenRouterModels, OpenRouterReasoningEffort } from "./models.ts";

export type OpenRouterModelOptions<TModel extends OpenRouterModels> =
  & Omit<OpenRouterAdapterOptions<TModel>, "client" | "reasoning">
  & { effort?: OpenRouterReasoningEffort }
  & { reasoning?: OpenRouterReasoningConfig };

export class OpenRouterModel<TModel extends OpenRouterModels = OpenRouterModels> extends Model<TModel> {
  adapter: OpenRouterAdapter<TModel>;
  readonly effort: OpenRouterReasoningEffort | undefined;

  constructor(options: OpenRouterModelOptions<TModel>) {
    super(options);

    this.effort = options.reasoning?.effort ?? options.effort;

    this.adapter = new OpenRouterAdapter({
      ...options,
      reasoning: options.reasoning ??
        (this.effort && { enabled: this.effort !== "none", effort: this.effort }),
    });
  }
}
