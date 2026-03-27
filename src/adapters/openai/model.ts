import { Model } from "../model.ts";
import type { OpenResponsesServiceTier } from "../open_responses/adapter.ts";
import { OpenAIAdapter, type OpenAIAdapterOptions } from "./adapter.ts";
import {
  getDefaultReasoningEffort,
  openAiModelReasoningSupport,
  type OpenAIModels,
  type OpenAIReasoningEffort,
  type SupportedReasoningEffort,
} from "./models.ts";

type EffortOption<TModel extends OpenAIModels> = SupportedReasoningEffort<TModel> extends never ? Record<never, never>
  : { effort?: SupportedReasoningEffort<TModel> };

export type OpenAIModelOptions<TModel extends OpenAIModels> = OpenAIAdapterOptions<TModel> & EffortOption<TModel>;

export class OpenAIModel<TModel extends OpenAIModels> extends Model<TModel> {
  adapter: OpenAIAdapter<TModel>;
  readonly effort: SupportedReasoningEffort<TModel> | undefined;
  readonly serviceTier: OpenResponsesServiceTier | undefined;

  constructor(options: OpenAIModelOptions<TModel>) {
    super(options);

    const modelConfig = openAiModelReasoningSupport[options.model];
    const typedOptions = options as { effort?: SupportedReasoningEffort<OpenAIModels> };

    this.effort = "schema" in modelConfig ? typedOptions.effort ?? getDefaultReasoningEffort(options.model) : undefined;
    this.serviceTier = options.serviceTier;

    this.adapter = new OpenAIAdapter({
      model: options.model,
      apiKey: options.apiKey,
      baseUrl: options.baseUrl,
      parallelToolCalls: options.parallelToolCalls,
      serviceTier: this.serviceTier,
      reasoning: this.effort
        ? {
          effort: this.effort as OpenAIReasoningEffort,
          summary: this.effort === "none" ? undefined : "auto",
        }
        : undefined,
    });
  }
}
