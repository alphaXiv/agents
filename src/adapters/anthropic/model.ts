import Anthropic from "@anthropic-ai/sdk";
import { crossPlatformEnv, requireEnv } from "../../util.ts";
import { Model } from "../model.ts";
import { AnthropicAdapter, type AnthropicAdapterOptions } from "./adapter.ts";
import {
  type AnthropicModels,
  anthropicModelThinkingSupport,
  type EffortLevel,
  getAnthropicMessagesStreamConfig,
  type SupportedEffortLevel,
  type SupportedThinkingLevel,
  type SupportsInterleaved,
  type ThinkingLevel,
} from "./models.ts";

// Only include thinkingLevel for models with extended thinking (adaptive-only models don't use budget_tokens).
type ThinkingLevelOption<TModel extends AnthropicModels> = SupportedThinkingLevel<TModel> extends never
  ? Record<never, never>
  : { thinkingLevel?: SupportedThinkingLevel<TModel> };

// Only include effort for models that support it (extended-only models don't have an effort API).
type EffortOption<TModel extends AnthropicModels> = SupportedEffortLevel<TModel> extends never ? Record<never, never>
  : { effort?: SupportedEffortLevel<TModel> };

// Only include interleaved for models with extended thinking; adaptive models handle it automatically.
type InterleavedOption<TModel extends AnthropicModels> = SupportsInterleaved<TModel> extends true
  ? { interleaved?: boolean }
  : Record<never, never>;

export type AnthropicModelOptions<TModel extends AnthropicModels> =
  & Omit<AnthropicAdapterOptions<TModel>, "client" | "streamConfig">
  & ThinkingLevelOption<TModel>
  & EffortOption<TModel>
  & InterleavedOption<TModel>
  & { baseUrl?: string; apiKey?: string };

export class AnthropicModel<TModel extends AnthropicModels = AnthropicModels> extends Model<TModel> {
  adapter: AnthropicAdapter<TModel>;
  readonly thinkingLevel: SupportedThinkingLevel<TModel> | undefined;
  readonly effort: SupportedEffortLevel<TModel> | undefined;
  readonly interleaved: boolean | undefined;

  constructor(options: AnthropicModelOptions<TModel>) {
    super(options);

    const modelConfig = anthropicModelThinkingSupport[options.model];
    // Cast to access fields that are conditionally typed per model
    const { thinkingLevel, effort, interleaved } = options as {
      thinkingLevel?: ThinkingLevel;
      effort?: EffortLevel;
      interleaved?: boolean;
    };

    this.thinkingLevel =
      ("schema" in modelConfig ? (thinkingLevel ?? modelConfig.schema.parse(undefined)) : undefined) as
        | SupportedThinkingLevel<TModel>
        | undefined;

    this.effort = ("effortSchema" in modelConfig ? (effort ?? modelConfig.effortSchema.parse(undefined)) : undefined) as
      | SupportedEffortLevel<TModel>
      | undefined;

    this.interleaved = interleaved;

    const thinkingConfig = getAnthropicMessagesStreamConfig({
      model: options.model,
      thinkingLevel: this.thinkingLevel as ThinkingLevel | undefined,
      effort: this.effort as EffortLevel | undefined,
      interleaved: this.interleaved,
    });

    this.adapter = new AnthropicAdapter<TModel>({
      model: options.model,
      streamConfig: thinkingConfig,
      client: new Anthropic({
        apiKey: options.apiKey ?? requireEnv("ANTHROPIC_API_KEY"),
        baseURL: options.baseUrl ?? crossPlatformEnv("ANTHROPIC_BASE_URL") ?? "https://api.anthropic.com",
      }),
    });
  }
}
