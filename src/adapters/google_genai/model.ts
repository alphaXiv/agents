import type { ThinkingConfig } from "@google/genai";
import { Model, type ModelOptions } from "../model.ts";
import type { GoogleGenAiAdapter } from "./adapter.ts";
import {
  getDefaultThinkingLevel,
  getGenAiThinkingLevel,
  getLegacyThinkingBudget,
  type GoogleModels,
  googleModelSupportedThinkingLevels,
  type SupportedThinkingLevel,
} from "./models.ts";

export interface GoogleGenAiModelOptions<TModel extends GoogleModels> extends ModelOptions<TModel> {
  thinkingLevel?: SupportedThinkingLevel<TModel>;
}

export abstract class GoogleGenAiModel<TModel extends GoogleModels> extends Model<TModel> {
  abstract override adapter: GoogleGenAiAdapter<TModel>;
  readonly thinkingLevel: SupportedThinkingLevel<TModel>;

  constructor(options: GoogleGenAiModelOptions<TModel>) {
    super(options);
    this.thinkingLevel = options.thinkingLevel ?? getDefaultThinkingLevel(options.model);
  }

  get thinkingConfig(): ThinkingConfig {
    const supportedThinkingLevels = googleModelSupportedThinkingLevels[this.model];
    switch (supportedThinkingLevels.type) {
      case "unsupported":
        return { includeThoughts: false, thinkingBudget: 0 };
      case "legacyThinkingBudget":
        return { includeThoughts: true, thinkingBudget: getLegacyThinkingBudget(this.thinkingLevel) };
      default:
        return { includeThoughts: true, thinkingLevel: getGenAiThinkingLevel(this.thinkingLevel) };
    }
  }
}
