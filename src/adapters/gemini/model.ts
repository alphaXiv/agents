import type { GoogleGenAI } from "@google/genai";
import { GoogleGenAiModel, type GoogleGenAiModelOptions } from "../google_genai/model.ts";
import type { GoogleModels } from "../google_genai/models.ts";
import { GeminiAdapter } from "./adapter.ts";

export interface GeminiModelOptions<TModel extends GoogleModels> extends GoogleGenAiModelOptions<TModel> {
  apiKey?: string;
  baseUrl?: string;
  client?: GoogleGenAI;
}

export class GeminiModel<TModel extends GoogleModels> extends GoogleGenAiModel<TModel> {
  name = "Gemini";
  adapter: GeminiAdapter<TModel>;

  constructor(options: GeminiModelOptions<TModel>) {
    super(options);

    this.adapter = new GeminiAdapter({
      model: options.model,
      apiKey: options.apiKey,
      baseUrl: options.baseUrl,
      client: options.client,
      thinkingConfig: this.thinkingConfig,
    });
  }
}
