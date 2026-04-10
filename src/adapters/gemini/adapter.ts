import { GoogleGenAI } from "@google/genai";
import { requireEnv } from "../../util.ts";
import { GoogleGenAiAdapter, type GoogleGenAiAdapterOptions } from "../google_genai/adapter.ts";
import type { GoogleModels } from "../google_genai/models.ts";

export interface GeminiAdapterOptions<TModel extends GoogleModels>
  extends Omit<GoogleGenAiAdapterOptions<TModel>, "client"> {
  apiKey?: string;
  client?: GoogleGenAI;
}

export class GeminiAdapter<TModel extends GoogleModels> extends GoogleGenAiAdapter<TModel> {
  name = "Gemini";

  constructor(options: GeminiAdapterOptions<TModel>) {
    const apiKey = options.apiKey ?? requireEnv("GEMINI_API_KEY");
    super({
      client: options.client ?? new GoogleGenAI({ apiKey }),
      model: options.model,
      baseUrl: options.baseUrl,
      thinkingConfig: options.thinkingConfig,
    });
  }
}
