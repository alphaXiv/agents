import { requireEnv } from "../../util.ts";
import type { Adapter } from "../adapter.ts";
import { googleGenerateContentAPIModel } from "../google_genai/adapter.ts";
import { getThinkingConfig, type GoogleModels, type SupportedThinkingLevel } from "../google_genai/models.ts";

export function geminiModel<zO, zI, TModel extends GoogleModels>(options: {
  model: TModel;
  thinkingLevel?: SupportedThinkingLevel<TModel>;
  apiKey?: string;
  baseUrl?: string;
}): Adapter<zO, zI> {
  return googleGenerateContentAPIModel<zO, zI>({
    provider: "Gemini",
    googleGenAIOptions: {
      apiKey: options.apiKey ?? requireEnv("GEMINI_API_KEY"),
      httpOptions: {
        baseUrl: options.baseUrl,
        // @google/genai 3.x defaults to v1, which lacks some preview models and features.
        apiVersion: "v1beta",
      },
    },
    thinkingConfig: getThinkingConfig(options.model, options.thinkingLevel),
    model: options.model,
  });
}
