import OpenAI from "openai";
import { crossPlatformEnv, requireEnv } from "../../util.ts";
import { OpenResponsesAdapter, type OpenResponsesAdapterOptions } from "../open_responses/adapter.ts";
import { getOpenAISupportedMimeTypes } from "./mimes.ts";
import { getModelModalities, type OpenAIModels } from "./models.ts";

export interface OpenAIAdapterOptions<TModel extends OpenAIModels>
  extends Omit<OpenResponsesAdapterOptions<TModel>, "client" | "name" | "supportedMimeTypes"> {
  apiKey?: string;
  baseUrl?: string;
}

export class OpenAIAdapter<TModel extends OpenAIModels> extends OpenResponsesAdapter<TModel> {
  constructor(options: OpenAIAdapterOptions<TModel>) {
    super({
      ...options,
      name: "OpenAI",
      supportedMimeTypes: getOpenAISupportedMimeTypes(getModelModalities(options.model)),
      client: new OpenAI({
        apiKey: options.apiKey ?? requireEnv("OPENAI_API_KEY"),
        baseURL: options.baseUrl ?? crossPlatformEnv("OPENAI_BASE_URL") ?? "https://api.openai.com/v1",
      }),
    });
  }
}
