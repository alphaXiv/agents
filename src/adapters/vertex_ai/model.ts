import { GoogleGenAiModel, type GoogleGenAiModelOptions } from "../google_genai/model.ts";
import type { GoogleModels } from "../google_genai/models.ts";
import { VertexAiAdapter } from "./adapter.ts";

export type VertexAiModelPriority = "standard" | "priority" | "flex";

export interface VertexAiModelOptions<TModel extends GoogleModels> extends GoogleGenAiModelOptions<TModel> {
  baseUrl?: string;

  /**
   * The priority level for the model. This determines the availability and performance of the model.
   * Higher priority levels may have faster response times and higher availability, but may also have higher costs.
   *
   * @see "flex" https://docs.cloud.google.com/vertex-ai/generative-ai/docs/flex-paygo
   * @see "standard" https://docs.cloud.google.com/vertex-ai/generative-ai/docs/standard-paygo
   * @see "priority" https://docs.cloud.google.com/vertex-ai/generative-ai/docs/priority-paygo
   * @default "standard"
   */
  priority?: VertexAiModelPriority;

  /**
   * Vertex AI project ID
   * @default `GOOGLE_CLOUD_PROJECT` environment variable
   */
  project?: string;
  /**
   * Vertex AI location
   * @default `GOOGLE_CLOUD_LOCATION` environment variable
   */
  location?: string;
}

export class VertexAiModel<TModel extends GoogleModels> extends GoogleGenAiModel<TModel> {
  adapter: VertexAiAdapter<TModel>;

  constructor(options: VertexAiModelOptions<TModel>) {
    super(options);
    this.adapter = new VertexAiAdapter({
      model: options.model,
      priority: options.priority,
      project: options.project,
      location: options.location,
      baseUrl: options.baseUrl,
      thinkingConfig: this.thinkingConfig,
    });
  }
}
