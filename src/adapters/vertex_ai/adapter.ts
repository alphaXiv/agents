import { GoogleGenAI } from "@google/genai";
import { requireEnv } from "../../util.ts";
import { GoogleGenAiAdapter, type GoogleGenAiAdapterOptions } from "../google_genai/adapter.ts";
import type { GoogleModels } from "../google_genai/models.ts";

export type VertexAiModelPriority = "standard" | "priority" | "flex";

function getVertexAiPriorityHeaders(priority?: VertexAiModelPriority) {
  switch (priority) {
    case "flex":
    case "priority":
      return {
        "X-Vertex-AI-LLM-Shared-Request-Type": priority,
      };
    default:
      return undefined;
  }
}

export interface VertexAiAdapterOptions<TModel extends GoogleModels>
  extends Omit<GoogleGenAiAdapterOptions<TModel>, "client"> {
  model: TModel;

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

export class VertexAiAdapter<TModel extends GoogleModels> extends GoogleGenAiAdapter<TModel> {
  name = "Vertex AI";

  constructor(options: VertexAiAdapterOptions<TModel>) {
    const project = options.project ?? requireEnv("GOOGLE_CLOUD_PROJECT");
    const location = options.location ?? requireEnv("GOOGLE_CLOUD_LOCATION");

    super({
      client: new GoogleGenAI({
        vertexai: true,
        project,
        location,
        httpOptions: { headers: getVertexAiPriorityHeaders(options.priority) },
      }),
      model: options.model,
      thinkingConfig: options.thinkingConfig,
    });
  }
}
