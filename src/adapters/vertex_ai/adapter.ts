import type { Adapter } from "@alphaxiv/agents";
import { requireEnv } from "../../util.ts";
import { googleGenerateContentAPIModel } from "../google_genai/adapter.ts";
import { getThinkingConfig, type GoogleModels, type SupportedThinkingLevel } from "../google_genai/models.ts";

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

export function vertexAIModel<zO, zI, TModel extends GoogleModels>(options: {
  model: TModel;
  thinkingLevel?: SupportedThinkingLevel<TModel>;
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
}): Adapter<zO, zI> {
  return googleGenerateContentAPIModel<zO, zI>({
    provider: "Vertex AI",
    googleGenAIOptions: {
      vertexai: true,
      project: options.project ?? requireEnv("GOOGLE_CLOUD_PROJECT"),
      location: options.location ?? requireEnv("GOOGLE_CLOUD_LOCATION"),
      httpOptions: { headers: getVertexAiPriorityHeaders(options.priority), baseUrl: options.baseUrl },
    },
    thinkingConfig: getThinkingConfig(options.model, options.thinkingLevel),
    model: options.model,
  });
}
