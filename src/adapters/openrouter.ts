import { crossPlatformEnv, requireEnv } from "../util.ts";
import { openAiCompletionsAdapter } from "./openai-completions.ts";

// TODO: keep this updated (pulled from https://openrouter.ai/models?fmt=cards&input_modalities=file)
const nativePdfSupport = [
  "openai/gpt-5-image-mini",
  "openai/gpt-5-image",
  "openai/o3-deep-research",
  "openai/o4-mini-deep-research",
  "openai/gpt-5-pro",
  "anthropic/claude-sonnet-4.5",
  "google/gemini-2.5-flash-preview-09-2025",
  "google/gemini-2.5-flash-lite-preview-09-2025",
  "openai/gpt-5-chat",
  "openai/gpt-5",
  "openai/gpt-5-mini",
  "openai/gpt-5-nano",
  "anthropic/claude-opus-4.1",
  "google/gemini-2.5-flash-lite",
  "google/gemini-2.5-flash-lite-preview-06-17",
  "google/gemini-2.5-flash",
  "google/gemini-2.5-pro",
  "openai/o3-pro",
  "google/gemini-2.5-pro-preview",
  "anthropic/claude-opus-4",
  "anthropic/claude-sonnet-4",
  "google/gemini-2.5-pro-preview-05-06",
  "openai/o4-mini-high",
  "openai/o3",
  "openai/o4-mini",
  "openai/gpt-4.1",
  "openai/gpt-4.1-mini",
  "openai/gpt-4.1-nano",
  "openai/o1-pro",
  "google/gemini-2.0-flash-lite-001",
  "anthropic/claude-3.7-sonnet",
  "anthropic/claude-3.7-sonnet",
  "openai/o3-mini-high",
  "google/gemini-2.0-flash-001",
  "openai/o3-mini",
  "openai/o1",
  "openai/gpt-4o-2024-11-20",
  "anthropic/claude-3.5-haiku-20241022",
  "anthropic/claude-3.5-sonnet",
  "openai/gpt-4o-2024-08-06",
  "openai/gpt-4o-mini",
  "openai/gpt-4o-mini-2024-07-18",
  "anthropic/claude-3.5-sonnet-20240620",
  "openai/gpt-4o",
  "openai/gpt-4o",
  "openai/gpt-4o-2024-05-13",
  "google/gemini-2.5-flash-preview-05-20",
  "google/gemini-2.5-flash-preview",
  "google/gemini-2.5-pro-exp-03-25",
];

// TODO: ensure this list is complete
const alwaysReasoningModels = [
  "openai/gpt-oss-20b",
  "openai/gpt-oss-120b",
  "x-ai/grok-4",
];

export function getOpenRouterAdapter() {
  return openAiCompletionsAdapter({
    name: "openrouter",

    url: crossPlatformEnv("OPENROUTER_BASE_URL") ??
      "https://openrouter.ai/api/v1",
    apiKey: requireEnv("OPENROUTER_API_KEY"),

    pdfSupport: (model) => ({
      mode: nativePdfSupport.includes(model) ? "native" : "text",
      // Openrouter seems to have an undocumented 5MB size limit on pdfs :) - 4 to be safe here
      maxSize: 4 * 1024 * 1024,
    }),
    extraRequestBody: ({ model, reasoningEffort }) => ({
      reasoning: alwaysReasoningModels.includes(model) ? undefined : {
        enabled: reasoningEffort === "normal",
      },
      plugins: nativePdfSupport.includes(model) ? undefined : [
        {
          id: "file-parser",
          pdf: {
            engine: "pdf-text",
          },
        },
      ],
    }),
  });
}
