// THIS FILE IS NECESSARY DUE TO `no-slow-types` BEING ENFORCED BY JSR.
// The models.ts contains actual inferable types, so if you update the types there you can just hover over the variables
// and copy the full types here to make jsr happy. I don't like this hack but I like strong types.
import type { z } from "zod";

type OpenAiModelModalityValue = "text" | "image" | "audio" | "video";

export interface NonReasoningModelSupport {
  type: "unsupported";
  modalities: readonly OpenAiModelModalityValue[];
}

export interface ReasoningModelSupport<T extends readonly [string, ...string[]]> {
  type: "reasoning";
  schema: z.ZodType<T[number]>;
  modalities: readonly OpenAiModelModalityValue[];
}

export interface OpenAiModelsMap {
  "gpt-5.5": ReasoningModelSupport<["none", "low", "medium", "high", "xhigh"]>;
  "gpt-5.5-pro": ReasoningModelSupport<["medium", "high", "xhigh"]>;
  "gpt-5.4": ReasoningModelSupport<["none", "low", "medium", "high", "xhigh"]>;
  "gpt-5.4-pro": ReasoningModelSupport<["medium", "high", "xhigh"]>;
  "gpt-5.4-mini": ReasoningModelSupport<["none", "low", "medium", "high"]>;
  "gpt-5.4-nano": ReasoningModelSupport<["none", "low", "medium", "high"]>;
  "gpt-5.3-codex": ReasoningModelSupport<["low", "medium", "high", "xhigh"]>;
  "gpt-5.2": ReasoningModelSupport<["none", "low", "medium", "high", "xhigh"]>;
  "gpt-5.2-codex": ReasoningModelSupport<["low", "medium", "high", "xhigh"]>;
  "gpt-5.2-pro": ReasoningModelSupport<["medium", "high", "xhigh"]>;
  "gpt-5.1": ReasoningModelSupport<["none", "low", "medium", "high"]>;
  "gpt-5.1-codex": ReasoningModelSupport<["none", "low", "medium", "high"]>;
  "gpt-5.1-codex-max": ReasoningModelSupport<["none", "low", "medium", "high"]>;
  "gpt-5.1-codex-mini": ReasoningModelSupport<["none", "low", "medium", "high"]>;
  "gpt-5": ReasoningModelSupport<["minimal", "low", "medium", "high"]>;
  "gpt-5-pro": ReasoningModelSupport<["high"]>;
  "gpt-5-mini": ReasoningModelSupport<["minimal", "low", "medium", "high"]>;
  "gpt-5-nano": ReasoningModelSupport<["minimal", "low", "medium", "high"]>;
  "gpt-5-codex": ReasoningModelSupport<["minimal", "low", "medium", "high"]>;
  "gpt-4.1": NonReasoningModelSupport;
  "gpt-4.1-mini": NonReasoningModelSupport;
  "gpt-4.1-nano": NonReasoningModelSupport;
  "gpt-oss-120b": ReasoningModelSupport<["low", "medium", "high"]>;
  "gpt-oss-20b": ReasoningModelSupport<["low", "medium", "high"]>;
  "o4-mini-deep-research": ReasoningModelSupport<["low", "medium", "high"]>;
  "o4-mini": ReasoningModelSupport<["low", "medium", "high"]>;
  "o1": ReasoningModelSupport<["low", "medium", "high"]>;
  "gpt-4o": NonReasoningModelSupport;
  "gpt-4o-mini": NonReasoningModelSupport;
  "gpt-4-turbo": NonReasoningModelSupport;
  "gpt-4": NonReasoningModelSupport;
}
