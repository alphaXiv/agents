export { Agent } from "./src/agent.ts";
export type {
  AgentOptions,
  AgentRunOptions,
  AgentRunResult,
  BeforeModelCall as BeforeModelCallFn,
  HandleModelError as HandleModelErrorFn,
} from "./src/agent.ts";

export { classifyError, createClassifiedError, ERROR_KINDS } from "./src/errors.ts";
export type { ClassifiedError, ErrorKind } from "./src/errors.ts";

export { DEFAULT_RETRY_STRATEGY, determineRetryBehavior, resolveRetryStrategy } from "./src/retry.ts";
export type { ResolvedRetryStrategy, RetryBehavior, RetryStrategy } from "./src/retry.ts";

export { cli } from "./src/cli.ts";
export { addStreamItem, convertChatItemsToStream } from "./src/client.ts";

export type { CliIo, CliOptions } from "./src/cli.ts";
export { ModelOutput, Tool } from "./src/tool.ts";
export type { AnyTool, ExecuteContext, ExecuteFunc, ExecuteFuncInput, ExecuteResult } from "./src/tool.ts";

export { Adapter } from "./src/adapters/adapter.ts";
export type { AdapterOptions, AdapterStreamOptions } from "./src/adapters/adapter.ts";

export { Model } from "./src/adapters/model.ts";
export type { ModelOptions } from "./src/adapters/model.ts";

export { resolveModel } from "./src/adapters/model_resolver.ts";
export type { ModelLike, ModelString } from "./src/adapters/model_resolver.ts";

export { AnthropicModel } from "./src/adapters/anthropic/model.ts";
export type { AnthropicModelOptions } from "./src/adapters/anthropic/model.ts";
export type {
  AnthropicModels,
  EffortLevel,
  SupportedEffortLevel as AnthropicSupportedEffortLevel,
  SupportedThinkingLevel as AnthropicSupportedThinkingLevel,
  SupportsInterleaved,
  ThinkingDisplay as AnthropicThinkingDisplay,
  ThinkingLevel,
} from "./src/adapters/anthropic/models.ts";

export { GeminiModel } from "./src/adapters/gemini/model.ts";
export type { GeminiModelOptions } from "./src/adapters/gemini/model.ts";

export { OpenAIModel } from "./src/adapters/openai/model.ts";
export type { OpenAIModelOptions } from "./src/adapters/openai/model.ts";
export type {
  OpenAIModelModality,
  OpenAIModels,
  OpenAIReasoningEffort,
  SupportedReasoningEffort,
} from "./src/adapters/openai/models.ts";

export { OpenRouterModel } from "./src/adapters/openrouter/model.ts";
export type { OpenRouterModelOptions } from "./src/adapters/openrouter/model.ts";
export type { OpenRouterModels, OpenRouterReasoningEffort } from "./src/adapters/openrouter/models.ts";

export type { SidModels } from "./src/adapters/sid/adapter.ts";
export { SidModel } from "./src/adapters/sid/model.ts";
export type { SidModelOptions } from "./src/adapters/sid/model.ts";

export { TributaryModel } from "./src/adapters/tributary/model.ts";
export type { TributaryModelOptions } from "./src/adapters/tributary/model.ts";
export type { TributaryModels } from "./src/adapters/tributary/models.ts";

export { VertexAiModel } from "./src/adapters/vertex_ai/model.ts";
export type { VertexAiModelOptions, VertexAiModelPriority } from "./src/adapters/vertex_ai/model.ts";

export { SidEmbeddingSearchTool, SidReportHelpfulIdsTool, SidTextSearchTool } from "./src/adapters/sid/tools.ts";

export { MessageTracer, newTrace, registerGlobalTracer, tracerAsyncLocalStorage, withTrace } from "./src/tracing.ts";
export type {
  ActiveCustomTrace,
  ActiveTrace,
  AgentTraceEvent,
  BaseTraceEvent,
  CustomTraceEvent,
  LogTraceEvent,
  MessageTraceEvent,
  ModelTraceEvent,
  PartialTraceEvent,
  ToolTraceEvent,
  TraceContent,
  TraceEvent,
  Tracer,
  TraceRef,
  TraceType,
} from "./src/tracing.ts";

export type {
  AdapterStreamIterator,
  AgentStreamIterator,
  Awaitable,
  ChatItem,
  ChatItemContextSummary,
  ChatItemInputFile,
  ChatItemToolResult,
  ChatItemToolResultFile,
  ChatItemToolResultText,
  ChatItemToolUse,
  ChatLike,
  ContextSummaryStartEvent,
  ProviderStreamMetadata,
  StreamItem,
  TokenUsage,
  ToolResultLike,
  WithTraceId,
} from "./src/types.ts";
