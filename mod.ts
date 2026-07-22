export { Agent } from "./src/agent.ts";
export type {
  AgentOptions,
  AgentRunOptions,
  AgentRunResult,
  BeforeModelCall as BeforeModelCallFn,
  HandleModelError as HandleModelErrorFn,
  ModelCallReason,
} from "./src/agent.ts";

export { classifyError, createClassifiedError, ERROR_KINDS, FirstTokenTimeoutError } from "./src/errors.ts";
export type { ClassifiedError, ErrorKind } from "./src/errors.ts";

export { DEFAULT_RETRY_STRATEGY, determineRetryBehavior, resolveRetryStrategy } from "./src/retry.ts";
export type { ResolvedRetryStrategy, RetryBehavior, RetryStrategy } from "./src/retry.ts";

export { cli } from "./src/cli.ts";
export { addStreamItem, convertChatItemsToStream } from "./src/client.ts";

export type { CliIo, CliOptions } from "./src/cli.ts";
export { ModelOutput, Tool } from "./src/tool.ts";
export type { AnyTool, ExecuteContext, ExecuteFunc, ExecuteFuncInput, ExecuteResult } from "./src/tool.ts";

export type { Adapter } from "./src/adapters/adapter.ts";
export type { AdapterOptions, AdapterStreamOptions } from "./src/adapters/adapter.ts";

export { resolveModel } from "./src/adapters/model_resolver.ts";
export type { AdapterLike, ModelString } from "./src/adapters/model_resolver.ts";

export type {
  AnthropicModels,
  EffortLevel,
  SupportedEffortLevel as AnthropicSupportedEffortLevel,
  SupportedThinkingLevel as AnthropicSupportedThinkingLevel,
  SupportsInterleaved,
  ThinkingDisplay as AnthropicThinkingDisplay,
  ThinkingLevel,
} from "./src/adapters/anthropic/models.ts";

export type {
  OpenAIModelModality,
  OpenAIModels,
  OpenAIReasoningEffort,
  SupportedReasoningEffort,
} from "./src/adapters/openai/models.ts";

export type { OpenRouterModels, OpenRouterReasoningEffort } from "./src/adapters/openrouter/models.ts";

export type { SidModels } from "./src/adapters/sid/adapter.ts";

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
