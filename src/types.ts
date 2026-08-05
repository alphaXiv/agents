import type { AgentRunResult } from "./agent.ts";
import type { ClassifiedError } from "./errors.ts";

/**
 * Token usage statistics for an agent run.
 * If the provider is unable to classify, then all tokens are "output" tokens.
 */
export interface TokenUsage {
  /**
   * Input tokens from the most recent model call. Providers that report prompt
   * caching exclude cached tokens here; providers that do not report it leave
   * their cached tokens counted in this number.
   */
  inputTokens: number;
  /** Output tokens from the most recent model call. */
  outputTokens: number;
  /** Input tokens read from the provider's prompt cache in the most recent model call. `0` if unreported. */
  cacheReadTokens: number;
  /**
   * Input tokens billed at the provider's cache write premium in the most recent
   * model call. `0` on providers whose caches populate for free, which is Gemini
   * and OpenAI before GPT-5.6.
   */
  cacheWriteTokens: number;

  /** Total input tokens used over the span of the agent run */
  totalInputTokens: number;
  /** Total output tokens used over the span of the agent run */
  totalOutputTokens: number;
  /** Total cached input tokens read over the span of the agent run */
  totalCacheReadTokens: number;
  /** Total input tokens written to cache over the span of the agent run */
  totalCacheWriteTokens: number;
}

export type ChatItemToolUse = {
  type: "tool_use";
  /** Provider generated string representing the id of the tool call */
  tool_use_id: string;
  /** The name of the function called */
  kind: string;
  /** The input parameters into the function, encoded as json string or nothing if void input  */
  content?: string;
};

export type ChatItemToolResultText = {
  type: "tool_result_text";
  /** Id of a previous tool call in this conversation. CANNOT BE INCLUDED UNLESS THE TOOL USE IS ALSO INCLUDED!! */
  tool_use_id: string;
  /** The result from the tool call */
  content: string;
};

export type ChatItemToolResultFile = {
  type: "tool_result_file";
  /** Id of a previous tool call in this conversation. CANNOT BE INCLUDED UNLESS THE TOOL USE IS ALSO INCLUDED!! */
  tool_use_id: string;
  /** Mime type of the file */
  kind: string;
  /** The resulting file URL from the tool call */
  content: string;
};

export type ChatItemInputFile = {
  type: "input_file";
  /** Mime type of the file */
  kind: string;
  /** File URL for the model */
  content: string;
};

export type ChatItemContextSummary = {
  type: "context_summary";
  /** Summary of prior conversation */
  content: string;
};

export type ContextSummaryStartEvent = { type: "context_summary_start" };

export type ChatItemToolResult =
  | ChatItemToolResultText
  | ChatItemToolResultFile;

/** ChatItem is designed to be stored in a database, this is why some names are suboptimal, we're trying to overlap as much as possible */
export type ChatItem =
  | {
    type: "input_text";
    /** Text input for the model */
    content: string;
  }
  | {
    type: "output_reasoning";
    /** Textual reasoning output from the model */
    content: string;
  }
  | {
    type: "output_text";
    /** Text output from the model */
    content: string;
  }
  | ChatItemInputFile
  | ChatItemContextSummary
  | ChatItemToolUse
  | ChatItemToolResultText
  | ChatItemToolResultFile;

/**
 * Agent runs return ChatItem and StreamItem items decorated with a tracing ID.
 * Traces are not needed as input anywhere, and you can ignore them if you don't
 * require it.
 */
export type WithTraceId<T> = T & {
  /**
   * ID of the {@linkcode TraceEvent} that created this item. For tools, this
   * points to the tool's trace id, and not the message trace for generating the
   * tool use.
   */
  trace: string;
};

export interface ModelInfo {
  provider: string;
  model: string;
}

export type ToolResultLike = string | ({
  type: "tool_result_file";
  kind: string;
  content: string;
} | {
  type: "tool_result_text";
  content: string;
})[];
export type ChatLike = string | ChatItem[];

type BaseStreamItem = {
  index: number;
};

type StreamItemType = {
  /** Content added to `output_text` */
  type: "delta_output_text";
  delta: string;
} | {
  /** Content added to `output_reasoning` */
  type: "delta_output_reasoning";
  delta: string;
} | {
  /** A tool use is being formed, metadata is available. */
  type: "tool_use_start";
  tool_use_id: string;
  kind: string;
} | {
  /** A tool use completed. Does not require `tool_use_start` */
  type: "tool_use";
  tool_use_id: string;
  kind: string;
  content?: string;
} | {
  /** The result of a tool. Implies a `tool_use` earlier in the history. */
  type: "tool_result_text";
  tool_use_id: string;
  content: string;
} | {
  /** The result of a tool. Implies a `tool_use` earlier in the chat. */
  type: "tool_result_file";
  tool_use_id: string;
  kind: string;
  content: string;
} | {
  /**
   * Indicates `output_reasoning` has started. Due to redacted reasoning on some
   * providers, this may not actually deliver the reasoning content. This event
   * is not required to receive `delta_output_reasoning`.
   */
  type: "reasoning_start";
} | {
  /** Compaction is running. Repeats while it reports progress, and may produce no summary. */
  type: "context_summary_start";
} | {
  /** A newly compacted summary. Only history from the most recent one is sent to the model. */
  type: "context_summary";
  content: string;
} | {
  /** Early hint to indicate token usage before a fully complete call. */
  type: "token_usage";
  usage: TokenUsage;
} | {
  /** The model switched to a fallback due to an error. */
  type: "model_switched";
  from: ModelInfo;
  to: ModelInfo;
  cause?: unknown;
  classified?: ClassifiedError;
};

export type StreamItem = BaseStreamItem & StreamItemType;

export type AdapterStreamIterator = AsyncGenerator<
  StreamItem,
  ProviderStreamMetadata,
  unknown
>;

export type AgentStreamIterator<T = unknown> = AsyncGenerator<
  WithTraceId<StreamItem>,
  AgentRunResult<T>,
  unknown
>;

export type Awaitable<T> = T | Promise<T>;

export interface ProviderStreamMetadata {
  /**
   * Count of input tokens billed at the full rate. Providers that report prompt
   * caching exclude cached tokens here, so the prompt size is
   * `inputTokens + cacheReadTokens + cacheWriteTokens`.
   */
  inputTokens: number | null;
  /** Count of output tokens. If the provider is unable to classify, then all tokens are "output" tokens. */
  outputTokens: number | null;
  /** Input tokens served from the provider's prompt cache, or `null` if the provider does not report caching. */
  cacheReadTokens?: number | null;
  /**
   * Input tokens billed at the provider's cache write premium, or `null` if the
   * provider does not report caching. Providers whose caches populate for free
   * have no write to bill and report `0`.
   */
  cacheWriteTokens?: number | null;
}
