import type { AgentRunResult } from "./agent.ts";

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
  type: "delta_output_text";
  delta: string;
} | {
  type: "delta_output_reasoning";
  delta: string;
} | {
  type: "tool_use_start";
  tool_use_id: string;
  kind: string;
} | {
  type: "tool_use";
  tool_use_id: string;
  kind: string;
  content?: string;
} | {
  type: "tool_result_text";
  tool_use_id: string;
  content: string;
} | {
  type: "tool_result_file";
  tool_use_id: string;
  kind: string;
  content: string;
};

export type ReasoningEffort = "minimal" | "normal"; // TODO: investigate adding "low" and "high" here

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
  /** Count of input tokens */
  inputTokens: number | null;
  /** Count of output tokens. If the provider is unable to classify, then all tokens are "output" tokens. */
  outputTokens: number | null;
}

export interface AdapterStreamSingleResult extends ProviderStreamMetadata {
  items: ChatItem[];
}
