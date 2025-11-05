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

export type AsyncStreamItemGenerator = AsyncGenerator<
  StreamItem,
  void,
  unknown
>;
