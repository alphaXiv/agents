/** ChatItem is designed to be stored in a database, this is why some names are suboptimal, we're trying to overlap as much as possible */
export type ChatItem =
  | {
    type: "input_text";
    /** Text input for the model */
    content: string;
  }
  | {
    type: "input_file";
    /** Mime type of the file */
    kind: string;
    /** File URL for the model */
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
  | {
    type: "tool_use";
    /** Provider generated string representing the id of the tool call */
    tool_use_id: string;
    /** The name of the function called */
    kind: string;
    /** The input parameters into the function, encoded as json string  */
    content: string;
  }
  | {
    type: "tool_result";
    /** Id of a previous tool call in this conversation. CANNOT BE INCLUDED UNLESS THE TOOL USE IS ALSO INCLUDED!! */
    tool_use_id: string;
    /** The result from the tool call */
    content: string;
  };

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
  content: string;
} | {
  type: "tool_result";
  tool_use_id: string;
  content: string;
};

export type StreamItem = BaseStreamItem & StreamItemType;

export type AsyncStreamItemGenerator = AsyncGenerator<
  StreamItem,
  void,
  unknown
>;
