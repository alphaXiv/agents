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
  | {
    type: "tool_use";
    /** Provider generated string representing the id of the tool call */
    tool_use_id: string;
    /** The name of the function called */
    name: string;
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
