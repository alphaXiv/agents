export type ChatItem =
  | {
    type: "input_text";
    text: string;
  }
  | {
    type: "output_reasoning";
    text: string;
  }
  | { type: "output_text"; text: string }
  | { type: "tool_use"; id: string; name: string; input: string }
  | { type: "tool_result"; tool_use_id: string; content: string };

export type ChatLike = string | ChatItem[];
