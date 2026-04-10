import z from "zod";
import type { AnyTool } from "../../tool.ts";

interface FunctionTool {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters: Record<string, unknown>;
    strict: boolean;
  };
}

export interface OpenAICompletionsToolMap {
  original: AnyTool;
  openAI: FunctionTool;
  wrapperObject: boolean;
  isVoid: boolean;
}

export function normalizeOpenAICompletionsTools(tools: AnyTool[]): OpenAICompletionsToolMap[] {
  return tools.map((tool): OpenAICompletionsToolMap => {
    if (tool.parameters instanceof z.ZodVoid) {
      return {
        original: tool,
        openAI: {
          type: "function",
          function: {
            name: tool.normalizedName,
            description: tool.description,
            parameters: {
              type: "object",
              properties: {},
              additionalProperties: false,
            },
            strict: false,
          },
        },
        wrapperObject: false,
        isVoid: true,
      };
    }

    const wrapperObject = !(tool.parameters instanceof z.ZodObject);

    return {
      original: tool,
      openAI: {
        type: "function",
        function: {
          name: tool.normalizedName,
          description: tool.description,
          strict: true,
          parameters: z.toJSONSchema(wrapperObject ? z.object({ content: tool.parameters }) : tool.parameters),
        },
      },
      wrapperObject,
      isVoid: false,
    };
  });
}
