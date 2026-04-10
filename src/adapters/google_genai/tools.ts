import type { FunctionDeclaration } from "@google/genai";
import z from "zod";
import type { Tool } from "../../tool.ts";

export interface GoogleToolMap {
  original: Tool;
  google: FunctionDeclaration;
  /** Google silently doesn't allow non-objects at the top level but we want to. We therefore wrap the tool input with a wrapper object which need to unwrap at the output */
  wrapperObject: boolean;
  /** No parameter specified */
  isVoid: boolean;
}

export function normalizeGoogleTools(tools: Tool[]): GoogleToolMap[] {
  return tools.map((tool): GoogleToolMap => {
    const name = tool.normalizedName;

    if (tool.parameters instanceof z.ZodVoid) {
      return {
        original: tool,
        google: {
          name,
          description: tool.description,
          parametersJsonSchema: {
            type: "object",
            properties: {},
            additionalProperties: false,
          },
        },
        wrapperObject: false,
        isVoid: true,
      };
    }

    const wrapperObject = !(tool.parameters instanceof z.ZodObject);

    return {
      original: tool,
      google: {
        name,
        description: tool.description,
        parametersJsonSchema: z.toJSONSchema(wrapperObject ? z.object({ content: tool.parameters }) : tool.parameters),
      },
      wrapperObject,
      isVoid: false,
    };
  });
}
