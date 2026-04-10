import type { Tool } from "openai/resources/responses/responses";
import z from "zod";
import type { AnyTool } from "../../tool.ts";

export interface OpenResponsesToolMap {
  original: AnyTool;
  openResponses: Tool;
  /** Open Responses function tools expect object-shaped params, so non-objects are wrapped under `content`. */
  wrapperObject: boolean;
  /** No parameter specified. */
  isVoid: boolean;
}

export function normalizeOpenResponsesTools(tools: AnyTool[]): OpenResponsesToolMap[] {
  return tools.map((tool): OpenResponsesToolMap => {
    if (tool.parameters instanceof z.ZodVoid) {
      return {
        original: tool,
        openResponses: {
          type: "function",
          name: tool.normalizedName,
          description: tool.description,
          parameters: z.object({}).toJSONSchema(),
          strict: false,
        },
        wrapperObject: false,
        isVoid: true,
      };
    }

    const wrapperObject = !(tool.parameters instanceof z.ZodObject);

    return {
      original: tool,
      openResponses: {
        type: "function",
        name: tool.normalizedName,
        description: tool.description,
        strict: false,
        parameters: z.toJSONSchema(wrapperObject ? z.object({ content: tool.parameters }) : tool.parameters),
      },
      wrapperObject,
      isVoid: false,
    };
  });
}
