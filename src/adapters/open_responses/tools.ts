import type { FunctionTool } from "openai/resources/responses/responses";
import z from "zod";
import type { AnyTool } from "../../tool.ts";
import { createOpenAICompatibleSchema } from "../shared/openai_compatibility.ts";
import type { SchemaCompatibility } from "../shared/schema_compatibility.ts";

export interface OpenResponsesToolMap {
  original: AnyTool;
  openResponses: FunctionTool;
  compatibility?: SchemaCompatibility;
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
        compatibility: undefined,
        wrapperObject: false,
        isVoid: true,
      };
    }

    const wrapperObject = !(tool.parameters instanceof z.ZodObject);
    const compatibility = createOpenAICompatibleSchema(tool.parameters, {
      kind: "tool",
      requireTopLevelObject: true,
      rootPath: "input",
    });

    return {
      original: tool,
      openResponses: {
        type: "function",
        name: tool.normalizedName,
        description: compatibility.instructions
          ? `${tool.description}\n\n${compatibility.instructions}`
          : tool.description,
        strict: false,
        parameters: compatibility.jsonSchema,
      },
      compatibility,
      wrapperObject,
      isVoid: false,
    };
  });
}
