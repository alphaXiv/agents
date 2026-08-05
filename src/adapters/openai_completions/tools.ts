import z from "zod";
import type { AnyTool } from "../../tool.ts";
import { createOpenAICompatibleSchema } from "../shared/openai_compatibility.ts";
import type { SchemaCompatibility } from "../shared/schema_compatibility.ts";

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
  compatibility?: SchemaCompatibility;
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
      openAI: {
        type: "function",
        function: {
          name: tool.normalizedName,
          description: compatibility.instructions
            ? `${tool.description}\n\n${compatibility.instructions}`
            : tool.description,
          strict: false,
          parameters: compatibility.jsonSchema,
        },
      },
      compatibility,
      wrapperObject,
      isVoid: false,
    };
  });
}
