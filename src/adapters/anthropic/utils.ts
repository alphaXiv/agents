import type { Tool as AnthropicTool } from "@anthropic-ai/sdk/resources/messages/messages";
import z from "zod";
import type { AnyTool, Tool } from "../../tool.ts";
import {
  createSchemaCompatibility,
  type ObjectSchemaCompatibility,
  type ObjectSchemaCompatibilityOptions,
  type SchemaCompatibility,
  type SchemaCompatibilityFeatures,
  type SchemaCompatibilityOptions,
} from "../shared/schema_compatibility.ts";

export type AnthropicToolMap = {
  original: Tool;
  anthropic: AnthropicTool;
  compatibility?: SchemaCompatibility;
  isVoid: boolean;
};

export interface AnthropicCompatibleSchemaOptions extends Omit<SchemaCompatibilityOptions, "features"> {
  features?: SchemaCompatibilityFeatures;
}

interface AnthropicCompatibleObjectSchemaOptions extends Omit<ObjectSchemaCompatibilityOptions, "features"> {
  features?: SchemaCompatibilityFeatures;
}

export const anthropicSchemaCompatibilityFeatures: SchemaCompatibilityFeatures = {
  tuples: {
    strategy: "object-surrogate",
    itemKeyPrefix: "item",
    restKey: "rest",
  },
  records: {
    finite: "explicit-object",
    dynamic: "entries-array",
    keyPropertyName: "key",
    valuePropertyName: "value",
  },
  strings: {
    length: "instructions",
  },
  numbers: {
    integerType: "number",
    bounds: "instructions",
    multipleOf: "instructions",
  },
  arrays: {
    length: "min1-only",
  },
};

export function normalizeAnthropicTools(tools: AnyTool[]): AnthropicToolMap[] {
  return tools.map((tool): AnthropicToolMap => {
    const name = tool.normalizedName;

    if (tool.parameters instanceof z.ZodVoid) {
      return {
        original: tool,
        anthropic: {
          name,
          strict: false,
          eager_input_streaming: true,
          input_schema: { type: "object" },
          description: tool.description,
        },
        isVoid: true,
      };
    }

    const compatibleSchema = createAnthropicCompatibleSchema(tool.parameters, {
      kind: "tool",
      requireTopLevelObject: true,
      rootPath: "input",
    });

    return {
      original: tool,
      anthropic: {
        name,
        strict: true,
        eager_input_streaming: true,
        input_schema: compatibleSchema.jsonSchema,
        description: compatibleSchema.instructions
          ? `${tool.description}\n\n${compatibleSchema.instructions}`
          : tool.description,
      },
      compatibility: compatibleSchema,
      isVoid: false,
    };
  });
}

export function createAnthropicCompatibleSchema(
  schema: z.ZodType,
  options: AnthropicCompatibleObjectSchemaOptions,
): ObjectSchemaCompatibility;
export function createAnthropicCompatibleSchema(
  schema: z.ZodType,
  options: AnthropicCompatibleSchemaOptions,
): SchemaCompatibility;
export function createAnthropicCompatibleSchema(
  schema: z.ZodType,
  options: AnthropicCompatibleSchemaOptions,
): SchemaCompatibility {
  return createSchemaCompatibility(schema, {
    ...options,
    features: options.features ?? anthropicSchemaCompatibilityFeatures,
  });
}
