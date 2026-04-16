import type z from "zod";
import {
  createSchemaCompatibility,
  type ObjectSchemaCompatibility,
  type ObjectSchemaCompatibilityOptions,
  type SchemaCompatibility,
  type SchemaCompatibilityFeatures,
  type SchemaCompatibilityOptions,
} from "./schema_compatibility.ts";

export interface OpenAICompatibleSchemaOptions extends Omit<SchemaCompatibilityOptions, "features"> {
  features?: SchemaCompatibilityFeatures;
}

interface OpenAICompatibleObjectSchemaOptions extends Omit<ObjectSchemaCompatibilityOptions, "features"> {
  features?: SchemaCompatibilityFeatures;
}

export const openAISchemaCompatibilityFeatures: SchemaCompatibilityFeatures = {
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

export function createOpenAICompatibleSchema(
  schema: z.ZodType,
  options: OpenAICompatibleObjectSchemaOptions,
): ObjectSchemaCompatibility;
export function createOpenAICompatibleSchema(
  schema: z.ZodType,
  options: OpenAICompatibleSchemaOptions,
): SchemaCompatibility;

export function createOpenAICompatibleSchema(
  schema: z.ZodType,
  options: OpenAICompatibleSchemaOptions,
): SchemaCompatibility {
  return createSchemaCompatibility(schema, {
    ...options,
    features: options.features ?? openAISchemaCompatibilityFeatures,
  });
}
