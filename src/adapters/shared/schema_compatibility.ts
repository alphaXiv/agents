import z from "zod";
import type { JSONSchema } from "zod/v4/core";

export type ZodJsonSchema = JSONSchema.BaseSchema;
export type ZodJsonSchemaInput = JSONSchema._JSONSchema;
type JsonValue = NonNullable<ZodJsonSchema["const"]>;

export interface SchemaCompatibility {
  /** The original JSON Schema produced directly from the Zod schema. */
  originalJsonSchema: ZodJsonSchema;
  /** Provider-compatible JSON Schema after compatibility rewriting. */
  jsonSchema: ZodJsonSchema;
  /** Prompt instructions that preserve constraints moved out of the JSON Schema. */
  instructions: string;
  /** Whether values need shape conversion before sending to, or after receiving from, the provider. */
  requiresValueTransformation: boolean;
  /** Converts a value from the original application-facing shape into the provider-facing shape described by `jsonSchema`. */
  toProvider(value: unknown): unknown;
  /** Converts a provider-produced value back into the original application-facingshape before the original Zod schema validates it. */
  fromProvider(value: unknown): unknown;
}

export interface ObjectSchemaCompatibility extends SchemaCompatibility {
  jsonSchema: JSONSchema.ObjectSchema;
}

/**
 * Describes which schema features a provider can represent directly and which
 * ones need to be moved into prompt instructions or surrogate shapes.
 */
export interface SchemaCompatibilityFeatures {
  tuples: {
    strategy: "native" | "object-surrogate";
    itemKeyPrefix?: string;
    restKey?: string;
  };
  records: {
    finite: "native" | "explicit-object";
    dynamic: "native" | "entries-array";
    keyPropertyName?: string;
    valuePropertyName?: string;
  };
  strings: {
    length: "native" | "instructions";
  };
  numbers: {
    integerType: "native" | "number";
    bounds: "native" | "instructions";
    multipleOf: "native" | "instructions";
  };
  arrays: {
    length: "native" | "min1-only" | "instructions";
  };
}

export interface SchemaCompatibilityOptions {
  kind: "output" | "tool";
  requireTopLevelObject?: boolean;
  rootPath: string;
  features: SchemaCompatibilityFeatures;
}

export interface ObjectSchemaCompatibilityOptions extends SchemaCompatibilityOptions {
  requireTopLevelObject: true;
}

interface CompatibilityNode {
  jsonSchema: ZodJsonSchema;
  toProvider(value: unknown): unknown;
  fromProvider(value: unknown): unknown;
}

interface ObjectCompatibilityNode extends CompatibilityNode {
  jsonSchema: JSONSchema.ObjectSchema;
}

interface CompatibilityContext {
  constraints: string[];
  usedRecord: boolean;
  usedDynamicRecordFallback: boolean;
  usedTupleSurrogate: boolean;
  features: SchemaCompatibilityFeatures;
}

/**
 * Converts a Zod schema into a provider-compatible contract while keeping the
 * rest of the application working with the original logical shape.
 *
 * Why this exists:
 * Some providers only accept a restricted JSON Schema subset.
 * Useful shapes like tuples, dynamic records, non-object top-level values, and length or
 * numeric constraints may need to be rewritten into a simpler schema plus a small amount of prompt guidance.
 *
 * What this returns:
 * - `jsonSchema`: the provider-facing schema to send upstream
 * - `instructions`: prompt text for semantics that no longer fit in schema form
 * - `toProvider`: converts original values into the provider-facing schema shape
 * - `fromProvider`: restores provider values back into the original application shape
 */
export function createSchemaCompatibility(
  schema: z.ZodType,
  options: ObjectSchemaCompatibilityOptions,
): ObjectSchemaCompatibility;
export function createSchemaCompatibility(schema: z.ZodType, options: SchemaCompatibilityOptions): SchemaCompatibility;
export function createSchemaCompatibility(schema: z.ZodType, options: SchemaCompatibilityOptions): SchemaCompatibility {
  const originalJsonSchema = z.toJSONSchema(schema);
  const context = createCompatibilityContext(options.features);

  const root = transformSchemaNode(originalJsonSchema, options.rootPath, context);
  const compatibility = shouldWrapTopLevelObject(options.requireTopLevelObject, root.jsonSchema)
    ? wrapTopLevel(root)
    : root;
  const requiresValueTransformation = context.usedTupleSurrogate || context.usedDynamicRecordFallback ||
    compatibility !== root;

  return {
    originalJsonSchema,
    jsonSchema: compatibility.jsonSchema,
    instructions: buildInstructions(context, options.kind),
    requiresValueTransformation,
    toProvider: compatibility.toProvider,
    fromProvider: compatibility.fromProvider,
  };
}

function transformSchemaNode(
  schema: ZodJsonSchemaInput | undefined,
  path: string,
  context: CompatibilityContext,
): CompatibilityNode {
  const normalizedSchema = isJsonSchema(schema) ? schema : {};

  if (Array.isArray(normalizedSchema.anyOf) && normalizedSchema.anyOf.length > 0) {
    return transformAnyOfSchema(normalizedSchema, path, context);
  }

  if (Array.isArray(normalizedSchema.prefixItems) && normalizedSchema.prefixItems.length > 0) {
    return context.features.tuples.strategy === "native"
      ? transformNativeTupleSchema(normalizedSchema, path, context)
      : transformTupleSchema(normalizedSchema, path, context);
  }

  if (isRecordSchema(normalizedSchema)) {
    return transformRecordSchema(normalizedSchema, path, context);
  }

  switch (getPrimaryType(normalizedSchema)) {
    case "object":
      return transformObjectSchema(normalizedSchema, path, context);
    case "array":
      return transformArraySchema(normalizedSchema, path, context);
    case "integer":
    case "number":
      return transformNumberSchema(normalizedSchema, path, context);
    case "string":
      return transformStringSchema(normalizedSchema, path, context);
    default:
      return identityNode(normalizedSchema, context);
  }
}

function transformAnyOfSchema(schema: ZodJsonSchema, path: string, context: CompatibilityContext): CompatibilityNode {
  const options = schema.anyOf ?? [];
  const nodes = options.map((option, index) => transformSchemaNode(option, `${path}[option ${index}]`, context));

  return {
    jsonSchema: {
      ...preserveSchemaMetadata(schema),
      anyOf: nodes.map((node) => node.jsonSchema),
    },
    toProvider(value) {
      const index = options.findIndex((option) => matchesSchema(value, option, context));
      return index === -1 ? value : nodes[index].toProvider(value);
    },
    fromProvider(value) {
      const index = nodes.findIndex((node) => matchesSchema(value, node.jsonSchema, context));
      return index === -1 ? value : nodes[index].fromProvider(value);
    },
  };
}

function transformObjectSchema(schema: ZodJsonSchema, path: string, context: CompatibilityContext): CompatibilityNode {
  const propertyEntries = Object.entries(schema.properties ?? {});
  const nodes = Object.fromEntries(
    propertyEntries.map(([key, value]) => [key, transformSchemaNode(value, `${path}.${key}`, context)]),
  );

  return {
    jsonSchema: {
      ...preserveSchemaMetadata(schema),
      type: "object",
      properties: Object.fromEntries(Object.entries(nodes).map(([key, node]) => [key, node.jsonSchema])),
      required: schema.required,
      additionalProperties: false,
    },
    toProvider(value) {
      if (!isPlainObject(value)) return value;
      const result: Record<string, unknown> = {};
      for (const [key, node] of Object.entries(nodes)) {
        if (key in value) result[key] = node.toProvider(value[key]);
      }
      return result;
    },
    fromProvider(value) {
      if (!isPlainObject(value)) return value;
      const result: Record<string, unknown> = {};
      for (const [key, node] of Object.entries(nodes)) {
        if (key in value) result[key] = node.fromProvider(value[key]);
      }
      return result;
    },
  };
}

function transformArraySchema(schema: ZodJsonSchema, path: string, context: CompatibilityContext): CompatibilityNode {
  const itemNode = transformSchemaNode(Array.isArray(schema.items) ? undefined : schema.items, `${path}[]`, context);
  const constraints: string[] = [];
  const minItems = typeof schema.minItems === "number" ? schema.minItems : undefined;
  const maxItems = typeof schema.maxItems === "number" ? schema.maxItems : undefined;
  const { length } = context.features.arrays;

  if (length !== "native") {
    if (minItems !== undefined && maxItems !== undefined && minItems === maxItems) {
      constraints.push(`exactly ${minItems} item${minItems === 1 ? "" : "s"}`);
    } else {
      if (minItems !== undefined && (length === "instructions" || minItems > 1)) {
        constraints.push(`at least ${minItems} item${minItems === 1 ? "" : "s"}`);
      }
      if (maxItems !== undefined) constraints.push(`at most ${maxItems} item${maxItems === 1 ? "" : "s"}`);
    }
  }

  if (constraints.length > 0) {
    context.constraints.push(`- \`${path}\` must contain ${constraints.join(" and ")}`);
  }

  const transformed = {
    ...omitSchemaKeywords(schema, getArrayKeywordsToOmit(length, minItems)),
    items: itemNode.jsonSchema,
  };

  return {
    jsonSchema: transformed,
    toProvider(value) {
      return Array.isArray(value) ? value.map((item) => itemNode.toProvider(item)) : value;
    },
    fromProvider(value) {
      return Array.isArray(value) ? value.map((item) => itemNode.fromProvider(item)) : value;
    },
  };
}

function transformNumberSchema(schema: ZodJsonSchema, path: string, context: CompatibilityContext): CompatibilityNode {
  const constraints: string[] = [];
  const { integerType, bounds, multipleOf } = context.features.numbers;
  const hasRedundantSafeIntegerBounds = isRedundantSafeIntegerBounds(schema);

  if (getPrimaryType(schema) === "integer" && integerType === "number") {
    constraints.push("an integer");
  }

  if (bounds === "instructions") {
    if (typeof schema.minimum === "number" && !hasRedundantSafeIntegerBounds) {
      constraints.push(`>= ${schema.minimum}`);
    }
    if (typeof schema.maximum === "number" && !hasRedundantSafeIntegerBounds) {
      constraints.push(`<= ${schema.maximum}`);
    }
  }

  if (multipleOf === "instructions" && typeof schema.multipleOf === "number") {
    constraints.push(`a multiple of ${schema.multipleOf}`);
  }

  const transformed: ZodJsonSchema = {
    ...omitSchemaKeywords(
      schema,
      getNumberKeywordsToOmit({
        bounds,
        multipleOf,
        hasRedundantSafeIntegerBounds,
      }),
    ),
    ...(getPrimaryType(schema) === "integer" && integerType === "number" ? { type: "number" } : {}),
  };

  if (constraints.length > 0) {
    context.constraints.push(`- \`${path}\` must be ${constraints.join(", ")}`);
  }

  return {
    jsonSchema: transformed,
    toProvider: identity,
    fromProvider: identity,
  };
}

function transformStringSchema(schema: ZodJsonSchema, path: string, context: CompatibilityContext): CompatibilityNode {
  const constraints: string[] = [];
  const minLength = typeof schema.minLength === "number" ? schema.minLength : undefined;
  const maxLength = typeof schema.maxLength === "number" ? schema.maxLength : undefined;

  if (context.features.strings.length === "instructions") {
    if (minLength !== undefined && maxLength !== undefined && minLength === maxLength) {
      constraints.push(`exactly ${minLength} character${minLength === 1 ? "" : "s"}`);
    } else {
      if (minLength !== undefined) {
        constraints.push(`at least ${minLength} character${minLength === 1 ? "" : "s"}`);
      }
      if (maxLength !== undefined) {
        constraints.push(`at most ${maxLength} character${maxLength === 1 ? "" : "s"}`);
      }
    }
  }

  const transformed = context.features.strings.length === "instructions"
    ? omitSchemaKeywords(schema, ["minLength", "maxLength"])
    : { ...schema };

  if (constraints.length > 0) {
    context.constraints.push(`- \`${path}\` must have ${constraints.join(" and ")}`);
  }

  return {
    jsonSchema: transformed,
    toProvider: identity,
    fromProvider: identity,
  };
}

function transformRecordSchema(schema: ZodJsonSchema, path: string, context: CompatibilityContext): CompatibilityNode {
  context.usedRecord = true;

  const keyInfo = getFinitePropertyNames(schema.propertyNames);
  const additionalProperties = isJsonSchema(schema.additionalProperties) ? schema.additionalProperties : {};
  const valueNode = transformSchemaNode(additionalProperties, `${path}.*`, context);
  const keyType = schemaTypeToString(schema.propertyNames);
  const valueType = schemaTypeToString(additionalProperties);
  context.constraints.push(`- \`${path}\` is a \`Record<${keyType}, ${valueType}>\``);

  if (keyInfo && context.features.records.finite === "explicit-object") {
    const required = keyInfo.partial ? undefined : keyInfo.keys;
    return {
      jsonSchema: {
        ...preserveSchemaMetadata(schema),
        type: "object",
        properties: Object.fromEntries(keyInfo.keys.map((key) => [key, valueNode.jsonSchema])),
        required,
        additionalProperties: false,
      },
      toProvider(value) {
        if (!isPlainObject(value)) return value;
        const result: Record<string, unknown> = {};
        for (const key of keyInfo.keys) {
          if (key in value) result[key] = valueNode.toProvider(value[key]);
        }
        return result;
      },
      fromProvider(value) {
        if (!isPlainObject(value)) return value;
        const result: Record<string, unknown> = {};
        for (const key of keyInfo.keys) {
          if (key in value) result[key] = valueNode.fromProvider(value[key]);
        }
        return result;
      },
    };
  }

  if (context.features.records.dynamic === "entries-array" && !keyInfo) {
    context.usedDynamicRecordFallback = true;
    const keyPropertyName = context.features.records.keyPropertyName ?? "key";
    const valuePropertyName = context.features.records.valuePropertyName ?? "value";
    const keySchema = transformPropertyNameSchema(schema.propertyNames, context);

    return {
      jsonSchema: {
        ...preserveSchemaMetadata(schema),
        type: "array",
        items: {
          type: "object",
          properties: {
            [keyPropertyName]: keySchema,
            [valuePropertyName]: valueNode.jsonSchema,
          },
          required: [keyPropertyName, valuePropertyName],
          additionalProperties: false,
        },
      },
      toProvider(value) {
        if (!isPlainObject(value)) return value;
        return Object.entries(value).map(([key, entryValue]) => ({
          [keyPropertyName]: coerceKeyFromObject(key, schema.propertyNames, context),
          [valuePropertyName]: valueNode.toProvider(entryValue),
        }));
      },
      fromProvider(value) {
        if (!Array.isArray(value)) return value;
        const result: Record<string, unknown> = {};
        for (const entry of value) {
          // Ignore malformed provider entries here and let the final Zod parse decide validity.
          if (!isPlainObject(entry) || !(keyPropertyName in entry)) continue;
          result[String(entry[keyPropertyName])] = valueNode.fromProvider(entry[valuePropertyName]);
        }
        return result;
      },
    };
  }

  return {
    jsonSchema: {
      ...preserveSchemaMetadata(schema),
      type: "object",
      propertyNames: transformPropertyNameSchema(schema.propertyNames, context),
      additionalProperties: valueNode.jsonSchema,
    },
    toProvider(value) {
      if (!isPlainObject(value)) return value;
      return Object.fromEntries(
        Object.entries(value).map(([key, entryValue]) => [key, valueNode.toProvider(entryValue)]),
      );
    },
    fromProvider(value) {
      if (!isPlainObject(value)) return value;
      return Object.fromEntries(
        Object.entries(value).map(([key, entryValue]) => [key, valueNode.fromProvider(entryValue)]),
      );
    },
  };
}

function transformNativeTupleSchema(
  schema: ZodJsonSchema,
  path: string,
  context: CompatibilityContext,
): CompatibilityNode {
  const itemNodes = (schema.prefixItems ?? []).map((item, index) =>
    transformSchemaNode(item, `${path}[${index}]`, context)
  );
  const restNode = isJsonSchema(schema.items) ? transformSchemaNode(schema.items, `${path}[rest]`, context) : undefined;
  const transformed = structuredClone(schema);
  transformed.prefixItems = itemNodes.map((node) => node.jsonSchema);
  transformed.items = restNode?.jsonSchema ?? schema.items;

  return {
    jsonSchema: transformed,
    toProvider(value) {
      if (!Array.isArray(value)) return value;
      return value.map((item, index) => {
        const node = itemNodes[index] ?? restNode;
        return node ? node.toProvider(item) : item;
      });
    },
    fromProvider(value) {
      if (!Array.isArray(value)) return value;
      return value.map((item, index) => {
        const node = itemNodes[index] ?? restNode;
        return node ? node.fromProvider(item) : item;
      });
    },
  };
}

function transformTupleSchema(schema: ZodJsonSchema, path: string, context: CompatibilityContext): CompatibilityNode {
  context.usedTupleSurrogate = true;
  const itemNodes = (schema.prefixItems ?? []).map((item, index) =>
    transformSchemaNode(item, `${path}.${context.features.tuples.itemKeyPrefix ?? "item"}${index}`, context)
  );
  const restNode = isJsonSchema(schema.items)
    ? transformSchemaNode(schema.items, `${path}.${context.features.tuples.restKey ?? "rest"}[]`, context)
    : undefined;
  context.constraints.push(`- \`${path}\` is a Tuple \`${schemaTypeToString(schema)}\``);

  const itemKeyPrefix = context.features.tuples.itemKeyPrefix ?? "item";
  const restKey = context.features.tuples.restKey ?? "rest";
  const properties: Record<string, ZodJsonSchema> = {};
  const required: string[] = [];

  for (let index = 0; index < itemNodes.length; index++) {
    properties[`${itemKeyPrefix}${index}`] = itemNodes[index].jsonSchema;
    required.push(`${itemKeyPrefix}${index}`);
  }

  if (restNode) {
    properties[restKey] = { type: "array", items: restNode.jsonSchema };
  }

  return {
    jsonSchema: {
      ...preserveSchemaMetadata(schema),
      type: "object",
      properties,
      required,
      additionalProperties: false,
    },
    toProvider(value) {
      if (!Array.isArray(value)) return value;

      const result: Record<string, unknown> = {};
      for (let index = 0; index < itemNodes.length; index++) {
        result[`${itemKeyPrefix}${index}`] = itemNodes[index].toProvider(value[index]);
      }
      if (restNode && value.length > itemNodes.length) {
        // Only emit the surrogate `rest` field when the tuple actually has overflow items.
        result[restKey] = value.slice(itemNodes.length).map((item) => restNode.toProvider(item));
      }
      return result;
    },
    fromProvider(value) {
      if (!isPlainObject(value)) return value;
      const result = itemNodes.map((node, index) => node.fromProvider(value[`${itemKeyPrefix}${index}`]));
      if (restNode && Array.isArray(value[restKey])) {
        result.push(...value[restKey].map((item) => restNode.fromProvider(item)));
      }
      return result;
    },
  };
}

function wrapTopLevel(root: CompatibilityNode): ObjectCompatibilityNode {
  return {
    jsonSchema: {
      type: "object",
      properties: { content: root.jsonSchema },
      required: ["content"],
      additionalProperties: false,
    },
    toProvider(value) {
      return { content: root.toProvider(value) };
    },
    fromProvider(value) {
      return isPlainObject(value) ? root.fromProvider(value.content) : root.fromProvider(value);
    },
  };
}

function buildInstructions(
  context: CompatibilityContext,
  kind: "output" | "tool",
): string {
  const glossary = getInstructionGlossary(context);

  if (context.constraints.length === 0 && glossary.length === 0) return "";

  const { tag, intro } = getInstructionEnvelope(kind);
  const body = [
    context.constraints.length > 0 ? context.constraints.join("\n") : "",
    glossary.length > 0 ? glossary.join("\n") : "",
  ].filter(Boolean).join("\n\n");

  return `\
<${tag}>
${intro}
${body}
</${tag}>
`;
}

function transformPropertyNameSchema(
  schema: ZodJsonSchemaInput | undefined,
  context: CompatibilityContext,
): ZodJsonSchema {
  if (!isJsonSchema(schema)) return { type: "string" };

  if (Array.isArray(schema.anyOf) && schema.anyOf.length > 0) {
    return {
      ...preserveSchemaMetadata(schema),
      anyOf: schema.anyOf
        .filter((option) => !isNeverSchema(option))
        .map((option) => stripUnsupportedKeywords(option, context)),
    };
  }

  return stripUnsupportedKeywords(schema, context);
}

function stripUnsupportedKeywords(schema: ZodJsonSchema, context: CompatibilityContext): ZodJsonSchema {
  const hasRedundantSafeIntegerBounds = isRedundantSafeIntegerBounds(schema);
  const stripped: ZodJsonSchema = Object.fromEntries(
    Object.entries(schema).flatMap(([key, value]) => {
      switch (key) {
        case "minimum":
        case "maximum":
          return context.features.numbers.bounds === "instructions" ? [] : [[key, value]];
        case "multipleOf":
          return context.features.numbers.multipleOf === "instructions" ? [] : [[key, value]];
        case "minLength":
        case "maxLength":
          return context.features.strings.length === "instructions" ? [] : [[key, value]];
        case "maxItems":
          return context.features.arrays.length === "native" ? [[key, value]] : [];
        case "minItems":
          if (context.features.arrays.length === "native") return [[key, value]];
          if (context.features.arrays.length === "min1-only" && value === 1) return [[key, value]];
          return [];
        case "items":
          if (isJsonSchema(value)) return [[key, stripUnsupportedKeywords(value, context)]];
          if (Array.isArray(value)) {
            return [[key, value.map((item) => isJsonSchema(item) ? stripUnsupportedKeywords(item, context) : item)]];
          }
          return [[key, value]];
        case "anyOf":
          return [[
            key,
            Array.isArray(value)
              ? value.map((option) => isJsonSchema(option) ? stripUnsupportedKeywords(option, context) : option)
              : value,
          ]];
        case "prefixItems":
          return [[
            key,
            Array.isArray(value)
              ? value.map((option) => isJsonSchema(option) ? stripUnsupportedKeywords(option, context) : option)
              : value,
          ]];
        case "properties":
          return [[
            key,
            isPlainObject(value)
              ? Object.fromEntries(
                Object.entries(value).map(([propertyKey, propertyValue]) => [
                  propertyKey,
                  isJsonSchema(propertyValue) ? stripUnsupportedKeywords(propertyValue, context) : propertyValue,
                ]),
              )
              : value,
          ]];
        case "additionalProperties":
        case "propertyNames":
          return [[key, isJsonSchema(value) ? stripUnsupportedKeywords(value, context) : value]];
        default:
          return [[key, value]];
      }
    }),
  );

  const normalized = hasRedundantSafeIntegerBounds ? omitSchemaKeywords(stripped, ["minimum", "maximum"]) : stripped;

  if (getPrimaryType(schema) === "integer" && context.features.numbers.integerType === "number") {
    return { ...normalized, type: "number" };
  }

  return normalized;
}

function createCompatibilityContext(features: SchemaCompatibilityFeatures): CompatibilityContext {
  return {
    constraints: [],
    usedRecord: false,
    usedDynamicRecordFallback: false,
    usedTupleSurrogate: false,
    features,
  };
}

function shouldWrapTopLevelObject(requireTopLevelObject: boolean | undefined, schema: ZodJsonSchema): boolean {
  return Boolean(requireTopLevelObject) && !isObjectLikeSchema(schema);
}

function getArrayKeywordsToOmit(
  length: SchemaCompatibilityFeatures["arrays"]["length"],
  minItems: number | undefined,
): string[] {
  if (length === "instructions") return ["minItems", "maxItems"];
  if (length === "min1-only") return minItems === 1 ? ["maxItems"] : ["minItems", "maxItems"];
  return [];
}

function getNumberKeywordsToOmit(options: {
  bounds: SchemaCompatibilityFeatures["numbers"]["bounds"];
  multipleOf: SchemaCompatibilityFeatures["numbers"]["multipleOf"];
  hasRedundantSafeIntegerBounds: boolean;
}): string[] {
  const keys: string[] = [];
  if (options.bounds === "instructions" || options.hasRedundantSafeIntegerBounds) {
    keys.push("minimum", "maximum");
  }
  if (options.multipleOf === "instructions") {
    keys.push("multipleOf");
  }
  return keys;
}

function getInstructionGlossary(context: CompatibilityContext): string[] {
  const glossary: string[] = [];

  if (context.usedTupleSurrogate) {
    glossary.push(
      "Tuple - An ordered fixed-length sequence. In the compatible schema it is represented as an object with keys `item0`, `item1`, etc. If the tuple has a rest element, it appears under `rest`.",
    );
  }

  if (context.usedDynamicRecordFallback) {
    glossary.push(
      "Record - A key-value map. Finite-key records are represented as explicit objects. Dynamic records are represented as arrays of `{ key, value }` objects because compatibility mode avoids `additionalProperties` schemas.",
    );
  } else if (context.usedRecord && context.features.records.finite === "explicit-object") {
    glossary.push(
      "Record - A key-value map. Finite-key records are represented as explicit objects in the compatible schema.",
    );
  }

  return glossary;
}

function getInstructionEnvelope(kind: "output" | "tool"): { tag: string; intro: string } {
  if (kind === "output") {
    return {
      tag: "output_requirements",
      intro:
        "Your final output must conform to the provided JSON Schema, and additionally satisfy the following requirements:",
    };
  }

  return {
    tag: "input_requirements",
    intro:
      "Tool arguments must conform to the provided JSON Schema, and additionally satisfy the following requirements:",
  };
}

function omitSchemaKeywords(schema: ZodJsonSchema, keys: string[]): ZodJsonSchema {
  if (keys.length === 0) return { ...schema };
  const keySet = new Set(keys);
  return Object.fromEntries(Object.entries(schema).filter(([key]) => !keySet.has(key)));
}

function identity(value: unknown): unknown {
  return value;
}

function identityNode(schema: ZodJsonSchema, context: CompatibilityContext): CompatibilityNode {
  return {
    jsonSchema: stripUnsupportedKeywords(schema, context),
    toProvider: identity,
    fromProvider: identity,
  };
}

function getPrimaryType(schema: ZodJsonSchema): string | undefined {
  return Array.isArray(schema.type) ? schema.type[0] : schema.type;
}

function isJsonSchema(value: unknown): value is ZodJsonSchema {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isObjectLikeSchema(schema: ZodJsonSchema): schema is JSONSchema.ObjectSchema {
  return getPrimaryType(schema) === "object" || !!schema.properties;
}

function isRecordSchema(schema: ZodJsonSchema): boolean {
  return getPrimaryType(schema) === "object" && isJsonSchema(schema.additionalProperties);
}

function isRedundantSafeIntegerBounds(schema: ZodJsonSchema): boolean {
  return getPrimaryType(schema) === "integer" &&
    schema.minimum === Number.MIN_SAFE_INTEGER &&
    schema.maximum === Number.MAX_SAFE_INTEGER;
}

function isNeverSchema(schema: ZodJsonSchema): boolean {
  return isJsonSchema(schema.not) && Object.keys(schema.not).length === 0;
}

function getFinitePropertyNames(schema: ZodJsonSchemaInput | undefined): { keys: string[]; partial: boolean } | null {
  if (!isJsonSchema(schema)) return null;

  if (
    Array.isArray(schema.enum) && schema.enum.every((value) => typeof value === "string" || typeof value === "number")
  ) {
    return { keys: schema.enum.map(String), partial: false };
  }

  if ((typeof schema.const === "string" || typeof schema.const === "number")) {
    return { keys: [String(schema.const)], partial: false };
  }

  if (Array.isArray(schema.anyOf)) {
    let partial = false;
    const keys: string[] = [];
    for (const option of schema.anyOf) {
      if (isNeverSchema(option)) {
        partial = true;
        continue;
      }
      const nested = getFinitePropertyNames(option);
      if (!nested || nested.partial) return null;
      keys.push(...nested.keys);
    }
    return keys.length > 0 ? { keys: [...new Set(keys)], partial } : null;
  }

  return null;
}

function schemaTypeToString(schema: ZodJsonSchemaInput | ZodJsonSchemaInput[] | undefined): string {
  if (schema === false) return "never";
  if (schema === true || schema === undefined) return "unknown";
  if (Array.isArray(schema)) {
    return schema.map((item) => schemaTypeToString(item)).join(" | ");
  }
  if (Array.isArray(schema.anyOf) && schema.anyOf.length > 0) {
    return schema.anyOf.map((option) => schemaTypeToString(option)).join(" | ");
  }
  if (Array.isArray(schema.enum) && schema.enum.length > 0) {
    return schema.enum.map((value) => JSON.stringify(value)).join(" | ");
  }
  if (schema.const !== undefined) return JSON.stringify(schema.const);
  if (Array.isArray(schema.prefixItems)) {
    const items = schema.prefixItems.map((item) => schemaTypeToString(item));
    if (schema.items) return `[${items.join(", ")}, ...${schemaTypeToString(schema.items)}[]]`;
    return `[${items.join(", ")}]`;
  }

  const type = getPrimaryType(schema);
  switch (type) {
    case "object": {
      if (isJsonSchema(schema.additionalProperties)) {
        return `Record<${schemaTypeToString(schema.propertyNames)}, ${
          schemaTypeToString(schema.additionalProperties)
        }>`;
      }
      return "object";
    }
    case "array":
      return `${schemaTypeToString(schema.items)}[]`;
    case "integer":
      return "integer";
    case "number":
    case "string":
    case "boolean":
    case "null":
      return type;
    default:
      return "unknown";
  }
}

/**
 * When we rebuild a schema node from scratch, keep only human-facing annotations.
 * Spreading the full schema would also copy structural keywords that are being intentionally replaced.
 */
function preserveSchemaMetadata(schema: ZodJsonSchema): ZodJsonSchema {
  const metadata: ZodJsonSchema = {};
  if (schema.description !== undefined) metadata.description = schema.description;
  if (schema.title !== undefined) metadata.title = schema.title;
  if (schema.default !== undefined) metadata.default = schema.default;
  return metadata;
}

function matchesSchema(value: unknown, schema: ZodJsonSchemaInput | undefined, context: CompatibilityContext): boolean {
  if (schema === false) return false;
  if (schema === true || schema === undefined) return true;

  if (Array.isArray(schema.anyOf) && schema.anyOf.length > 0) {
    return schema.anyOf.some((option) => matchesSchema(value, option, context));
  }

  if (Array.isArray(schema.enum)) return schema.enum.some((candidate) => candidate === value);
  if (schema.const !== undefined) return schema.const === value;

  if (Array.isArray(schema.prefixItems)) {
    return context.features.tuples.strategy === "native" ? Array.isArray(value) : isPlainObject(value);
  }

  switch (getPrimaryType(schema)) {
    case "object":
      return isPlainObject(value);
    case "array":
      return Array.isArray(value);
    case "integer":
      return typeof value === "number" && Number.isInteger(value);
    case "number":
      return typeof value === "number";
    case "string":
      return typeof value === "string";
    case "boolean":
      return typeof value === "boolean";
    case "null":
      return value === null;
    default:
      return true;
  }
}

function coerceKeyFromObject(
  key: string,
  propertyNames: ZodJsonSchemaInput | undefined,
  context: CompatibilityContext,
): JsonValue {
  if (!isJsonSchema(propertyNames)) return key;

  if (Array.isArray(propertyNames.anyOf)) {
    for (const option of propertyNames.anyOf) {
      if (isNeverSchema(option)) continue;
      const coerced = coerceKeyFromObject(key, option, context);
      if (matchesSchema(coerced, option, context)) return coerced;
    }
    return key;
  }

  if (getPrimaryType(propertyNames) === "number" || getPrimaryType(propertyNames) === "integer") {
    const numeric = Number(key);
    return Number.isNaN(numeric) ? key : numeric;
  }
  if ((typeof propertyNames.const === "string" || typeof propertyNames.const === "number")) {
    return propertyNames.const;
  }
  if (Array.isArray(propertyNames.enum)) {
    const match = propertyNames.enum.find((candidate) => String(candidate) === key);
    return match ?? key;
  }
  return key;
}
