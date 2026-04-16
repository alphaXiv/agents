import type { Tool as AnthropicTool } from "@anthropic-ai/sdk/resources/messages/messages";
import z from "zod";
import type { JSONSchema } from "zod/v4/core";
import type { AnyTool, Tool } from "../../tool.ts";

type ZodJsonSchema = JSONSchema.BaseSchema;
type ZodJsonSchemaInput = JSONSchema._JSONSchema;
type JsonValue = NonNullable<ZodJsonSchema["const"]>;

export interface AnthropicCompatibleSchema {
  /** The original Zod schema used to define the tool's parameters or structured output. */
  originalJsonSchema: ZodJsonSchema;
  /** Anthropic-compatible JSON Schema used for structured output or tool input validation. */
  jsonSchema: ZodJsonSchema;
  /** Additional prompt instructions required to preserve semantics stripped from the schema. */
  instructions: string;
  /** Convert a value in the original schema shape into the Anthropic-compatible surrogate shape. */
  toAnthropic(value: unknown): unknown;
  /** Convert a value produced in the Anthropic-compatible surrogate shape back to the original schema shape. */
  fromAnthropic(value: unknown): unknown;
}

interface AnthropicCompatibleObjectSchema extends AnthropicCompatibleSchema {
  jsonSchema: JSONSchema.ObjectSchema;
}

export type AnthropicToolMap = {
  original: Tool;
  anthropic: AnthropicTool;
  /** Anthropic-compatible schema and bidirectional conversion for the tool's parameters. */
  compatibility?: AnthropicCompatibleSchema;
  /** No parameter specified */
  isVoid: boolean;
};

interface CreateAnthropicCompatibleSchemaOptions {
  kind: "output" | "tool";
  requireTopLevelObject?: boolean;
  rootPath: string;
}

interface CreateAnthropicCompatibleObjectSchemaOptions extends CreateAnthropicCompatibleSchemaOptions {
  requireTopLevelObject: true;
}

interface CompatibilityNode {
  jsonSchema: ZodJsonSchema;
  toAnthropic(value: unknown): unknown;
  fromAnthropic(value: unknown): unknown;
}

interface ObjectCompatibilityNode extends CompatibilityNode {
  jsonSchema: JSONSchema.ObjectSchema;
}

interface TransformContext {
  constraints: string[];
  usedRecord: boolean;
  usedDynamicRecordFallback: boolean;
  usedTuple: boolean;
}

const INTEGER_BOUNDS =
  `within the JavaScript safe integer range (${Number.MIN_SAFE_INTEGER} to ${Number.MAX_SAFE_INTEGER})`;

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
        description: compatibleSchema?.instructions
          ? `${tool.description}\n\n${compatibleSchema.instructions}`
          : tool.description,
      },
      compatibility: compatibleSchema,
      isVoid: false,
    };
  });
}

/**
 * Converts a Zod schema into an Anthropic-compatible contract while preserving
 * the original logical shape for the rest of the app.
 *
 * Why this exists:
 * Anthropic structured output accepts a restricted JSON Schema subset. Some
 * shapes that are valid and useful in our original Zod schema are awkward or
 * unsupported there, especially:
 * - dynamic records (`Record<string, T>`) because they rely on  `additionalProperties`
 * - tuples (`[A, B, ...]`) because they rely on `prefixItems`
 * - min and max constraints on strings, numbers, and arrays because they rely on keywords like `minLength`, `maxLength` etc.
 *   - This also made integer schemas unsupported because zod automatically attaches MIN_SAFE_INTEGER and MAX_SAFE_INTEGER bounds to them
 * - constraints we still care about but cannot or do not want to enforce purely through Anthropic's accepted schema subset
 *
 * What this function returns:
 * - `jsonSchema`: the provider-facing schema we actually send to Anthropic
 * - `instructions`: extra prompt guidance describing any semantics moved out of the schema
 * - `toAnthropic`: converts an original-shape value into the surrogate shape Anthropic expects
 * - `fromAnthropic`: restores Anthropic's surrogate output back into the original logical shape before the Agent validates it with the original Zod schema
 *
 * In practice, this lets the provider work with a simplified JSON Schema while
 * the rest of the system continues to think in terms of the original schema.
 * @example
 * ```ts
 * const compatibility = createAnthropicCompatibleSchema(
 *   z.object({ tags: z.record(z.string(), z.string()) }),
 *   { kind: "output", rootPath: "output" },
 * );
 *
 * compatibility.toAnthropic({ tags: { a: "1" } });
 * // => { tags: [{ key: "a", value: "1" }] }
 *
 * compatibility.fromAnthropic({ tags: [{ key: "a", value: "1" }] });
 * // => { tags: { a: "1" } }
 * ```
 */
export function createAnthropicCompatibleSchema(
  schema: z.ZodType,
  options: CreateAnthropicCompatibleObjectSchemaOptions,
): AnthropicCompatibleObjectSchema;
export function createAnthropicCompatibleSchema(
  schema: z.ZodType,
  options: CreateAnthropicCompatibleSchemaOptions,
): AnthropicCompatibleSchema;

export function createAnthropicCompatibleSchema(
  schema: z.ZodType,
  options: CreateAnthropicCompatibleSchemaOptions,
): AnthropicCompatibleSchema {
  const originalJsonSchema = z.toJSONSchema(schema);
  const context: TransformContext = {
    constraints: [],
    usedRecord: false,
    usedDynamicRecordFallback: false,
    usedTuple: false,
  };

  const root = transformSchemaNode(originalJsonSchema, options.rootPath, context);

  if (!options.requireTopLevelObject) {
    return {
      originalJsonSchema,
      jsonSchema: root.jsonSchema,
      instructions: buildInstructions(context, options.kind),
      toAnthropic: root.toAnthropic,
      fromAnthropic: root.fromAnthropic,
    };
  }

  if (isObjectLikeSchema(root.jsonSchema)) {
    return {
      originalJsonSchema,
      jsonSchema: root.jsonSchema,
      instructions: buildInstructions(context, options.kind),
      toAnthropic: root.toAnthropic,
      fromAnthropic: root.fromAnthropic,
    };
  }

  const compatibility = wrapTopLevel(root);

  return {
    originalJsonSchema,
    jsonSchema: compatibility.jsonSchema,
    instructions: buildInstructions(context, options.kind),
    toAnthropic: compatibility.toAnthropic,
    fromAnthropic: compatibility.fromAnthropic,
  };
}

function transformSchemaNode(
  schema: ZodJsonSchemaInput | undefined,
  path: string,
  context: TransformContext,
): CompatibilityNode {
  const normalizedSchema = isJsonSchema(schema) ? schema : {};

  if (Array.isArray(normalizedSchema.anyOf) && normalizedSchema.anyOf.length > 0) {
    return transformAnyOfSchema(normalizedSchema, path, context);
  }

  if (Array.isArray(normalizedSchema.prefixItems) && normalizedSchema.prefixItems.length > 0) {
    return transformTupleSchema(normalizedSchema, path, context);
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
      return identityNode(normalizedSchema);
  }
}

function transformAnyOfSchema(schema: ZodJsonSchema, path: string, context: TransformContext): CompatibilityNode {
  const options = schema.anyOf ?? [];
  const nodes = options.map((option, index) => transformSchemaNode(option, `${path}[option ${index}]`, context));

  return {
    jsonSchema: {
      ...preserveMeta(schema),
      anyOf: nodes.map((node) => node.jsonSchema),
    },
    toAnthropic(value) {
      const index = options.findIndex((option) => matchesSchema(value, option));
      return index === -1 ? value : nodes[index].toAnthropic(value);
    },
    fromAnthropic(value) {
      const index = nodes.findIndex((node) => matchesSchema(value, node.jsonSchema));
      return index === -1 ? value : nodes[index].fromAnthropic(value);
    },
  };
}

function transformObjectSchema(schema: ZodJsonSchema, path: string, context: TransformContext): CompatibilityNode {
  const propertyEntries = Object.entries(schema.properties ?? {});
  const nodes = Object.fromEntries(
    propertyEntries.map(([key, value]) => [key, transformSchemaNode(value, `${path}.${key}`, context)]),
  );

  return {
    jsonSchema: {
      ...preserveMeta(schema),
      type: "object",
      properties: Object.fromEntries(Object.entries(nodes).map(([key, node]) => [key, node.jsonSchema])),
      required: schema.required,
      additionalProperties: false,
    },
    toAnthropic(value) {
      if (!isPlainObject(value)) return value;
      const result: Record<string, unknown> = {};
      for (const [key, node] of Object.entries(nodes)) {
        if (key in value) result[key] = node.toAnthropic(value[key]);
      }
      return result;
    },
    fromAnthropic(value) {
      if (!isPlainObject(value)) return value;
      const result: Record<string, unknown> = {};
      for (const [key, node] of Object.entries(nodes)) {
        if (key in value) result[key] = node.fromAnthropic(value[key]);
      }
      return result;
    },
  };
}

function transformArraySchema(schema: ZodJsonSchema, path: string, context: TransformContext): CompatibilityNode {
  const itemNode = transformSchemaNode(Array.isArray(schema.items) ? undefined : schema.items, `${path}[]`, context);
  const constraints: string[] = [];
  const minItems = typeof schema.minItems === "number" ? schema.minItems : undefined;
  const maxItems = typeof schema.maxItems === "number" ? schema.maxItems : undefined;

  if (minItems !== undefined && maxItems !== undefined && minItems === maxItems) {
    constraints.push(`exactly ${minItems} item${minItems === 1 ? "" : "s"}`);
  } else {
    if (minItems !== undefined && minItems > 1) {
      constraints.push(`at least ${minItems} item${minItems === 1 ? "" : "s"}`);
    }
    if (maxItems !== undefined) constraints.push(`at most ${maxItems} item${maxItems === 1 ? "" : "s"}`);
  }

  if (constraints.length > 0) {
    context.constraints.push(`- \`${path}\` must contain ${constraints.join(" and ")}`);
  }

  const transformed: ZodJsonSchema = {
    ...preserveMeta(schema),
    type: "array",
    items: itemNode.jsonSchema,
  };
  if (minItems === 1) transformed.minItems = 1;

  return {
    jsonSchema: transformed,
    toAnthropic(value) {
      return Array.isArray(value) ? value.map((item) => itemNode.toAnthropic(item)) : value;
    },
    fromAnthropic(value) {
      return Array.isArray(value) ? value.map((item) => itemNode.fromAnthropic(item)) : value;
    },
  };
}

function transformNumberSchema(schema: ZodJsonSchema, path: string, context: TransformContext): CompatibilityNode {
  const constraints: string[] = [];

  if (getPrimaryType(schema) === "integer") {
    constraints.push("an integer");
    if (schema.minimum === Number.MIN_SAFE_INTEGER && schema.maximum === Number.MAX_SAFE_INTEGER) {
      constraints.push(INTEGER_BOUNDS);
    }
  }

  if (
    typeof schema.minimum === "number" &&
    !(schema.minimum === Number.MIN_SAFE_INTEGER && schema.maximum === Number.MAX_SAFE_INTEGER)
  ) {
    constraints.push(`>= ${schema.minimum}`);
  }
  if (
    typeof schema.maximum === "number" &&
    !(schema.minimum === Number.MIN_SAFE_INTEGER && schema.maximum === Number.MAX_SAFE_INTEGER)
  ) {
    constraints.push(`<= ${schema.maximum}`);
  }
  if (typeof schema.multipleOf === "number") constraints.push(`a multiple of ${schema.multipleOf}`);

  if (constraints.length > 0) {
    context.constraints.push(`- \`${path}\` must be ${constraints.join(", ")}`);
  }

  return {
    jsonSchema: {
      ...preserveMeta(schema),
      type: "number",
    },
    toAnthropic: identity,
    fromAnthropic: identity,
  };
}

function transformStringSchema(schema: ZodJsonSchema, path: string, context: TransformContext): CompatibilityNode {
  const constraints: string[] = [];
  const minLength = typeof schema.minLength === "number" ? schema.minLength : undefined;
  const maxLength = typeof schema.maxLength === "number" ? schema.maxLength : undefined;

  if (minLength !== undefined && maxLength !== undefined && minLength === maxLength) {
    constraints.push(`exactly ${minLength} character${minLength === 1 ? "" : "s"}`);
  } else {
    if (minLength !== undefined) constraints.push(`at least ${minLength} character${minLength === 1 ? "" : "s"}`);
    if (maxLength !== undefined) constraints.push(`at most ${maxLength} character${maxLength === 1 ? "" : "s"}`);
  }

  if (constraints.length > 0) {
    context.constraints.push(`- \`${path}\` must have ${constraints.join(" and ")}`);
  }

  const transformed = structuredClone(schema);
  delete transformed.minLength;
  delete transformed.maxLength;
  return {
    jsonSchema: transformed,
    toAnthropic: identity,
    fromAnthropic: identity,
  };
}

function transformRecordSchema(schema: ZodJsonSchema, path: string, context: TransformContext): CompatibilityNode {
  context.usedRecord = true;

  const keyInfo = getFinitePropertyNames(schema.propertyNames);
  const additionalProperties = isJsonSchema(schema.additionalProperties) ? schema.additionalProperties : {};
  const valueNode = transformSchemaNode(additionalProperties, `${path}.*`, context);
  const keyType = schemaTypeToString(schema.propertyNames);
  const valueType = schemaTypeToString(additionalProperties);
  context.constraints.push(`- \`${path}\` is a \`Record<${keyType}, ${valueType}>\``);

  if (keyInfo) {
    const required = keyInfo.partial ? undefined : keyInfo.keys;
    return {
      jsonSchema: {
        ...preserveMeta(schema),
        type: "object",
        properties: Object.fromEntries(keyInfo.keys.map((key) => [key, valueNode.jsonSchema])),
        required,
        additionalProperties: false,
      },
      toAnthropic(value) {
        if (!isPlainObject(value)) return value;
        const result: Record<string, unknown> = {};
        for (const key of keyInfo.keys) {
          if (key in value) result[key] = valueNode.toAnthropic(value[key]);
        }
        return result;
      },
      fromAnthropic(value) {
        if (!isPlainObject(value)) return value;
        const result: Record<string, unknown> = {};
        for (const key of keyInfo.keys) {
          if (key in value) result[key] = valueNode.fromAnthropic(value[key]);
        }
        return result;
      },
    };
  }

  context.usedDynamicRecordFallback = true;
  const keySchema = transformPropertyNameSchema(schema.propertyNames);
  return {
    jsonSchema: {
      ...preserveMeta(schema),
      type: "array",
      items: {
        type: "object",
        properties: {
          key: keySchema,
          value: valueNode.jsonSchema,
        },
        required: ["key", "value"],
        additionalProperties: false,
      },
    },
    toAnthropic(value) {
      if (!isPlainObject(value)) return value;
      return Object.entries(value).map(([key, entryValue]) => ({
        key: coerceKeyFromObject(key, schema.propertyNames),
        value: valueNode.toAnthropic(entryValue),
      }));
    },
    fromAnthropic(value) {
      if (!Array.isArray(value)) return value;
      const result: Record<string, unknown> = {};
      for (const entry of value) {
        if (!isPlainObject(entry) || !("key" in entry)) continue;
        result[String(entry.key)] = valueNode.fromAnthropic(entry.value);
      }
      return result;
    },
  };
}

function transformTupleSchema(schema: ZodJsonSchema, path: string, context: TransformContext): CompatibilityNode {
  context.usedTuple = true;
  const itemNodes = (schema.prefixItems ?? []).map((item, index) =>
    transformSchemaNode(item, `${path}.item${index}`, context)
  );
  const restNode = isJsonSchema(schema.items)
    ? transformSchemaNode(schema.items, `${path}.rest[]`, context)
    : undefined;
  context.constraints.push(`- \`${path}\` is a Tuple \`${schemaTypeToString(schema)}\``);

  const properties: Record<string, ZodJsonSchema> = {};
  const required: string[] = [];
  for (let index = 0; index < itemNodes.length; index++) {
    properties[`item${index}`] = itemNodes[index].jsonSchema;
    required.push(`item${index}`);
  }
  if (restNode) properties.rest = { type: "array", items: restNode.jsonSchema };

  return {
    jsonSchema: {
      ...preserveMeta(schema),
      type: "object",
      properties,
      required,
      additionalProperties: false,
    },
    toAnthropic(value) {
      if (!Array.isArray(value)) return value;
      const result: Record<string, unknown> = {};
      for (let index = 0; index < itemNodes.length; index++) {
        result[`item${index}`] = itemNodes[index].toAnthropic(value[index]);
      }
      if (restNode && value.length > itemNodes.length) {
        result.rest = value.slice(itemNodes.length).map((item) => restNode.toAnthropic(item));
      }
      return result;
    },
    fromAnthropic(value) {
      if (!isPlainObject(value)) return value;
      const result = itemNodes.map((node, index) => node.fromAnthropic(value[`item${index}`]));
      if (restNode && Array.isArray(value.rest)) {
        result.push(...value.rest.map((item) => restNode.fromAnthropic(item)));
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
    toAnthropic(value) {
      return { content: root.toAnthropic(value) };
    },
    fromAnthropic(value) {
      return isPlainObject(value) ? root.fromAnthropic(value.content) : root.fromAnthropic(value);
    },
  };
}

function buildInstructions(context: TransformContext, kind: "output" | "tool"): string {
  if (context.constraints.length === 0 && !context.usedRecord && !context.usedTuple) return "";

  const glossary: string[] = [];
  if (context.usedTuple) {
    glossary.push(
      "Tuple - An ordered fixed-length sequence. In the Anthropic-compatible schema it is represented as an object with keys `item0`, `item1`, etc. If the tuple has a rest element, it appears under `rest`.",
    );
  }
  if (context.usedRecord) {
    glossary.push(
      context.usedDynamicRecordFallback
        ? "Record - A key-value map. Finite-key records are represented as explicit objects. Dynamic records are represented as arrays of `{ key, value }` objects because Anthropic does not allow `additionalProperties` schemas."
        : "Record - A key-value map. Finite-key records are represented as explicit objects in the Anthropic-compatible schema.",
    );
  }

  if (kind === "output") {
    return `\
<output_requirements>
Your final output must conform to the provided JSON Schema, and additionally satisfy the following requirements:
${context.constraints.length > 0 ? context.constraints.join("\n") : ""}
${glossary.length > 0 ? "\n\n" + glossary.join("\n") : ""}
</output_requirements>
`;
  }

  return `\
<input_requirements>
Tool arguments must conform to the provided JSON Schema, and additionally satisfy the following requirements:
${context.constraints.length > 0 ? context.constraints.join("\n") : ""}
${glossary.length > 0 ? "\n\n" + glossary.join("\n") : ""}
</input_requirements>
`;
}

function transformPropertyNameSchema(schema: ZodJsonSchemaInput | undefined): ZodJsonSchema {
  if (!isJsonSchema(schema)) return { type: "string" };

  if (Array.isArray(schema.anyOf) && schema.anyOf.length > 0) {
    return {
      ...preserveMeta(schema),
      anyOf: schema.anyOf
        .filter((option) => !isNeverSchema(option))
        .map((option) => stripUnsupportedKeywords(option)),
    };
  }

  return stripUnsupportedKeywords(schema);
}

function stripUnsupportedKeywords(schema: ZodJsonSchema): ZodJsonSchema {
  const stripped: ZodJsonSchema = Object.fromEntries(
    Object.entries(schema).flatMap(([key, value]) => {
      switch (key) {
        case "minimum":
        case "maximum":
        case "multipleOf":
        case "minLength":
        case "maxLength":
        case "maxItems":
          return [];
        case "minItems":
          return value === 1 ? [[key, value]] : [];
        case "items":
          if (isJsonSchema(value)) return [[key, stripUnsupportedKeywords(value)]];
          if (Array.isArray(value)) {
            return [[key, value.map((item) => isJsonSchema(item) ? stripUnsupportedKeywords(item) : item)]];
          }
          return [[key, value]];
        case "anyOf":
          return [[key, Array.isArray(value) ? value.map((option) => stripUnsupportedKeywords(option)) : value]];
        case "prefixItems":
          return [[
            key,
            Array.isArray(value)
              ? value.map((option) => isJsonSchema(option) ? stripUnsupportedKeywords(option) : option)
              : value,
          ]];
        case "properties":
          return [[
            key,
            isPlainObject(value)
              ? Object.fromEntries(
                Object.entries(value).map(([propertyKey, propertyValue]) => [
                  propertyKey,
                  isJsonSchema(propertyValue) ? stripUnsupportedKeywords(propertyValue) : propertyValue,
                ]),
              )
              : value,
          ]];
        case "additionalProperties":
        case "propertyNames":
          return [[key, isJsonSchema(value) ? stripUnsupportedKeywords(value) : value]];
        default:
          return [[key, value]];
      }
    }),
  );

  return stripped;
}

function identity(value: unknown): unknown {
  return value;
}

function identityNode(schema: ZodJsonSchema): CompatibilityNode {
  return {
    jsonSchema: stripUnsupportedKeywords(schema),
    toAnthropic: identity,
    fromAnthropic: identity,
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

function preserveMeta(schema: ZodJsonSchema): ZodJsonSchema {
  const meta: ZodJsonSchema = {};
  if (schema.description !== undefined) meta.description = schema.description;
  if (schema.title !== undefined) meta.title = schema.title;
  if (schema.default !== undefined) meta.default = schema.default;
  return meta;
}

function matchesSchema(value: unknown, schema: ZodJsonSchemaInput | undefined): boolean {
  if (schema === false) return false;
  if (schema === true || schema === undefined) return true;

  if (Array.isArray(schema.anyOf) && schema.anyOf.length > 0) {
    return schema.anyOf.some((option) => matchesSchema(value, option));
  }

  if (Array.isArray(schema.enum)) return schema.enum.some((candidate) => candidate === value);
  if (schema.const !== undefined) return schema.const === value;

  if (Array.isArray(schema.prefixItems)) return isPlainObject(value);

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

function coerceKeyFromObject(key: string, propertyNames: ZodJsonSchemaInput | undefined): JsonValue {
  if (!isJsonSchema(propertyNames)) return key;

  if (Array.isArray(propertyNames.anyOf)) {
    for (const option of propertyNames.anyOf) {
      if (isNeverSchema(option)) continue;
      const coerced = coerceKeyFromObject(key, option);
      if (matchesSchema(coerced, option)) return coerced;
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
