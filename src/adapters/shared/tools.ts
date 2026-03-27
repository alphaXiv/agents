import type z from "zod";

export interface SharedToolShape {
  wrapperObject: boolean;
  isVoid: boolean;
}

export interface SharedToolSchemaInfo extends SharedToolShape {
  schema: z.ZodType;
  jsonSchema: Record<string, unknown>;
}

export function serializeWrappedToolArguments(content: string | undefined, tool: SharedToolShape | undefined): string {
  if (tool?.isVoid) return "{}";
  if (!content) return "{}";
  if (!tool?.wrapperObject) return content;
  return `{"content":${content}}`;
}

export function restoreWrappedToolArguments(content: string, tool: SharedToolShape | undefined): string | undefined {
  if (tool?.isVoid) return undefined;
  if (!tool?.wrapperObject) return content;

  try {
    const parsed = JSON.parse(content) as { content?: unknown };
    return parsed.content === undefined ? undefined : JSON.stringify(parsed.content);
  } catch {
    return content;
  }
}
