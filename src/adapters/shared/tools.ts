interface SharedSchemaCompatibility {
  toProvider(value: unknown): unknown;
  fromProvider(value: unknown): unknown;
}

export interface SharedToolShape {
  wrapperObject: boolean;
  isVoid: boolean;
  compatibility?: SharedSchemaCompatibility;
}

export function serializeWrappedToolArguments(content: string | undefined, tool: SharedToolShape | undefined): string {
  if (tool?.isVoid) return "{}";
  if (!content) return "{}";

  if (tool?.compatibility) {
    try {
      return JSON.stringify(tool.compatibility.toProvider(JSON.parse(content)));
    } catch {
      return content;
    }
  }

  if (!tool?.wrapperObject) return content;
  return `{"content":${content}}`;
}

export function restoreWrappedToolArguments(content: string, tool: SharedToolShape | undefined): string | undefined {
  if (tool?.isVoid) return undefined;

  if (tool?.compatibility) {
    try {
      const restored = tool.compatibility.fromProvider(JSON.parse(content));
      return restored === undefined ? undefined : JSON.stringify(restored);
    } catch {
      return content;
    }
  }

  if (!tool?.wrapperObject) return content;

  try {
    const parsed = JSON.parse(content) as { content?: unknown };
    return parsed.content === undefined ? undefined : JSON.stringify(parsed.content);
  } catch {
    return content;
  }
}
