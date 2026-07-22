interface SharedSchemaCompatibility {
  toProvider(value: unknown): unknown;
  fromProvider(value: unknown): unknown;
}

export interface SharedToolShape {
  wrapperObject: boolean;
  isVoid: boolean;
  compatibility?: SharedSchemaCompatibility;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * A tool call's `input` must be a JSON object; providers such as Anthropic reject
 * a bare scalar with `input: Input should be an object`. A wrapper-object tool
 * stores only its inner value, so replaying that call after the tool is no longer
 * registered (nothing left to re-wrap it) would surface the scalar. Wrap stray
 * non-objects under `content`, mirroring the wrapper shape.
 */
export function ensureToolInputObject(value: unknown): Record<string, unknown> {
  if (isPlainObject(value)) return value;
  return value === undefined ? {} : { content: value };
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

  if (tool?.wrapperObject) return `{"content":${content}}`;

  // Known object tool → `content` is already an object; an unregistered tool
  // (tool === undefined) may hold a wrapper tool's scalar. Guarantee an object.
  try {
    const parsed = JSON.parse(content);
    return isPlainObject(parsed) ? content : JSON.stringify(ensureToolInputObject(parsed));
  } catch {
    return content;
  }
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
