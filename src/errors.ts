import { errMessage } from "./util.ts";

/**
 * Known error categories that can occur during model calls.
 * Used to determine retry behavior.
 */
export const ERROR_KINDS = [
  "quota_exceeded",
  "rate_limit",
  "model_unavailable",
  "network",
  "timeout",
  "server",
  "client",
  "auth",
  "aborted",
  "context_overflow",
  "unsupported_file_type",
  "image_too_large",
  "unknown",
] as const;

export type ErrorKind = (typeof ERROR_KINDS)[number];

export interface ClassifiedError {
  kind: ErrorKind;
  status?: number;
  original: unknown;
}

function errObj(error: unknown): Record<string, unknown> {
  if (error && typeof error === "object") {
    return error as Record<string, unknown>;
  }
  return {};
}

function tryParseJsonRecord(text: string): Record<string, unknown> | null {
  try {
    const parsed = JSON.parse(text);
    return parsed && typeof parsed === "object" ? (parsed as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

/**
 * Extracts HTTP status code from various error object shapes.
 * Handles nested structures like { error: { status: 503 } }.
 */
function extractStatusFromPayload(value: unknown): number | undefined {
  let current = value;

  // Check up to 3 levels deep for common status code fields
  for (let depth = 0; depth < 3; depth += 1) {
    if (!current || typeof current !== "object") return undefined;

    const record = current as Record<string, unknown>;
    if (typeof record["status"] === "number") return record["status"];
    if (typeof record["statusCode"] === "number") return record["statusCode"];
    if (typeof record["code"] === "number") return record["code"];

    current = record["error"];
  }

  return undefined;
}

function isLikelyProviderRateLimit(text: string): boolean {
  const lower = text.toLowerCase();
  return (
    lower.includes("rate limit") ||
    lower.includes("too many requests") ||
    lower.includes("429") ||
    lower.includes("resource exhausted") ||
    lower.includes("quota")
  );
}

function isLikelyModelUnavailable(text: string): boolean {
  const lower = text.toLowerCase();
  return (
    lower.includes("specified api usage limits") ||
    (lower.includes("regain access on") && lower.includes("usage limits")) ||
    lower.includes("model is currently overloaded") ||
    lower.includes("currently experiencing high demand")
  );
}

function isLikelyNetworkError(text: string): boolean {
  const lower = text.toLowerCase();
  return (
    lower.includes("failed to fetch") ||
    lower.includes("network") ||
    lower.includes("connection refused") ||
    lower.includes("econnrefused") ||
    lower.includes("econnreset") ||
    lower.includes("socket hang up")
  );
}

function isLikelyTimeout(text: string): boolean {
  const lower = text.toLowerCase();
  return (
    lower.includes("timeout") ||
    lower.includes("timed out") ||
    lower.includes("etimedout") ||
    lower.includes("request aborted")
  );
}

function isLikelyContextOverflow(text: string): boolean {
  const lower = text.toLowerCase();
  return (
    lower.includes("prompt is too long") ||
    lower.includes("failed to buffer the request body") ||
    lower.includes("input too long") ||
    lower.includes("input too large") ||
    lower.includes("body too long") ||
    lower.includes("body too large") ||
    lower.includes("request length exceeded") ||
    lower.includes("request body too large") ||
    lower.includes("token count exceed") ||
    lower.includes("context_length_exceeded") ||
    lower.includes("context window") ||
    lower.includes("maximum number of tokens allowed") ||
    lower.includes("token limit") ||
    lower.includes("context length") ||
    lower.includes("maximum context length") ||
    (lower.includes("exceeds") && lower.includes("context")) ||
    (lower.includes("too long") && lower.includes("token"))
  );
}

function isLikelyUnsupportedFileType(text: string): boolean {
  const lower = text.toLowerCase();
  return (
    lower.includes("unsupported mime type") ||
    lower.includes("unsupported media type") ||
    lower.includes("unsupported file type") ||
    lower.includes("don't support media type") ||
    lower.includes("does not support media type") ||
    lower.includes("doesn't support media type") ||
    lower.includes("don't support the following media type") ||
    lower.includes("does not support the following media type") ||
    lower.includes("doesn't support the following media type")
  );
}

function isLikelyImageTooLarge(text: string): boolean {
  return /image dimensions? exceed max allowed size/i.test(text);
}

export function createClassifiedError(kind: ErrorKind, original: unknown, status?: number): ClassifiedError {
  return { kind, status, original };
}

/**
 * Classifies an error into a known category to determine retry behavior.
 * This is the heuristic-based classifier used when adapters don't provide
 * their own provider-specific classification.
 *
 * @param error - The error to classify (can be any type)
 * @param status - Optional HTTP status code if known from context
 * @returns Classification result with error kind and retry action
 */
export function classifyError(error: unknown, status?: number): ClassifiedError {
  const normalizedError = errObj(error);
  const message = errMessage(error);

  // Try to extract status from various sources
  if (typeof status !== "number") {
    status = extractStatusFromPayload(normalizedError);
  }

  if (typeof status !== "number") {
    status = extractStatusFromPayload(tryParseJsonRecord(message));
  }

  if (typeof status !== "number") {
    const possibleStatusCode = message.match(/^(\d+)\s.+/);
    if (possibleStatusCode) status = Number(possibleStatusCode[1]);
  }

  let kind: ErrorKind = "unknown";

  if (normalizedError.name === "AbortError") {
    kind = "aborted";
  } else if (normalizedError.name === "TimeoutError") {
    kind = "timeout";
  } else if (
    normalizedError["networkError"] === true ||
    normalizedError.message === "Failed to fetch"
  ) {
    kind = "network";
  } else if (isLikelyModelUnavailable(message)) {
    kind = "model_unavailable";
  } else if (isLikelyUnsupportedFileType(message)) {
    kind = "unsupported_file_type";
  } else if (isLikelyImageTooLarge(message)) {
    kind = "image_too_large";
  } else if (isLikelyContextOverflow(message)) {
    kind = "context_overflow";
  } else if (isLikelyTimeout(message)) {
    kind = "timeout";
  } else if (typeof status === "number") {
    if (status === 413) {
      kind = "context_overflow";
    } else if (status === 429) {
      kind = message.toLowerCase().includes("quota exceeded") ? "quota_exceeded" : "rate_limit";
    } else if (status === 401 || status === 403) {
      kind = "auth";
    } else if (status === 504) {
      kind = "timeout";
    } else if (status === 503 || status === 529) {
      kind = "model_unavailable";
    } else if (status >= 400 && status < 500) {
      kind = "client";
    } else if (status >= 500) {
      kind = "server";
    }
  } else if (isLikelyProviderRateLimit(message)) {
    kind = "rate_limit";
  } else if (isLikelyNetworkError(message)) {
    kind = "network";
  }

  return createClassifiedError(kind, error, status);
}
