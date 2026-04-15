import OpenAI from "openai";
import { type ClassifiedError, createClassifiedError } from "../../errors.ts";

export function classifyOpenAIError(error: unknown): ClassifiedError | null {
  if (error instanceof OpenAI.APIConnectionTimeoutError) {
    return createClassifiedError("timeout", error, error.status);
  } else if (error instanceof OpenAI.APIConnectionError) {
    return createClassifiedError("network", error, error.status);
  } else if (error instanceof OpenAI.RateLimitError) {
    return createClassifiedError("rate_limit", error, error.status);
  } else if (error instanceof OpenAI.AuthenticationError) {
    return createClassifiedError("auth", error, error.status);
  } else if (error instanceof OpenAI.PermissionDeniedError) {
    return createClassifiedError("auth", error, error.status);
  } else if (error instanceof OpenAI.InternalServerError) {
    return createClassifiedError("server", error, error.status);
  } else if (error instanceof OpenAI.APIUserAbortError) {
    return createClassifiedError("aborted", error, error.status);
  }
  return null;
}
