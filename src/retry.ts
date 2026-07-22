import type { ClassifiedError, ErrorKind } from "./errors.ts";

/** Retry behavior options for different error types. */
export type RetryBehavior = "retry-same" | "switch-model" | "no-retry";

/**
 * Error kinds that are a deterministic property of the (model, request) pair
 * rather than a transient condition. Because an agent's history only grows, a
 * model that rejects the request this way will reject it identically on every
 * later turn, so it should be skipped for the rest of the run instead of being
 * re-attempted (and re-failed) as the primary each turn.
 */
const DETERMINISTIC_ERROR_KINDS = new Set<ErrorKind>([
  "client",
  "unsupported_file_type",
  "image_too_large",
]);

export function isDeterministicModelError(kind: ErrorKind): boolean {
  return DETERMINISTIC_ERROR_KINDS.has(kind);
}

/** Configuration for retry behavior. All fields are optional with sensible defaults. */
export interface RetryStrategy {
  /**
   * How many times to retry transient errors (network issues, server errors)
   * on the SAME model before switching to the next one.
   * @default 2
   */
  sameModelRetries?: number;

  /**
   * How many complete cycles through all models to attempt.
   * A cycle means trying each model once (after exhausting same-model retries).
   * @default 1
   */
  modelCycles?: number;

  /**
   * If the current model produces no tokens within this many milliseconds, treat it
   * as a timeout and fall back to the next model (honoring {@link onTimeout}).
   *
   * Only armed when a fallback is actually available: more than one model is configured
   * and this is not the final model of the final cycle. The last model is allowed to take
   * as long as it needs, since rolling over would have nowhere to go.
   *
   * Set to `0` to disable the watchdog entirely.
   * @default 10000
   */
  firstTokenTimeoutMs?: number;

  /**
   * Behavior when a timeout is detected (model hung/slow).
   * @default 'switch-model'
   */
  onTimeout?: RetryBehavior;

  /**
   * Behavior when rate limited.
   * @default 'switch-model'
   */
  onRateLimit?: RetryBehavior;

  /**
   * Behavior when model is unavailable/overloaded.
   * @default 'switch-model'
   */
  onModelUnavailable?: RetryBehavior;

  /**
   * Behavior on network errors (connection issues, DNS failures).
   * @default 'retry-same'
   */
  onNetworkError?: RetryBehavior;

  /**
   * Behavior on server errors (5xx responses).
   * @default 'retry-same'
   */
  onServerError?: RetryBehavior;

  /**
   * Custom function to determine retry behavior for a classified error.
   * If provided, this is called first and can override default behavior.
   * Return null to fall through to default behavior.
   */
  customHandler?: (classified: ClassifiedError) => RetryBehavior | null;
}

/** Resolved retry strategy with all defaults applied. */
export interface ResolvedRetryStrategy {
  sameModelRetries: number;
  modelCycles: number;
  firstTokenTimeoutMs: number;
  onTimeout: RetryBehavior;
  onRateLimit: RetryBehavior;
  onModelUnavailable: RetryBehavior;
  onNetworkError: RetryBehavior;
  onServerError: RetryBehavior;
  customHandler?: (classified: ClassifiedError) => RetryBehavior | null;
}

/**
 * Default retry strategy values - optimized for reliability.
 *
 * These defaults are designed to:
 * - Retry transient errors (network blips) on the same model
 * - Switch models immediately for rate limits and timeouts
 * - Always try fallback for model-specific errors (different model might work)
 */
export const DEFAULT_RETRY_STRATEGY: ResolvedRetryStrategy = {
  sameModelRetries: 2,
  modelCycles: 1,
  firstTokenTimeoutMs: 10000,
  onTimeout: "switch-model",
  onRateLimit: "switch-model",
  onModelUnavailable: "switch-model",
  onNetworkError: "retry-same",
  onServerError: "retry-same",
};

/**
 * Resolves a partial retry strategy with defaults.
 */
export function resolveRetryStrategy(strategy?: RetryStrategy): ResolvedRetryStrategy {
  if (!strategy) return { ...DEFAULT_RETRY_STRATEGY };

  return {
    sameModelRetries: strategy.sameModelRetries ?? DEFAULT_RETRY_STRATEGY.sameModelRetries,
    modelCycles: strategy.modelCycles ?? DEFAULT_RETRY_STRATEGY.modelCycles,
    firstTokenTimeoutMs: strategy.firstTokenTimeoutMs ?? DEFAULT_RETRY_STRATEGY.firstTokenTimeoutMs,
    onTimeout: strategy.onTimeout ?? DEFAULT_RETRY_STRATEGY.onTimeout,
    onRateLimit: strategy.onRateLimit ?? DEFAULT_RETRY_STRATEGY.onRateLimit,
    onModelUnavailable: strategy.onModelUnavailable ?? DEFAULT_RETRY_STRATEGY.onModelUnavailable,
    onNetworkError: strategy.onNetworkError ?? DEFAULT_RETRY_STRATEGY.onNetworkError,
    onServerError: strategy.onServerError ?? DEFAULT_RETRY_STRATEGY.onServerError,
    customHandler: strategy.customHandler,
  };
}

/** Maps an error kind to the strategy-configured behavior. */
function getStrategyBehavior(kind: ErrorKind, strategy: ResolvedRetryStrategy): RetryBehavior {
  switch (kind) {
    case "aborted":
      return "no-retry";
    case "timeout":
      return strategy.onTimeout;
    case "rate_limit":
      return strategy.onRateLimit;
    case "model_unavailable":
      return strategy.onModelUnavailable;
    case "network":
      return strategy.onNetworkError;
    case "server":
    case "unknown":
      return strategy.onServerError;
    default:
      return "switch-model";
  }
}

/**
 * Determines the retry behavior for a classified error.
 * If retry-same retries are exhausted, escalates to switch-model.
 */
export function determineRetryBehavior(
  classified: ClassifiedError,
  strategy: ResolvedRetryStrategy,
  sameModelRetriesUsed: number,
): RetryBehavior {
  let behavior: RetryBehavior;

  if (strategy.customHandler) {
    const custom = strategy.customHandler(classified);
    behavior = custom !== null ? custom : getStrategyBehavior(classified.kind, strategy);
  } else {
    behavior = getStrategyBehavior(classified.kind, strategy);
  }

  if (behavior === "retry-same" && sameModelRetriesUsed >= strategy.sameModelRetries) {
    return "switch-model";
  }

  return behavior;
}
