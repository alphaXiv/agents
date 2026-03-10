import type { ChatItem } from "@alphaxiv/agents";
import { generate } from "@std/uuid/v7";
import { AsyncLocalStorage } from "node:async_hooks";
import { errMessage } from "./util.ts";

export interface Tracer {
  /**
   * Recieve start metadata. Allows you to represent pending traces in your DB.
   * Note that `log` traces do not go through this path.
   */
  start?: (event: PartialTraceEvent) => void;
  /** Recieve full event metadata */
  event: (event: TraceEvent) => void;
}

/**
 * TraceEvent is designed to be stored in a database, this is why some names are
 * suboptimal, we're trying to overlap as much as possible. Additionally, it is
 * designed in a way where you can trivially join the traces table on user chat
 * messages.
 */
export type TraceEvent =
  | AgentTraceEvent
  | ModelTraceEvent
  | ToolTraceEvent
  | MessageTraceEvent
  | LogTraceEvent;
export type PartialTraceEvent = Pick<
  TraceEvent,
  "id" | "type" | "parent" | "start" | "content"
>;

export type TraceType =
  | "agent"
  | "model"
  | "message"
  | "tool"
  | "log";

/**
 * JSON serializable. Metadata depends on the type of trace. Allows you to extract
 * some piece of data in a type safe way (token counts from model runs, tool names).
 */
export type TraceContent<T extends TraceType = TraceType> = Extract<
  TraceEvent,
  { type: T }
>["content"];

export interface BaseTraceEvent {
  /** Generated UUIDv7 */
  id: string;
  type: TraceType;
  /** Connects this to a parent trace event. */
  parent: string | null;
  /** The millisecond timestamp this span started. This time is also encoded in the `id` */
  start: number;
  /** The millisecond timestamp this span ended. */
  end: number;
  /** If not null, indicates the operation errored with the following message */
  errorMessage: null | string;
  /** The original error object if your backend supports rich errors */
  errorObject: unknown;

  /**
   * JSON serializable. Metadata depends on the type of trace. Allows you to extract
   * some piece of data in a type safe way (token counts from model runs, tool names).
   */
  content: string | Record<string, unknown>;
}

/** An invocation to `agent.run` or `agent.stream`. This is the root node for traces. */
export interface AgentTraceEvent extends BaseTraceEvent {
  type: "agent";
  content: {
    provider: string;
    model: string;
  };
}

/**
 * An invocation to a model provider. Happens many times per `agent` trace, for
 * example when tools resolve a second model run is done for you. Or, if the SDK
 * performs a retry, both runs are visible in the trace.
 */
export interface ModelTraceEvent extends BaseTraceEvent {
  type: "model";
  content: {
    /** The reason this model invocation happened. */
    reason: string;
    /** The name of the provider used */
    provider: string;
    /** The model given to the provider */
    model: string;
    /** Total count of input tokens. If the provider cannot provide this info, `null` */
    inputTokens: number | null;
    /** Total count of output tokens. If the provider is unable to classify, then all tokens are "output" tokens. */
    outputTokens: number | null;
  };
}

/**
 * A message from the LLM model. These are measured from the first stream event
 * seen until the start of the next message, or by the end of the stream. A
 * provider may be able to indicate percise end times, but most do not.
 */
export interface MessageTraceEvent extends BaseTraceEvent {
  type: "message";
  content: {
    /** The name of the tool */
    type: ChatItem["type"];
    // (get the rest of your data by joining on your ChatItem table)
  };
}

/** An invocation of a tool. */
interface ToolTraceEvent extends BaseTraceEvent {
  type: "tool";
  content: {
    /** The name of the tool */
    name: string;
    // (get the rest of your data by joining on your ChatItem table)
  };
}

/**
 * An log from the Agents SDK or the user. These can contain extra info on why a
 * model was retried when `reason` isn't enough, or when certain provider
 * workarounds happened.
 */
interface LogTraceEvent extends BaseTraceEvent {
  type: "log";
  content: string;
}

const globalTracers = new Set<Tracer>();

/**
 * Register a tracer to be added to every invocation of `Agent.run`. Unregister
 * by calling the cleanup callback. After de-registration, you'll still get more
 * trace events until the end of the model invocation.
 */
export function registerGlobalTracer(x: Tracer): () => void {
  globalTracers.add(x);
  return () => globalTracers.delete(x);
}

/** Allows tracing sub-agents without manual instrumentation. */
export const tracerAsyncLocalStorage = new AsyncLocalStorage<TraceRef>();

export interface TraceRef {
  id: string;
  parentTracers?: Tracer[];
}

interface TraceInit<T extends TraceType> {
  id?: string;
  type: T;
  start?: number | null;
  parent?: TraceRef | null;
  tracers?: Tracer[];
  content: TraceContent<T>;
}

export interface ActiveTrace<T extends Exclude<TraceType, "log">>
  extends TraceRef {
  resolved: boolean;
  error(err: unknown, content?: Partial<TraceContent<T>>): void;
  success(content?: Partial<TraceContent<T>>): void;
  log(message: string, error?: unknown): void;
}

/** Use `try` / `finally` to call either `success` or `error` */
export function newTrace<T extends Exclude<TraceType, "log">>(
  init: TraceInit<T>,
): ActiveTrace<T> {
  const start = init.start ?? Date.now();
  let { content, type } = init;
  const ref = init.parent ?? tracerAsyncLocalStorage.getStore();
  const parent = ref?.id ?? null;
  const id = init.id ?? generate(start);

  const tracers = [
    ...new Set([
      ...ref?.parentTracers ?? [],
      ...init.tracers ?? [],
      ...globalTracers,
    ]),
  ];

  // no-op tracer implementation
  let resolved: boolean;
  if (tracers.length === 0) {
    return {
      id,
      parentTracers: [],
      get resolved() {
        return resolved;
      },
      error() {
        resolved = true;
      },
      success() {
        resolved = true;
      },
      log() {},
    };
  }

  // actually trace
  tracers.forEach((t) =>
    t.start?.({ id, type, parent, start, content } as PartialTraceEvent)
  );
  return {
    id,
    parentTracers: tracers,
    get resolved() {
      return resolved;
    },
    error(err, finalContent) {
      if (finalContent != null) {
        if (typeof finalContent === "string") content = finalContent;
        content = { ...content, ...finalContent } as TraceContent<T>;
      }
      resolved = true;
      const event = {
        id,
        type,
        parent,
        start,
        end: Date.now(),
        errorMessage: errMessage(err),
        errorObject: err,
        content,
      } satisfies BaseTraceEvent as TraceEvent;
      tracers.forEach((t) => t.event(event));
    },
    success(finalContent) {
      if (finalContent != null) {
        if (typeof finalContent === "string") content = finalContent;
        content = { ...content, ...finalContent } as TraceContent<T>;
      }
      resolved = true;
      const event = {
        id,
        type,
        parent,
        start,
        end: Date.now(),
        errorMessage: null,
        errorObject: null,
        content,
      } satisfies BaseTraceEvent as TraceEvent;
      tracers.forEach((t) => t.event(event));
    },
    log(content, error) {
      const time = Date.now();
      const event: TraceEvent = {
        id: generate(time),
        type: "log",
        parent: id,
        start: time,
        end: time,
        errorMessage: error != null ? errMessage(error) : null,
        errorObject: error ?? null,
        content,
      };
      tracers.forEach((t) => t.event(event));
    },
  };
}
