import { generate } from "@std/uuid/v7";
import { AsyncLocalStorage } from "node:async_hooks";
import { errMessage } from "./util.ts";
import type { ChatItem } from "./types.ts";

export interface Tracer {
  /**
   * Receive start metadata. Allows you to represent pending traces in your DB.
   * Note that `log` traces do not go through this path.
   */
  start?: (event: PartialTraceEvent) => void;
  /** Receive full event metadata */
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
  | LogTraceEvent
  | CustomTraceEvent;

type PartialTraceKeys = "id" | "type" | "parent" | "start" | "content";
export type PartialTraceEvent =
  | (Pick<AgentTraceEvent, PartialTraceKeys> & Partial<AgentTraceEvent>)
  | (Pick<ModelTraceEvent, PartialTraceKeys> & Partial<ModelTraceEvent>)
  | (Pick<ToolTraceEvent, PartialTraceKeys> & Partial<ToolTraceEvent>)
  | (Pick<MessageTraceEvent, PartialTraceKeys> & Partial<MessageTraceEvent>)
  | (Pick<LogTraceEvent, PartialTraceKeys> & Partial<LogTraceEvent>)
  | (Pick<CustomTraceEvent, PartialTraceKeys> & Partial<CustomTraceEvent>);

export type TraceType =
  | "agent"
  | "model"
  | "message"
  | "tool"
  | "log"
  | "custom";

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
  content: Record<string, unknown>;
}

/** An invocation to `agent.run` or `agent.stream`. This is the root node for traces. */
export interface AgentTraceEvent extends BaseTraceEvent {
  type: "agent";
  content: {
    name?: string;
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
 * provider may be able to indicate precise end times, but most do not.
 */
export interface MessageTraceEvent extends BaseTraceEvent {
  type: "message";
  content: {
    type: ChatItem["type"];
    // (get the rest of your data by joining on your ChatItem table)
  };
}

/** An invocation of a tool. */
export interface ToolTraceEvent extends BaseTraceEvent {
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
export interface LogTraceEvent extends BaseTraceEvent {
  type: "log";
  content: { message: string };
}

/**
 * An custom span from the user. This can be used by an application to group
 * larger orchestrations into a shared trace. For example, you could wrap two
 * agent executions in a row.
 */
export interface CustomTraceEvent extends BaseTraceEvent {
  type: "custom";
  /**
   * If using object form, the convention is to at least include a `label`
   * property to label this entry in flamegraphs.
   */
  content: {
    label?: string;
    [extra: string]: unknown;
  };
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
  error(err: unknown, content?: Partial<TraceContent<T>>): void;
  success(content?: Partial<TraceContent<T>>): void;
  log(message: string, error?: unknown): void;
  [Symbol.dispose](): void;
}

/**
 * @internal Create any trace type without requiring a function callback. There
 * are probably places where the callback approach is better, but for things
 * like `MessageTracer` this primitive is required.

 * Use a `catch` block to call `error`, `content` can also be edited upon calling success.
 */
export function newTrace<T extends Exclude<TraceType, "log">>(
  init: TraceInit<T>,
): ActiveTrace<T> {
  const start = init.start ?? Date.now();
  let { content, type } = init;
  const ref = init.parent ?? tracerAsyncLocalStorage.getStore();
  const parent = ref?.id ?? null;
  const id = init.id ?? generate(start);

  // discover any new global tracers alongside parent attached tracers.
  const tracers = Array.from(
    new Set([
      ...ref?.parentTracers ?? [],
      ...init.tracers ?? [],
      ...globalTracers,
    ]),
  );

  let resolved: boolean = false;
  tracers.forEach((t) =>
    t.start?.({ id, type, parent, start, content } as PartialTraceEvent)
  );
  return {
    id,
    parentTracers: tracers,
    error(err, finalContent) {
      if (resolved) return;
      if (finalContent != null) {
        if (typeof finalContent === "string") content = finalContent;
        else {content = {
            ...content as object,
            ...finalContent,
          } as TraceContent<T>;}
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
      if (resolved) return;
      if (finalContent != null) {
        if (typeof finalContent === "string") content = finalContent;
        else {content = {
            ...content as object,
            ...finalContent,
          } as TraceContent<T>;}
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
    [Symbol.dispose]() {
      this.success();
    },
    log(message, error) {
      const time = Date.now();
      const event: TraceEvent = {
        id: generate(time),
        type: "log",
        parent: id,
        start: time,
        end: time,
        errorMessage: error != null ? errMessage(error) : null,
        errorObject: error ?? null,
        content: { message },
      };
      tracers.forEach((t) => t.event(event));
    },
  };
}

export type ActiveCustomTrace = Pick<ActiveTrace<"custom">, "id" | "log">;

/**
 * Create a custom span. This can be used by your application to group larger
 * orchestrations into a shared trace. For example, you could wrap two agent
 * executions in a row. Additionally, this gives you access to emit log events
 * within your callback. Custom traces automatically propagate using the Async
 * Context API.
 */
export async function withTrace<T>(
  content: string | TraceContent<"custom">,
  callback: (ref: ActiveCustomTrace) => Promise<T>,
): Promise<T> {
  using trace = newTrace({
    type: "custom",
    content: typeof content === "string" ? { label: content } : content,
  });
  try {
    return await tracerAsyncLocalStorage.run(trace, () => callback(trace));
  } catch (err) {
    trace.error(err);
    throw err;
  }
}

/**
 * Stateful tracking of message trace objects. Since adapters are usually not
 * capable of knowing when a message ends (outside of tools), the end times have
 * to be inferred based on the start of a next item. Since only one item is
 * active at once, that makes this pretty simple: just restart the trace if the
 * index changes.
 *
 * This use-case is why the lower level `newTrace` function does not enforce
 * callback wrapping.
 */
export class MessageTracer {
  current: {
    trace: ActiveTrace<"message">;
    index: number;
  } | null = null;
  modelTrace: TraceRef;

  constructor(modelTrace: TraceRef) {
    this.modelTrace = modelTrace;
  }

  startOrContinue({ index, type }: { index: number; type: ChatItem["type"] }) {
    // if the index is different than the one we're locked in on tracing, replace it
    if (!this.current || this.current.index !== index) {
      this.endMessageTraceIfStarted();
      const trace = newTrace({
        type: "message",
        parent: this.modelTrace,
        content: { type },
      });
      this.current = { trace, index };
    }
    return this.current.trace.id;
  }

  endMessageTraceIfStarted() {
    if (this.current) this.current.trace.success();
    this.current = null;
  }

  cancel() {
    if (this.current) {
      this.current.trace.error(new Error("Cancelled by provider error"));
    }
    this.current = null;
  }

  [Symbol.dispose]() {
    this.endMessageTraceIfStarted();
  }
}
