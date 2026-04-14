import type z from "zod";
import type { ClassifiedError } from "../errors.ts";
import type { AnyTool } from "../tool.ts";
import type { AdapterStreamIterator, ChatItem } from "../types.ts";

export interface AdapterOptions<SupportedModels extends string> {
  model: SupportedModels;
}

export interface AdapterStreamOptions<zO, zI> {
  /** Structured output schema */
  output?: z.ZodType<zO, zI>;
  /**
   * Available tool definition
   * Do not call the tool implementations yourself.
   */
  tools: AnyTool[];
  /** Primary instructions / developer prompt / system prompt */
  instructions: string;
  /** Previous conversation history */
  history: ChatItem[];
  /** Cancellation signal */
  signal: AbortSignal;
}

export abstract class Adapter<SupportedModels extends string> {
  abstract name: string;
  model: SupportedModels;

  constructor(options: AdapterOptions<SupportedModels>) {
    this.model = options.model;
  }

  abstract stream<zO, zI>(options: AdapterStreamOptions<zO, zI>): AdapterStreamIterator;

  /**
   * Classify an error using provider-specific error types.
   * Override this in adapter implementations to provide precise error classification.
   *
   * @experimental Might be removed or have its behaviour modified without any notice
   * @param error The error thrown during a model call
   * @returns ClassifiedError if the adapter can classify it, null to fall back to heuristics
   */
  classifyError?(error: unknown): ClassifiedError | null;
}
