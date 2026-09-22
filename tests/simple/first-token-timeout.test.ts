import { assert, assertEquals } from "@std/assert";
import { Agent, type StreamItem } from "../../mod.ts";
import type { Adapter } from "../../src/adapters/adapter.ts";
import { deterministicTestModel } from "./testing-model.ts";

/** A model that withholds its first token for `delayMs`, aborting cleanly when its signal fires. */
function slowFirstTokenModel(delayMs: number, provider = "slow"): Adapter<unknown, unknown> {
  return {
    provider,
    model: provider,
    stream(options) {
      return (async function* () {
        await new Promise<void>((resolve, reject) => {
          const timer = setTimeout(resolve, delayMs);
          options.signal.addEventListener("abort", () => {
            clearTimeout(timer);
            reject(options.signal.reason ?? new Error("aborted"));
          }, { once: true });
        });
        yield { type: "delta_output_text" as const, index: 0, delta: "late reply" };
        return { inputTokens: 0, outputTokens: 0 };
      })();
    },
  };
}

/**
 * A model that waits `prepareMs` before the provider call (as file uploads do), optionally
 * yields `request_start`, then waits `respondMs` before its first token.
 */
function preparingModel(
  options: { prepareMs: number; respondMs: number; announce: boolean },
  provider = "preparing",
): Adapter<unknown, unknown> {
  return {
    provider,
    model: provider,
    stream(streamOptions) {
      const wait = (ms: number) =>
        new Promise<void>((resolve, reject) => {
          const timer = setTimeout(resolve, ms);
          streamOptions.signal.addEventListener("abort", () => {
            clearTimeout(timer);
            reject(streamOptions.signal.reason ?? new Error("aborted"));
          }, { once: true });
        });

      return (async function* () {
        await wait(options.prepareMs);
        if (options.announce) yield { type: "request_start" as const, index: 0 };
        await wait(options.respondMs);
        yield { type: "delta_output_text" as const, index: 0, delta: "late reply" };
        return { inputTokens: 0, outputTokens: 0 };
      })();
    },
  };
}

async function collect(agent: Agent, input: string): Promise<StreamItem[]> {
  const items: StreamItem[] = [];
  for await (const item of agent.stream(input)) items.push(item);
  return items;
}

Deno.test("rolls over to the next model when the first token stalls past the timeout", async () => {
  const agent = new Agent({
    model: [slowFirstTokenModel(1000), deterministicTestModel()],
    instructions: "You are a friendly assistant",
    retryStrategy: { firstTokenTimeoutMs: 20, sameModelRetries: 0 },
  });

  const items = await collect(agent, "Hello!");

  const switchEvent = items.find((i) => i.type === "model_switched");
  assert(switchEvent && switchEvent.type === "model_switched", "should switch models on stall");
  assertEquals(switchEvent.from.provider, "slow");
  assertEquals(switchEvent.to.provider, "deterministic");
  assert(switchEvent.classified?.kind === "timeout", "stall must classify as a timeout");

  const text = items.filter((i) => i.type === "delta_output_text").map((i) => i.delta).join("");
  assert(!text.includes("late reply"), "the stalled model's output must not leak through");
});

Deno.test("does not arm the watchdog when only one model is configured", async () => {
  const agent = new Agent({
    model: slowFirstTokenModel(60),
    instructions: "You are a friendly assistant",
    retryStrategy: { firstTokenTimeoutMs: 20, sameModelRetries: 0 },
  });

  const items = await collect(agent, "Hello!");
  const text = items.filter((i) => i.type === "delta_output_text").map((i) => i.delta).join("");
  assertEquals(text, "late reply");
});

Deno.test("does not arm the watchdog on the final model of the final cycle", async () => {
  const failing: Adapter<unknown, unknown> = {
    provider: "boom",
    model: "boom",
    stream() {
      throw new Error("Deterministic Provider Error");
    },
  };

  const agent = new Agent({
    model: [failing, slowFirstTokenModel(60)],
    instructions: "x",
    retryStrategy: { firstTokenTimeoutMs: 20, sameModelRetries: 0 },
  });

  const items = await collect(agent, "Hello!");
  const text = items.filter((i) => i.type === "delta_output_text").map((i) => i.delta).join("");
  assertEquals(text, "late reply", "final model must be allowed to run past the timeout");
});

Deno.test("firstTokenTimeoutMs: 0 disables the watchdog", async () => {
  const agent = new Agent({
    model: [slowFirstTokenModel(60), deterministicTestModel()],
    instructions: "You are a friendly assistant",
    retryStrategy: { firstTokenTimeoutMs: 0, sameModelRetries: 0 },
  });

  const items = await collect(agent, "Hello!");
  assert(!items.some((i) => i.type === "model_switched"), "no switch when disabled");
  const text = items.filter((i) => i.type === "delta_output_text").map((i) => i.delta).join("");
  assertEquals(text, "late reply");
});

Deno.test("request_start restarts the watchdog so preparation gets its own budget", async () => {
  const agent = new Agent({
    model: [
      preparingModel({ prepareMs: 60, respondMs: 60, announce: true }),
      deterministicTestModel(),
    ],
    instructions: "You are a friendly assistant",
    retryStrategy: { firstTokenTimeoutMs: 100, sameModelRetries: 0 },
  });

  const items = await collect(agent, "Hello!");

  assert(!items.some((i) => i.type === "model_switched"), "restarted watchdog must not switch models");
  const text = items.filter((i) => i.type === "delta_output_text").map((i) => i.delta).join("");
  assertEquals(text, "late reply");
});

Deno.test("without request_start the same preparation trips the watchdog", async () => {
  const agent = new Agent({
    model: [
      preparingModel({ prepareMs: 60, respondMs: 60, announce: false }),
      deterministicTestModel(),
    ],
    instructions: "You are a friendly assistant",
    retryStrategy: { firstTokenTimeoutMs: 100, sameModelRetries: 0 },
  });

  const items = await collect(agent, "Hello!");

  const switchEvent = items.find((i) => i.type === "model_switched");
  assert(switchEvent && switchEvent.type === "model_switched", "should switch models on stall");
  assertEquals(switchEvent.from.provider, "preparing");
  assert(switchEvent.classified?.kind === "timeout", "stall must classify as a timeout");
});

Deno.test("request_start does not excuse a provider that then stalls", async () => {
  const agent = new Agent({
    model: [
      preparingModel({ prepareMs: 10, respondMs: 1000, announce: true }),
      deterministicTestModel(),
    ],
    instructions: "You are a friendly assistant",
    retryStrategy: { firstTokenTimeoutMs: 20, sameModelRetries: 0 },
  });

  const items = await collect(agent, "Hello!");

  const switchEvent = items.find((i) => i.type === "model_switched");
  assert(switchEvent && switchEvent.type === "model_switched", "restarted watchdog must still fire");
  assertEquals(switchEvent.to.provider, "deterministic");
  const text = items.filter((i) => i.type === "delta_output_text").map((i) => i.delta).join("");
  assert(!text.includes("late reply"), "the stalled model's output must not leak through");
});
