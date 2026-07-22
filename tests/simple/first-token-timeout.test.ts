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
