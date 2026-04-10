import { assert, assertEquals, assertRejects } from "@std/assert";
import z from "zod";
import { Agent, Tool } from "../../mod.ts";
import { Adapter, type AdapterStreamOptions } from "../../src/adapters/adapter.ts";
import { Model } from "../../src/adapters/model.ts";
import { convertChatItemsToStream } from "../../src/client.ts";
import type { AdapterStreamIterator } from "../../src/types.ts";

class ToolSignalTestAdapter extends Adapter<"tool-signal-model"> {
  name = "tool-signal-test";
  #streamImpl: <zO, zI>(options: AdapterStreamOptions<zO, zI>) => AdapterStreamIterator;

  constructor(stream: <zO, zI>(options: AdapterStreamOptions<zO, zI>) => AdapterStreamIterator) {
    super({ model: "tool-signal-model" });
    this.#streamImpl = stream;
  }

  stream<zO, zI>(options: AdapterStreamOptions<zO, zI>): AdapterStreamIterator {
    return this.#streamImpl(options);
  }
}

class ToolSignalTestModel extends Model<"tool-signal-model"> {
  adapter: ToolSignalTestAdapter;

  constructor(stream: <zO, zI>(options: AdapterStreamOptions<zO, zI>) => AdapterStreamIterator) {
    super({ model: "tool-signal-model" });
    this.adapter = new ToolSignalTestAdapter(stream);
  }
}

function createToolLoopModel() {
  return new ToolSignalTestModel(({ history, tools }) => {
    const last = history.at(-1);
    if (!last || last.type === "input_text") {
      const tool = tools[0];
      if (!tool) {
        throw new Error("Missing tool");
      }
      return convertChatItemsToStream({
        items: [{
          type: "tool_use",
          tool_use_id: "tool-1",
          kind: tool.name,
        }],
        inputTokens: 0,
        outputTokens: 0,
      });
    }

    if (last.type !== "tool_result_text") {
      throw new Error(`Unexpected history item: ${last.type}`);
    }

    return convertChatItemsToStream({
      items: [{ type: "output_text", content: last.content }],
      inputTokens: 0,
      outputTokens: 0,
    });
  });
}

Deno.test("tool-specific abort signals cancel the tool without aborting the agent", async () => {
  let sawAbort = false;

  const slowTool = new Tool({
    name: "slow_tool",
    description: "Waits for either completion or cancellation",
    parameters: z.void(),
    signal: AbortSignal.timeout(10),
    execute: (_, { signal }) => {
      return new Promise<string>((resolve, reject) => {
        const timeout = setTimeout(() => resolve("late success"), 200);
        const onAbort = () => {
          sawAbort = true;
          clearTimeout(timeout);
          reject(signal.reason);
        };

        if (signal.aborted) {
          onAbort();
          return;
        }

        signal.addEventListener("abort", onAbort, { once: true });
      });
    },
  });

  const agent = new Agent({
    model: createToolLoopModel(),
    instructions: "Use the tool once.",
    tools: [slowTool],
  });

  const run = await agent.run("go");

  assert(sawAbort);
  assertEquals(run.outputText.startsWith("Error:"), true);
  assertEquals(run.history[1]?.type, "tool_result_text");
});

Deno.test("agent abort propagates into a running tool", async () => {
  let sawAbort = false;

  const slowTool = new Tool({
    name: "slow_tool",
    description: "Waits for cancellation",
    parameters: z.void(),
    execute: (_, { signal }) => {
      return new Promise<string>((resolve, reject) => {
        const timeout = setTimeout(() => resolve("late success"), 200);
        const onAbort = () => {
          sawAbort = true;
          clearTimeout(timeout);
          reject(signal.reason);
        };

        if (signal.aborted) {
          onAbort();
          return;
        }

        signal.addEventListener("abort", onAbort, { once: true });
      });
    },
  });

  const agent = new Agent({
    model: createToolLoopModel(),
    instructions: "Use the tool once.",
    tools: [slowTool],
  });

  const controller = new AbortController();
  setTimeout(() => controller.abort(), 10);

  await assertRejects(
    () => agent.run("go", { signal: controller.signal }),
    DOMException,
  );

  assert(sawAbort);
});
