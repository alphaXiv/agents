import { assertEquals } from "@std/assert";
import z from "zod";
import { Agent, cli, type CliIo, Tool } from "../../mod.ts";
import { Adapter, type AdapterStreamOptions } from "../../src/adapters/adapter.ts";
import { Model } from "../../src/adapters/model.ts";
import { convertChatItemsToStream } from "../../src/client.ts";
import type { AdapterStreamIterator, ChatItem } from "../../src/types.ts";

class FakeIo implements CliIo {
  prompts: string[] = [];
  writes: string[] = [];
  #inputs: (string | null)[];
  #interruptHandler?: () => void;
  #pendingReadLineResolve?: (value: string | null) => void;

  constructor(inputs: (string | null)[]) {
    this.#inputs = [...inputs];
  }

  readLine(prompt: string): Promise<string | null> {
    this.prompts.push(prompt);
    const next = this.#inputs.shift();
    if (next !== undefined) {
      return Promise.resolve(next);
    }

    return new Promise((resolve) => {
      this.#pendingReadLineResolve = resolve;
    });
  }

  write(text: string): void {
    this.writes.push(text);
  }

  onInterrupt(handler: () => void): () => void {
    this.#interruptHandler = handler;
    return () => {
      if (this.#interruptHandler === handler) {
        this.#interruptHandler = undefined;
      }
    };
  }

  close(): void {
    this.#pendingReadLineResolve?.(null);
    this.#pendingReadLineResolve = undefined;
  }

  triggerInterrupt(): void {
    this.#interruptHandler?.();
  }
}

class ReplTestAdapter extends Adapter<"repl-test"> {
  name = "repl-test";

  stream<zO, zI>(options: AdapterStreamOptions<zO, zI>): AdapterStreamIterator {
    const userInputs = options.history.filter((item): item is Extract<ChatItem, { type: "input_text" }> =>
      item.type === "input_text"
    );
    const lastInput = userInputs.at(-1);
    if (!lastInput) {
      throw new Error("Expected at least one input_text item");
    }

    return convertChatItemsToStream({
      items: [{
        type: "output_text",
        content: `turn ${userInputs.length}: ${lastInput.content}`,
      }],
      inputTokens: 0,
      outputTokens: 0,
    });
  }
}

class ReplTestModel extends Model<"repl-test"> {
  adapter: ReplTestAdapter;

  constructor() {
    super({ model: "repl-test" });
    this.adapter = new ReplTestAdapter({ model: "repl-test" });
  }
}

class StatusReplAdapter extends Adapter<"repl-status-test"> {
  name = "repl-status-test";

  async *stream<zO, zI>(options: AdapterStreamOptions<zO, zI>): AdapterStreamIterator {
    const lastItem = options.history.at(-1);
    if (!lastItem) {
      throw new Error("Expected conversation history");
    }

    if (lastItem.type === "input_text") {
      const lookupTool = options.tools[0];
      if (!lookupTool) {
        throw new Error("Expected at least one tool");
      }

      yield {
        type: "delta_output_reasoning",
        delta: "Let me look that up.",
        index: 0,
      };
      yield {
        type: "tool_use_start",
        tool_use_id: "tool-1",
        kind: lookupTool.normalizedName,
        index: 1,
      };
      yield {
        type: "tool_use",
        tool_use_id: "tool-1",
        kind: lookupTool.normalizedName,
        content: '"hello"',
        index: 1,
      };
      return { inputTokens: 0, outputTokens: 0 };
    }

    if (lastItem.type === "tool_result_text") {
      yield {
        type: "delta_output_text",
        delta: `final: ${lastItem.content}`,
        index: 0,
      };
      return { inputTokens: 0, outputTokens: 0 };
    }

    throw new Error(`Unexpected history item: ${lastItem.type}`);
  }
}

class StatusReplModel extends Model<"repl-status-test"> {
  adapter: StatusReplAdapter;

  constructor() {
    super({ model: "repl-status-test" });
    this.adapter = new StatusReplAdapter({ model: "repl-status-test" });
  }
}

Deno.test("cli streams agent responses and exits on command", async () => {
  const io = new FakeIo(["hello", "/exit"]);
  const agent = new Agent({
    model: new ReplTestModel(),
    instructions: "Reply deterministically.",
  });

  await cli(agent, {
    io,
    greeting: false,
  });

  assertEquals(io.prompts, ["> ", "> "]);
  assertEquals(io.writes.join(""), "[responding]\nturn 1: hello\n");
});

Deno.test("cli reset clears conversation history", async () => {
  const io = new FakeIo(["first", "second", "/reset", "third", "/exit"]);
  const agent = new Agent({
    model: new ReplTestModel(),
    instructions: "Reply deterministically.",
  });

  await cli(agent, {
    io,
    prompt: "agent> ",
    greeting: "ready",
  });

  assertEquals(io.prompts, ["agent> ", "agent> ", "agent> ", "agent> ", "agent> "]);
  assertEquals(
    io.writes.join(""),
    "ready\n[responding]\nturn 1: first\n[responding]\nturn 2: second\nHistory cleared.\n[responding]\nturn 1: third\n",
  );
});

Deno.test("cli shows thinking, tool, and responding indicators", async () => {
  const io = new FakeIo(["hello", "/exit"]);
  const lookupTool = new Tool({
    name: "lookup",
    description: "Look up a value",
    parameters: z.string(),
    execute: (param) => `result for ${param}`,
  });
  const agent = new Agent({
    model: new StatusReplModel(),
    instructions: "Reply deterministically.",
    tools: [lookupTool],
  });

  await cli(agent, {
    io,
    greeting: false,
  });

  assertEquals(
    io.writes.join(""),
    "[thinking]\n[using tool: lookup]\n[tool finished: lookup]\n[responding]\nfinal: result for hello\n",
  );
});

Deno.test("cli exits cleanly on ctrl+c while waiting for input", async () => {
  const io = new FakeIo([]);
  const agent = new Agent({
    model: new ReplTestModel(),
    instructions: "Reply deterministically.",
  });

  const replPromise = cli(agent, {
    io,
    greeting: false,
  });

  queueMicrotask(() => io.triggerInterrupt());
  await replPromise;

  assertEquals(io.prompts, ["> "]);
  assertEquals(io.writes.join(""), "\n");
});
