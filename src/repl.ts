import process from "node:process";
import { createInterface, type Interface } from "node:readline/promises";
import type { Agent, AgentRunOptions } from "./agent.ts";
import type { AnyTool } from "./tool.ts";
import type { ChatItem } from "./types.ts";
import { errMessage } from "./util.ts";

const DEFAULT_PROMPT = "> ";
const DEFAULT_GREETING = [
  "Interactive agent REPL.",
  "Commands: /reset clears history, /exit quits.",
].join(" ");

export interface ReplIo {
  readLine(prompt: string): Promise<string | null>;
  write(text: string): void;
  onInterrupt?(handler: () => void): (() => void) | void;
  close?(): void;
}

export interface ReplOptions {
  prompt?: string;
  greeting?: string | false;
  runOptions?: AgentRunOptions;
  io?: ReplIo;
}

function createNodeReplIo(): ReplIo {
  const readline: Interface = createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: true,
  });

  return {
    async readLine(prompt: string): Promise<string | null> {
      try {
        return await readline.question(prompt);
      } catch (error) {
        if (error && typeof error === "object" && "code" in error && error.code === "ERR_READLINE_CLOSE") {
          return null;
        }
        throw error;
      }
    },
    write(text: string): void {
      process.stdout.write(text);
    },
    onInterrupt(handler: () => void): () => void {
      // FIXME(Im-Beast): This is still kinda janky i don't like it
      // Both listeners are needed:
      // - readline.on("SIGINT") prevents readline from auto-closing when it sees ^C in the terminal input stream
      // - process.on("SIGINT") prevents Deno/Node from exiting on the OS-level signal
      // Deduplicate so `handler` runs only once per Ctrl+C.
      let suppress = false;
      const wrapped = () => {
        if (suppress) return;
        suppress = true;
        queueMicrotask(() => {
          suppress = false;
        });
        handler();
      };
      readline.on("SIGINT", wrapped);
      process.on("SIGINT", wrapped);
      return () => {
        readline.off("SIGINT", wrapped);
        process.off("SIGINT", wrapped);
      };
    },
    close(): void {
      readline.close();
    },
  };
}

function humanizeToolName(name: string): string {
  return name.replaceAll("_", " ");
}

export async function repl<zO, zI, const Tools extends AnyTool[]>(
  agent: Agent<zO, zI, Tools>,
  options?: ReplOptions,
): Promise<void> {
  const io = options?.io ?? createNodeReplIo();
  const managesIoLifecycle = options?.io == null;
  const prompt = options?.prompt ?? DEFAULT_PROMPT;
  const greeting = options?.greeting === undefined ? DEFAULT_GREETING : options.greeting;

  const conversation: ChatItem[] = [];
  let activeAbortController: AbortController | null = null;

  const onSigInt = () => {
    if (activeAbortController) {
      activeAbortController.abort(new Error("Interrupted by SIGINT"));
      io.write("\n");
      return;
    }

    io.write("\n");
    io.close?.();
  };

  if (greeting !== false) {
    io.write(`${greeting}\n`);
  }

  const disposeInterrupt = io.onInterrupt?.(onSigInt);

  try {
    while (true) {
      const input = (await io.readLine(prompt))?.trim();
      if (input == null) {
        break;
      }

      if (input === "") {
        continue;
      } else if (input === "/exit") {
        break;
      } else if (input === "/reset") {
        conversation.length = 0;
        io.write("History cleared.\n");
        continue;
      } else if (input.startsWith("/")) {
        io.write(`Unknown command: ${input}\n`);
        continue;
      }

      activeAbortController = new AbortController();

      let streamedOutput = false;
      let announcedResponding = false;
      let announcedThinking = false;
      const activeTools = new Map<string, string>();

      try {
        const stream = agent.stream(
          [...conversation, { type: "input_text", content: input }],
          {
            ...options?.runOptions,
            signal: activeAbortController.signal,
          },
        );

        while (true) {
          const next = await stream.next();
          if (next.done) {
            const run = next.value;
            conversation.push({ type: "input_text", content: input }, ...run.history);
            if (!streamedOutput && run.outputText) {
              io.write(run.outputText);
            }
            io.write("\n");
            break;
          }

          if (next.value.type === "delta_output_reasoning" && !announcedThinking) {
            announcedThinking = true;
            io.write("[thinking]\n");
          } else if (next.value.type === "tool_use_start") {
            activeTools.set(next.value.tool_use_id, next.value.kind);
            io.write(`[using tool: ${humanizeToolName(next.value.kind)}]\n`);
          } else if (
            next.value.type === "tool_result_text" ||
            next.value.type === "tool_result_file"
          ) {
            const toolName = activeTools.get(next.value.tool_use_id) ?? next.value.tool_use_id;
            io.write(`[tool finished: ${humanizeToolName(toolName)}]\n`);
          } else if (
            (next.value.type === "delta_output_text" || next.value.type === "delta_output_preview") &&
            !announcedResponding
          ) {
            announcedResponding = true;
            io.write("[responding]\n");
          }

          if (next.value.type === "delta_output_text") {
            streamedOutput = true;
            io.write(next.value.delta);
          }
        }
      } catch (error) {
        if (activeAbortController.signal.aborted) {
          io.write("[aborted]\n");
        } else {
          io.write(`Error: ${errMessage(error)}\n`);
        }
      } finally {
        activeAbortController = null;
      }
    }
  } finally {
    disposeInterrupt?.();
    if (managesIoLifecycle) {
      io.close?.();
    }
  }
}
