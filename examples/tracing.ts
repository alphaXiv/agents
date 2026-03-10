// This example shows how you can use the tracing APIs in @alphaxiv/agents to
// visualize and debug complicated agent runs. Tracing is a completely optional
// layer that you build on top of regular streaming. By running this, you'll be
// able to watch a live visualization of the agent and sub-agent process.
//
// deno-lint-ignore-file no-import-prefix
import z from "zod";
import {
  type Adapter,
  Agent,
  type ModelString,
  registerGlobalTracer,
  Tool,
  type TraceEvent,
} from "@alphaxiv/agents";
import * as log from "jsr:@clo/lib@3.0.0/log.ts";
import * as async from "jsr:@clo/lib@3.0.0/async.ts";
import type { PartialTraceEvent } from "../src/tracing.ts";
import process from "node:process";

const adapter: Adapter | undefined = undefined;
const model: ModelString = "anthropic:claude-haiku-4-5";

const snackMenu = [
  { name: "miso ramen cup", calories: 420, prepMinutes: 6, price: 8 },
  { name: "citrus yogurt bowl", calories: 310, prepMinutes: 3, price: 7 },
  { name: "kimchi toastie", calories: 470, prepMinutes: 8, price: 9 },
  { name: "sesame edamame", calories: 260, prepMinutes: 4, price: 6 },
] as const;

async function main() {
  // For convenience, a tracer can be registered globally. A tracer at minimum
  // must listen to `event`, which is emitted for every completed trace. The
  // `start` callback is used to respond to the beginning of traces.
  //
  // In this example, the traces are stored in an array, but also integrating
  // start events with `@clo/lib/log.ts` for an interactive visual.
  let traces: PartialTraceEvent[] = [];
  let rerender: (() => void) | null = null;
  const unregister = registerGlobalTracer({
    event: (event) => {
      traces = traces.filter((x) => x.id !== event.id); // remove start event
      traces.push(event);
      rerender?.();
    },
    start: (start) => {
      traces.push(start);
      rerender?.();
    },
  });
  const widgetHost = log.headlessWidgetHost({
    writeOutput: (string) => process.stdout.write(string),
    writeInteractive: (string) => process.stderr.write(string),
    now: () => performance.now(),
    delay: async.delay,
    // TODO: fix @clo/log LMAO
    getSize: () => ({ rows: 100, columns: Infinity }),
  });
  const widget = widgetHost.startWidget({
    format: () =>
      renderTraceFlamegraph(traces, {
        width: terminalChartWidth(),
      }),
    onChange: (cb) => {
      rerender = cb;
      return () => rerender = null;
    },
    fps: 15,
  });

  const subagent = new Agent({
    adapter,
    model,
    instructions: [
      "You are the snack specialist.",
      "Always call load_menu first.",
      "Then call score_snacks exactly once before answering.",
      "Recommend one item and mention why it fits the request.",
    ].join(" "),
    tools: [loadMenu, scoreSnacks],
  });

  const delegateSpecialist = new Tool({
    name: "delegate_specialist",
    description: "Ask the snack specialist to evaluate the request.",
    parameters: z.object({
      request: z.string().describe("The snack brief to hand off."),
    }),
    async execute({ param }) {
      await sleep(250 + 150 * Math.random());
      const run = await subagent.run(param.request);
      return run.outputText;
    },
  });

  const agent = new Agent({
    adapter,
    model,
    instructions: [
      "You coordinate requests for a snack specialist.",
      "Always call delegate_specialist exactly once before answering.",
      "After the tool returns, summarize the recommendation in one concise sentence.",
    ].join(" "),
    tools: [delegateSpecialist],
  });

  try {
    const result = await agent.run(
      "Pick a comforting late-night snack under 450 calories that can be ready in 6 minutes.",
    );
    console.log(result.outputText);
  } catch (err) {
    console.error(err);
  } finally {
    unregister();
    widget();
    widgetHost.getDrawLock();
    console.log(
      renderTraceFlamegraph(traces, {
        width: terminalChartWidth(),
      }),
    );
  }
}

const RESET = "\x1b[0m";

type Color = readonly [number, number, number];

interface Style {
  fg?: Color;
  bg?: Color;
  bold?: boolean;
  dim?: boolean;
}

interface Cell extends Style {
  char: string;
}

export interface RenderTraceFlamegraphOptions {
  width?: number;
}

const BORDER: Color = [100, 116, 139];
const LABEL: Color = [148, 163, 184];
const PALETTE = {
  agent: [124, 58, 21],
  model: [215, 96, 26],
  tool: [217, 153, 27],
  message: [38, 112, 176],
  log: [32, 149, 136],
  error: [185, 28, 47],
} as const satisfies Record<TraceEvent["type"] | "error", Color>;

const loadMenu = new Tool({
  name: "load_menu",
  description: "Load the current snack menu for the pop-up kitchen.",
  parameters: z.void(),
  async execute() {
    await sleep(350 + 250 * Math.random());
    return JSON.stringify(snackMenu);
  },
});

const scoreSnacks = new Tool({
  name: "score_snacks",
  description: "Score the current snack menu against the user's constraints.",
  parameters: z.object({
    calorieLimit: z.number(),
    maxPrepMinutes: z.number(),
    mood: z.string().describe("What kind of snack experience the user wants."),
  }),
  async execute({ param }) {
    await sleep(450 + 250 * Math.random());

    const ranked = snackMenu
      .filter((item) =>
        item.calories <= param.calorieLimit &&
        item.prepMinutes <= param.maxPrepMinutes
      )
      .map((item) => ({
        ...item,
        score: scoreSnack(item, param.mood),
      }))
      .sort((a, b) =>
        b.score - a.score ||
        a.prepMinutes - b.prepMinutes ||
        a.calories - b.calories
      );

    return JSON.stringify({
      mood: param.mood,
      winner: ranked[0] ?? null,
      considered: ranked,
    });
  },
});

function scoreSnack(
  item: (typeof snackMenu)[number],
  mood: string,
): number {
  const normalizedMood = mood.toLowerCase();
  let score = 100 - item.calories / 10 - item.prepMinutes * 4 - item.price;
  if (normalizedMood.includes("comfort")) score += 18;
  if (normalizedMood.includes("late-night")) score += 10;
  if (normalizedMood.includes("light") || normalizedMood.includes("fresh")) {
    score -= item.calories / 18;
  }
  if (item.name.includes("ramen")) score += 8;
  if (item.name.includes("yogurt")) score += 12;
  if (item.name.includes("edamame")) score += 6;
  return Math.round(score);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

type PartialEventWithEnd = PartialTraceEvent & { end: number };

export function renderTraceFlamegraph(
  partialEvents: PartialTraceEvent[],
  options: RenderTraceFlamegraphOptions = {},
): string {
  if (partialEvents.length === 0) return "";

  const now = Date.now();
  const events = partialEvents.map((x) => ({
    ...x,
    end: x.end || now,
  }));

  const spans = [...events].sort((a, b) =>
    a.start - b.start || b.end - a.end || a.id.localeCompare(b.id)
  );
  const width = clamp(options.width ?? 72, 32, 140);
  const zero = spans[0].start;
  const total = Math.max(Math.max(...spans.map((span) => span.end)) - zero, 1);
  const byId = new Map(spans.map((span) => [span.id, span]));
  const childrenByParent = new Map<string | null, PartialEventWithEnd[]>();

  for (const span of spans) {
    const parent = span.parent != null && byId.has(span.parent)
      ? span.parent
      : null;
    const siblings = childrenByParent.get(parent) ?? [];
    siblings.push(span);
    childrenByParent.set(parent, siblings);
  }

  for (const siblings of childrenByParent.values()) {
    siblings.sort((a, b) =>
      a.start - b.start || b.end - a.end || a.id.localeCompare(b.id)
    );
  }

  const depthById = new Map<string, number>();
  const assignDepth = (span: PartialEventWithEnd, depth: number) => {
    if (depthById.has(span.id)) return;
    depthById.set(span.id, depth);
    for (const child of childrenByParent.get(span.id) ?? []) {
      assignDepth(child, depth + 1);
    }
  };

  for (const root of childrenByParent.get(null) ?? []) assignDepth(root, 0);
  for (const span of spans) assignDepth(span, 0);

  const maxDepth = Math.max(...depthById.values());
  const tickColumns = [
    ...new Set(
      [0, 0.25, 0.5, 0.75, 1].map((ratio) =>
        Math.min(width - 1, Math.round(ratio * (width - 1)))
      ),
    ),
  ];

  const rows = Array.from(
    { length: maxDepth + 1 },
    () => makeGuideRow(width, tickColumns),
  );

  for (const span of spans) {
    const depth = depthById.get(span.id) ?? 0;
    const row = rows[depth];
    const from = Math.min(
      width - 1,
      Math.floor(((span.start - zero) / total) * width),
    );
    const to = Math.max(
      from + 1,
      Math.min(width, Math.ceil(((span.end - zero) / total) * width)),
    );
    const bg = spanColor(span, depth);
    const fg = contrast(bg);

    for (let i = from; i < to; i++) {
      row[i] = { char: " ", bg, fg };
    }

    const label = fitLabel(spanLabel(span), Math.max(0, to - from - 2));
    const start = from + Number(to - from > 2);
    for (
      let i = 0;
      i < label.length && start + i < to - Number(to - from > 2);
      i++
    ) {
      row[start + i] = { char: label[i], bg, fg, bold: true };
    }
  }

  const border = styleText({ fg: BORDER, dim: true }, "|");
  const header = styleText({ fg: LABEL, bold: true }, "trace flamegraph");
  const rulerLabels = styleText(
    { fg: LABEL, dim: true },
    placeLabels(width, total),
  );
  const guide = styleText(
    { fg: BORDER, dim: true },
    makeGuideLine(width, tickColumns),
  );
  const flameRows = rows.map((row) => {
    return `${border}${renderCells(row)}${border}`;
  });

  return [
    `${header}`,
    rulerLabels,
    guide,
    ...flameRows,
    guide,
  ].join("\n");
}

function makeGuideRow(width: number, tickColumns: number[]): Cell[] {
  const row: Cell[] = Array.from({ length: width }, () => ({ char: " " }));
  for (const column of tickColumns) {
    row[column] = { char: "|", fg: BORDER, dim: true };
  }
  return row;
}

function makeGuideLine(width: number, tickColumns: number[]): string {
  const line = Array.from({ length: width }, () => "-");
  for (const column of tickColumns) line[column] = "+";
  return line.join("");
}

function placeLabels(width: number, total: number): string {
  const line = Array.from({ length: width }, () => " ");
  let lastEnd = -1;

  for (const ratio of [0, 0.25, 0.5, 0.75, 1]) {
    const label = ms(Math.round(total * ratio));
    let start = Math.round(ratio * (width - 1) - label.length / 2);
    start = clamp(start, 0, Math.max(0, width - label.length));
    if (start <= lastEnd) start = lastEnd + 1;
    if (start + label.length > width) continue;
    for (let i = 0; i < label.length; i++) line[start + i] = label[i];
    lastEnd = start + label.length - 1;
  }

  return line.join("");
}

function spanLabel(event: PartialTraceEvent): string {
  const suffix = event.errorMessage == null ? "" : " !";
  if (!event.content) {
    return `${event.type}`;
  }
  switch (event.type) {
    case "agent":
      return `agent ${event.content.model}${suffix}`;
    case "model":
      return `model ${event.content.reason}${suffix}`;
    case "tool":
      return `tool ${event.content.name}${suffix}`;
    case "message":
      return `message ${event.content.type}${suffix}`;
    case "log":
      return typeof event.content === "string"
        ? `log ${event.content}${suffix}`
        : `log${suffix}`;
  }
}

function spanColor(event: PartialTraceEvent, depth: number): Color {
  const base = event.errorMessage == null ? PALETTE[event.type] : PALETTE.error;
  const lift = ((depth * 19 + Math.floor(event.start / 83)) % 18) - 6;
  return [
    clampChannel(base[0] + lift),
    clampChannel(base[1] + Math.floor(lift * 0.8)),
    clampChannel(base[2] - Math.floor(lift * 0.6)),
  ];
}

function renderCells(cells: Cell[]): string {
  let out = "";
  let activeKey = "plain";
  let styled = false;

  for (const cell of cells) {
    const style = { fg: cell.fg, bg: cell.bg, bold: cell.bold, dim: cell.dim };
    const key = styleKey(style);
    if (key !== activeKey) {
      if (styled) out += RESET;
      const open = ansi(style);
      if (open) {
        out += open;
        styled = true;
      } else {
        styled = false;
      }
      activeKey = key;
    }
    out += cell.char;
  }

  if (styled) out += RESET;
  return out;
}

function styleText(style: Style, text: string): string {
  const open = ansi(style);
  return open ? `${open}${text}${RESET}` : text;
}

function ansi(style: Style): string {
  const codes: string[] = [];
  if (style.bold) codes.push("1");
  if (style.dim) codes.push("2");
  if (style.fg) codes.push(`38;2;${style.fg[0]};${style.fg[1]};${style.fg[2]}`);
  if (style.bg) codes.push(`48;2;${style.bg[0]};${style.bg[1]};${style.bg[2]}`);
  return codes.length > 0 ? `\x1b[${codes.join(";")}m` : "";
}

function styleKey(style: Style): string {
  return [
    style.bold ? "1" : "0",
    style.dim ? "1" : "0",
    style.fg?.join(",") ?? "",
    style.bg?.join(",") ?? "",
  ].join("|");
}

function contrast(color: Color): Color {
  const luminance = (color[0] * 299 + color[1] * 587 + color[2] * 114) / 1000;
  return luminance >= 150 ? [15, 23, 42] : [248, 250, 252];
}

function fitLabel(label: string, width: number): string {
  if (width <= 0) return "";
  if (label.length <= width) return label;
  if (width <= 3) return label.slice(0, width);
  return `${label.slice(0, width - 3)}...`;
}

function ms(n: number): string {
  return n >= 1_000 ? `${(n / 1_000).toFixed(n >= 10_000 ? 1 : 2)}s` : `${n}ms`;
}

function clamp(n: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, n));
}

function clampChannel(n: number): number {
  return clamp(Math.round(n), 0, 255);
}

function terminalChartWidth(): number {
  try {
    return clamp(Deno.consoleSize().columns - 8, 48, 110);
  } catch {
    return 72;
  }
}

if (import.meta.main) {
  await main();
}
