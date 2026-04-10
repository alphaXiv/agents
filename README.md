# @alphaxiv/agents

TypeScript agents for real applications.

`@alphaxiv/agents` is a small agent framework built around a few pragmatic ideas:

- Flat, database-friendly chat history types
- Zod-based structured output
- First-class tool calling with retries and cancellation
- Streaming, tracing, and interactive CLI support
- Multimodal inputs and tool results
- Multiple model providers behind one agent loop
- Fallback models built into the agent constructor

## Quickstart

Models and Adapters for these providers are shipped out of the box:

- Anthropic
- OpenAI
- Gemini
- Vertex AI
- Tributary
- OpenRouter

As well as adapters for commonly supported [Open Responses](https://www.openresponses.org/) and
[OpenAI Completions](https://developers.openai.com/api/docs/guides/completions) API's so you can easily BYO models.

Provider model constructors accept options like `apiKey` and `baseUrl`. If you do not pass an API key explicitly,
provider-specific environment variables are used where supported.

```ts
import z from "npm:zod";
import { Agent, OpenAIModel, Tool } from "jsr:@alphaxiv/agents";

const calculator = new Tool({
  name: "calculator",
  description: "Do basic arithmetic.",
  parameters: z.object({
    operation: z.enum(["add", "subtract", "multiply", "divide"]),
    left: z.number(),
    right: z.number(),
  }),
  execute: ({ operation, left, right }) => {
    if (operation === "add") return String(left + right);
    if (operation === "subtract") return String(left - right);
    if (operation === "multiply") return String(left * right);
    return String(left / right);
  },
});

const agent = new Agent({
  model: new OpenAIModel({ model: "gpt-4.1-mini" }), // Or via shorthand "openai:gpt-4.1-mini"
  instructions: "You are a helpful assistant. Use the calculator when math is needed.",
  tools: [calculator],
});

const result = await agent.run("What is 144 divided by 12?");

console.log(result.outputText);
console.log(result.history);
console.log(result.inputTokens, result.outputTokens);
```

`agent.run(...)` accepts either a plain string or a `ChatItem[]` history.

The returned object includes:

- `output`: the structured result if you configured one, otherwise `undefined` unless a tool returns `ModelOutput`
- `outputText`: a convenient string form of the final answer
- `history`: flat chat history items that are easy to store
- `inputTokens` and `outputTokens`

## Structured output

If you provide an `output` Zod schema, the agent guarantees that the output will match it.

```ts
import z from "npm:zod";
import { Agent, GeminiModel } from "jsr:@alphaxiv/agents";

const paperExtractor = new Agent({
  model: new GeminiModel({ model: "gemini-2.0-flash" }),
  instructions: "Extract the paper title and abstract from the raw text.",
  output: z.object({
    title: z.string(),
    abstract: z.string(),
  }),
});

const result = await paperExtractor.run(rawPaperText);

console.log(result.output.title);
console.log(result.output.abstract);
```

## Tools

Tools are plain class instances with:

- `name`
- `description`
- `parameters`: a Zod schema
- `execute(input, { signal })`

Tool names are normalized internally so provider-specific restrictions do not leak into your app.

### Returning text or files from a tool

`execute` can return a string, or an array of typed tool result items.

```ts
import z from "npm:zod";
import { Agent, AnthropicModel, Tool } from "jsr:@alphaxiv/agents";

const getImage = new Tool({
  name: "get_image",
  description: "Return an example image.",
  parameters: z.void(),
  execute: () => [
    {
      type: "tool_result_text",
      content: "Here is the image you asked for.",
    },
    {
      type: "tool_result_file",
      kind: "image/png",
      content: "https://paper-assets.alphaxiv.org/image/2510.18234v1.png",
    },
  ],
});

const agent = new Agent({
  model: new AnthropicModel({ model: "claude-haiku-4-5" }),
  instructions: "You are a helpful assistant.",
  tools: [getImage],
});
```

### Returning final output directly from a tool

If a tool already knows the final answer, return `new ModelOutput(...)` to end the run immediately.

```ts
import z from "npm:zod";
import { Agent, ModelOutput, OpenAIModel, Tool } from "jsr:@alphaxiv/agents";

const reportHelpfulIds = new Tool({
  name: "report_helpful_ids",
  description: "Finish the run by returning the selected ids.",
  parameters: z.object({
    ids: z.array(z.string()),
  }),
  execute: ({ ids }) => new ModelOutput(ids),
});

const agent = new Agent({
  model: new OpenAIModel({ model: "gpt-4.1-mini" }),
  instructions: "Find relevant ids, then call report_helpful_ids.",
  tools: [reportHelpfulIds],
});
```

### Retries and cancellation

- Tools support per-tool retries, timeouts and signal cancellation
- Agent-level aborts propagate into running tools

## Fallback models

An agent can be configured with multiple models. The run retries in round-robin order and falls back across providers if
needed.

```ts
import { Agent, AnthropicModel, OpenAIModel } from "jsr:@alphaxiv/agents";

const agent = new Agent({
  model: [
    new OpenAIModel({ model: "gpt-4.1-mini" }),
    new AnthropicModel({ model: "claude-haiku-4-5" }),
  ],
  instructions: "You are a reliable assistant.",
  maxRetries: 3,
});
```

## Streaming

Use `agent.stream(...)` when you want token-by-token output, reasoning deltas, and live tool events.

```ts
import process from "node:process";
import { Agent, OpenAIModel } from "jsr:@alphaxiv/agents";

const agent = new Agent({
  model: new OpenAIModel({ model: "gpt-5.4-mini" }),
  instructions: "You are a helpful assistant.",
});

const stream = agent.stream("Explain gradient descent simply.");

while (true) {
  const next = await stream.next();
  if (next.done) {
    console.log("final:", next.value.outputText);
    break;
  }

  if (next.value.type === "delta_output_text") {
    process.stdout.write(next.value.delta);
  }
}
```

The helpers `addStreamItem(...)` and `convertChatItemsToStream(...)` are also exported for rebuilding or adapting
streamed histories.

## CLI

Built-in way to convert agents into a CLI app.

```ts
import z from "npm:zod";
import { Agent, cli, OpenAIModel, Tool } from "jsr:@alphaxiv/agents";

const echo = new Tool({
  name: "echo",
  description: "Echo text back.",
  parameters: z.string(),
  execute: (input) => input,
});

const agent = new Agent({
  model: new OpenAIModel({ model: "gpt-4.1-mini" }),
  instructions: "You are a helpful assistant.",
  tools: [echo],
});

await cli(agent);
```

The built-in CLI supports:

- Streaming output
- Tool activity announcements
- `/reset` to clear history
- `/exit` to quit
- `Ctrl+C` to abort an active run

## Multimodal input

Pass files directly as chat items.

```ts
import { Agent, OpenRouterModel } from "jsr:@alphaxiv/agents";

const agent = new Agent({
  model: new OpenRouterModel({ model: "openrouter:meta-llama/llama-4-maverick" }),
  instructions: "You are a helpful assistant.",
});

const result = await agent.run([
  {
    type: "input_file",
    kind: "application/pdf",
    content: "https://fetcher.alphaxiv.org/v2/pdf/2511.02824v1.pdf",
  },
  {
    type: "input_text",
    content: "Who wrote this paper?",
  },
]);

console.log(result.outputText);
```

This works for file inputs like PDFs, images, and CSVs, depending on the model provider.

## Tracing

Tracing is optional, but built in. You can:

- Create traces with `newTrace(...)`
- Wrap custom work with `withTrace(...)`
- Register global tracers with `registerGlobalTracer(...)`
- Inspect agent, model, tool, and message spans

This is useful for debugging nested agents, provider retries, and slow tools.
