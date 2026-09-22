import { assertEquals, assertRejects } from "@std/assert";
import type OpenAI from "openai";
import { Agent } from "../../mod.ts";
import { openAICompletionsModel } from "../../src/adapters/openai_completions/adapter.ts";
import { InvalidAttachmentError } from "../../src/errors.ts";
import type { StreamItem } from "../../src/types.ts";

const MISSING_URL = "https://example.com/uploads/gone.txt";
const FAKE_PDF_URL = "https://example.com/uploads/notes.pdf";

function stubFetch() {
  const calls: string[] = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = ((input: string | URL | Request) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
    calls.push(url);
    if (url === MISSING_URL) return Promise.resolve(new Response("Not Found", { status: 404 }));
    return Promise.resolve(new Response("plain text, not a PDF"));
  }) as typeof fetch;

  return { calls, [Symbol.dispose]: () => void (globalThis.fetch = originalFetch) };
}

function createFakeClient() {
  const requests: unknown[] = [];
  const client = {
    chat: {
      completions: {
        stream(request: unknown) {
          requests.push(request);
          return {
            async *[Symbol.asyncIterator]() {
              yield { choices: [{ delta: { content: "ok" } }] };
            },
            // deno-lint-ignore require-await
            async finalChatCompletion() {
              return { usage: undefined };
            },
          };
        },
      },
    },
  } as unknown as Pick<OpenAI, "chat">;

  return { client, requests };
}

function textModel(provider: string, client: Pick<OpenAI, "chat">) {
  return openAICompletionsModel({ model: "gpt-test", provider, client, pdfSupport: { mode: "text" } });
}

Deno.test("an attachment that 404s fails the run without a model switch", async () => {
  using fetchStub = stubFetch();
  const primary = createFakeClient();
  const fallback = createFakeClient();

  const agent = new Agent({
    model: [textModel("primary", primary.client), textModel("fallback", fallback.client)],
    instructions: "test",
  });

  const items: StreamItem[] = [];
  await assertRejects(async () => {
    for await (const item of agent.stream([{ type: "input_file", kind: "text/plain", content: MISSING_URL }])) {
      items.push(item);
    }
  }, InvalidAttachmentError);

  assertEquals(fetchStub.calls, [MISSING_URL]);
  assertEquals(primary.requests.length, 0);
  assertEquals(fallback.requests.length, 0);
  assertEquals(items.filter((item) => item.type === "model_switched").length, 0);
});

Deno.test("a text file served as a PDF fails the run without a model switch", async () => {
  using fetchStub = stubFetch();
  const primary = createFakeClient();
  const fallback = createFakeClient();

  const agent = new Agent({
    model: [textModel("primary", primary.client), textModel("fallback", fallback.client)],
    instructions: "test",
  });

  const items: StreamItem[] = [];
  await assertRejects(
    async () => {
      for await (const item of agent.stream([{ type: "input_file", kind: "application/pdf", content: FAKE_PDF_URL }])) {
        items.push(item);
      }
    },
    InvalidAttachmentError,
    "could not be parsed as a PDF: Invalid PDF structure",
  );

  assertEquals(fetchStub.calls, [FAKE_PDF_URL]);
  assertEquals(primary.requests.length, 0);
  assertEquals(fallback.requests.length, 0);
  assertEquals(items.filter((item) => item.type === "model_switched").length, 0);
});
