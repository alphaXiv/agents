import { assertEquals, assertRejects } from "@std/assert";
import { FakeTime } from "@std/testing/time";
import { Agent } from "../../mod.ts";
import { azureOpenAIModel } from "../../src/adapters/azure_openai/adapter.ts";
import type { OpenResponsesClient } from "../../src/adapters/open_responses/adapter.ts";
import { openAIModel } from "../../src/adapters/openai/adapter.ts";
import type { Adapter } from "../../src/adapters/adapter.ts";
import type { ChatItem, ProviderFileStore, StreamItem } from "../../src/types.ts";

const IMAGE_URL = "https://example.com/uploads/user/cats.png";
const PDF_URL = "https://example.com/uploads/user/paper.pdf";

const HISTORY: ChatItem[] = [
  { type: "input_file", kind: "image/png", content: IMAGE_URL },
  { type: "input_file", kind: "application/pdf", content: PDF_URL },
];

interface FakeClient {
  client: OpenResponsesClient;
  requests: { input: unknown[] }[];
  created: { purpose: string; expiresAfter: unknown; filename: string }[];
  createOptions: { signal?: AbortSignal }[];
  deleted: string[];
  nextId: () => string;
  failWith?: Error;
  /** Fails the first request only, so the adapter's own resend can succeed. */
  failOnce?: Error;
}

function createFakeClient(idPrefix: string): FakeClient {
  let counter = 0;
  const fake: FakeClient = {
    requests: [],
    created: [],
    createOptions: [],
    deleted: [],
    nextId: () => `${idPrefix}-${++counter}`,
    client: undefined as unknown as OpenResponsesClient,
  };

  fake.client = {
    responses: {
      stream(request: { input: unknown[] }) {
        fake.requests.push(request);
        const failWith = fake.failWith ?? (fake.requests.length === 1 ? fake.failOnce : undefined);
        return {
          async *[Symbol.asyncIterator]() {
            if (failWith) throw failWith;
            yield* [];
          },
          finalResponse() {
            return Promise.resolve({ usage: { input_tokens: 1, output_tokens: 1 } });
          },
        };
      },
    },
    files: {
      create(
        body: { purpose: string; expires_after?: unknown; file: { name: string } },
        options?: { signal?: AbortSignal },
      ) {
        fake.created.push({
          purpose: body.purpose,
          expiresAfter: body.expires_after,
          filename: body.file.name,
        });
        fake.createOptions.push(options ?? {});
        // The real client rejects an aborted request instead of uploading.
        if (options?.signal?.aborted) return Promise.reject(new Error("Request was aborted"));
        return Promise.resolve({ id: fake.nextId() });
      },
      delete(fileId: string) {
        fake.deleted.push(fileId);
        return Promise.resolve({ id: fileId, deleted: true });
      },
    },
  } as unknown as OpenResponsesClient;

  return fake;
}

function createMemoryStore(rows = new Map<string, { fileId: string; expiresAt: Date }>()): ProviderFileStore & {
  rows: Map<string, { fileId: string; expiresAt: Date }>;
} {
  return {
    rows,
    get(url) {
      return Promise.resolve(rows.get(url));
    },
    set(url, fileId, expiresAt) {
      const existing = rows.get(url);
      // A live row keeps its id, as the store contract requires.
      if (existing && existing.expiresAt > new Date()) return Promise.resolve({ fileId: existing.fileId });
      rows.set(url, { fileId, expiresAt });
      return Promise.resolve({ fileId });
    },
    delete(url, fileId) {
      if (rows.get(url)?.fileId === fileId) rows.delete(url);
      return Promise.resolve();
    },
  };
}

function stubFetch() {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (() => Promise.resolve(new Response("file-bytes"))) as typeof fetch;
  return { [Symbol.dispose]: () => void (globalThis.fetch = originalFetch) };
}

/** Runs `work` with the adapter's resend pauses skipped. */
async function withFakeTime<T>(work: () => Promise<T>): Promise<T> {
  using time = new FakeTime();
  let settled = false;
  const result = work();
  result.then(() => settled = true, () => settled = true);
  while (!settled) await time.tickAsync(100);
  return await result;
}

function runStream(
  adapter: Adapter<unknown, unknown>,
  history: ChatItem[] = HISTORY,
  signal: AbortSignal = AbortSignal.timeout(5000),
) {
  return withFakeTime(async () => {
    const stream = adapter.stream({
      history,
      instructions: "test",
      tools: [],
      signal,
    });
    while (!(await stream.next()).done) { /* drain */ }
  });
}

function collectStream(
  adapter: Adapter<unknown, unknown>,
  history: ChatItem[] = HISTORY,
): Promise<StreamItem[]> {
  return withFakeTime(async () => {
    const items: StreamItem[] = [];
    const stream = adapter.stream({
      history,
      instructions: "test",
      tools: [],
      signal: AbortSignal.timeout(5000),
    });
    for (let next = await stream.next(); !next.done; next = await stream.next()) {
      items.push(next.value);
    }
    return items;
  });
}

function contentOf(request: { input: unknown[] }, index: number) {
  const message = request.input[index] as { content: unknown[] };
  return message.content[0] as Record<string, unknown>;
}

Deno.test("uploads on a store miss and reuses the stored id on the next call", async () => {
  using _fetch = stubFetch();
  const fake = createFakeClient("file");
  const store = createMemoryStore();
  const adapter = openAIModel({ model: "gpt-5.6-luna", client: fake.client, fileStore: store });

  await runStream(adapter);

  assertEquals(fake.created.length, 2);
  assertEquals(fake.created[0], {
    purpose: "user_data",
    expiresAfter: { anchor: "created_at", seconds: 9 * 24 * 60 * 60 },
    filename: "cats.png",
  });
  assertEquals(contentOf(fake.requests[0], 0), { type: "input_image", file_id: "file-1", detail: "auto" });
  assertEquals(contentOf(fake.requests[0], 1), { type: "input_file", file_id: "file-2" });

  await runStream(adapter);

  assertEquals(fake.created.length, 2);
  assertEquals(fake.deleted, []);
  assertEquals(contentOf(fake.requests[1], 0), { type: "input_image", file_id: "file-1", detail: "auto" });
  assertEquals(contentOf(fake.requests[1], 1), { type: "input_file", file_id: "file-2" });
});

Deno.test("Azure uploads with purpose assistants and no expires_after", async () => {
  using _fetch = stubFetch();
  const fake = createFakeClient("assistant");
  const store = createMemoryStore();
  const adapter = azureOpenAIModel({ model: "gpt-5.6-luna", client: fake.client, fileStore: store });

  await runStream(adapter);

  assertEquals(fake.created.map((created) => created.purpose), ["assistants", "assistants"]);
  assertEquals(fake.created.map((created) => created.expiresAfter), [undefined, undefined]);
  assertEquals(contentOf(fake.requests[0], 0), { type: "input_image", file_id: "assistant-1", detail: "auto" });
});

Deno.test("an expired row is replaced and the old id deleted on the provider", async () => {
  using _fetch = stubFetch();
  const fake = createFakeClient("file");
  const store = createMemoryStore(
    new Map([
      [IMAGE_URL, { fileId: "file-old", expiresAt: new Date(Date.now() - 1000) }],
      [PDF_URL, { fileId: "file-live", expiresAt: new Date(Date.now() + 60_000) }],
    ]),
  );
  const adapter = openAIModel({ model: "gpt-5.6-luna", client: fake.client, fileStore: store });

  await runStream(adapter);

  assertEquals(fake.deleted, ["file-old"]);
  assertEquals(fake.created.length, 1);
  assertEquals(contentOf(fake.requests[0], 0), { type: "input_image", file_id: "file-1", detail: "auto" });
  assertEquals(contentOf(fake.requests[0], 1), { type: "input_file", file_id: "file-live" });
});

Deno.test("a set that returns a foreign id deletes our upload and uses the winner", async () => {
  using _fetch = stubFetch();
  const fake = createFakeClient("file");
  const store = createMemoryStore();
  store.set = () => Promise.resolve({ fileId: "file-winner" });
  const adapter = openAIModel({ model: "gpt-5.6-luna", client: fake.client, fileStore: store });

  await runStream(adapter, [HISTORY[0]]);

  assertEquals(fake.deleted, ["file-1"]);
  assertEquals(contentOf(fake.requests[0], 0), { type: "input_image", file_id: "file-winner", detail: "auto" });
});

/** Rows for both fixture URLs, live, as an earlier call would have left them. */
function storedRows() {
  const expiresAt = new Date(Date.now() + 60_000);
  return new Map([[IMAGE_URL, { fileId: "file-a", expiresAt }], [PDF_URL, { fileId: "file-b", expiresAt }]]);
}

Deno.test("a stored id the provider rejects is dropped, uploaded again and resent inside the same call", async () => {
  using _fetch = stubFetch();
  const fake = createFakeClient("file");
  const store = createMemoryStore(storedRows());
  const adapter = openAIModel({ model: "gpt-5.6-luna", client: fake.client, fileStore: store });
  fake.failOnce = new Error("404 Files [file-b] were not found");

  await runStream(adapter);

  assertEquals(fake.requests.length, 2);
  assertEquals(fake.created.length, 1);
  assertEquals(contentOf(fake.requests[1], 0), { type: "input_image", file_id: "file-a", detail: "auto" });
  assertEquals(contentOf(fake.requests[1], 1), { type: "input_file", file_id: "file-1" });
  assertEquals(store.rows.get(PDF_URL)?.fileId, "file-1");
  assertEquals(store.rows.get(IMAGE_URL)?.fileId, "file-a");
});

Deno.test("an id uploaded in this call that the provider rejects is resent as is, not uploaded again", async () => {
  using _fetch = stubFetch();
  const fake = createFakeClient("file");
  const store = createMemoryStore();
  const adapter = openAIModel({ model: "gpt-5.6-luna", client: fake.client, fileStore: store });
  fake.failOnce = new Error("400 Files [file-2] were not found due to ownership verification failure.");

  await runStream(adapter);

  assertEquals(fake.requests.length, 2);
  assertEquals(fake.created.length, 2);
  assertEquals(contentOf(fake.requests[1], 1), { type: "input_file", file_id: "file-2" });
  assertEquals(store.rows.get(PDF_URL)?.fileId, "file-2");
});

Deno.test("a resend yields request_start again", async () => {
  using _fetch = stubFetch();
  const fake = createFakeClient("file");
  const adapter = openAIModel({ model: "gpt-5.6-luna", client: fake.client, fileStore: createMemoryStore() });
  fake.failOnce = new Error("404 Files [file-2] were not found");

  const items = await collectStream(adapter);

  assertEquals(items.filter((item) => item.type === "request_start").length, 2);
});

Deno.test("a file the provider keeps rejecting is sent three times, then the provider error surfaces", async () => {
  using _fetch = stubFetch();
  const fake = createFakeClient("file");
  const store = createMemoryStore(storedRows());
  const adapter = openAIModel({ model: "gpt-5.6-luna", client: fake.client, fileStore: store });

  // The provider rejects the PDF on every request, naming whichever id it carried.
  const send = fake.client.responses.stream.bind(fake.client.responses);
  fake.client.responses.stream = ((request: { input: unknown[] }) => {
    fake.failWith = new Error(`404 Files [${contentOf(request, 1).file_id as string}] were not found`);
    return send(request as never);
  }) as typeof fake.client.responses.stream;

  await assertRejects(() => runStream(adapter), Error, "Files [file-1] were not found");

  assertEquals(fake.requests.length, 3);
  // The stored id was replaced once. The fresh upload was resent, never uploaded again.
  assertEquals(fake.created.length, 1);
  assertEquals(store.rows.get(PDF_URL)?.fileId, "file-1");
  assertEquals(store.rows.get(IMAGE_URL)?.fileId, "file-a");
});

Deno.test("an unrelated provider error is rethrown untouched", async () => {
  using _fetch = stubFetch();
  const fake = createFakeClient("file");
  const store = createMemoryStore();
  const adapter = openAIModel({ model: "gpt-5.6-luna", client: fake.client, fileStore: store });
  fake.failWith = new Error("500 something else broke");

  await assertRejects(() => runStream(adapter), Error, "500 something else broke");

  assertEquals(fake.requests.length, 1);
  assertEquals(store.rows.size, 2);
});

Deno.test("without a fileStore the input keeps image_url and file_data", async () => {
  using _fetch = stubFetch();
  const fake = createFakeClient("file");
  const adapter = openAIModel({ model: "gpt-5.6-luna", client: fake.client });

  await runStream(adapter);

  assertEquals(fake.created, []);
  assertEquals(contentOf(fake.requests[0], 0), { type: "input_image", image_url: IMAGE_URL, detail: "auto" });
  assertEquals(contentOf(fake.requests[0], 1), {
    type: "input_file",
    file_data: "data:application/pdf;base64,ZmlsZS1ieXRlcw==",
    filename: "paper.pdf",
  });
});

Deno.test("an upload is named with the extension for its mime type when the URL has none", async () => {
  using _fetch = stubFetch();
  const fake = createFakeClient("file");
  const adapter = openAIModel({ model: "gpt-5.6-luna", client: fake.client, fileStore: createMemoryStore() });

  await runStream(adapter, [
    { type: "input_file", kind: "application/pdf", content: "https://example.com/pdf/1706.03762v7" },
    { type: "input_file", kind: "image/jpeg", content: "https://example.com/i/abc" },
    { type: "input_file", kind: "image/png", content: "https://example.com/i/Logo.PNG" },
    { type: "input_file", kind: "image/jpeg", content: "https://example.com/i/photo.jpeg" },
    { type: "input_file", kind: "image/jpeg", content: "https://example.com/i/photo.jpg" },
  ]);

  assertEquals(fake.created.map((c) => c.filename), [
    "1706.03762v7.pdf",
    "abc.jpg",
    "Logo.PNG",
    "photo.jpeg",
    "photo.jpg",
  ]);
});

Deno.test("the upload carries the stream's abort signal", async () => {
  using _fetch = stubFetch();
  const fake = createFakeClient("file");
  const adapter = openAIModel({ model: "gpt-5.6-luna", client: fake.client, fileStore: createMemoryStore() });
  const signal = AbortSignal.timeout(5000);

  await runStream(adapter, [HISTORY[0]], signal);

  assertEquals(fake.createOptions, [{ signal }]);
});

Deno.test("an aborted upload is rethrown and nothing is stored", async () => {
  using _fetch = stubFetch();
  const fake = createFakeClient("file");
  const store = createMemoryStore();
  const adapter = openAIModel({ model: "gpt-5.6-luna", client: fake.client, fileStore: store });

  await assertRejects(() => runStream(adapter, [HISTORY[0]], AbortSignal.abort()), Error, "Request was aborted");

  assertEquals(store.rows.size, 0);
  assertEquals(fake.requests, []);
});

Deno.test("a row replaced while the request was in flight survives the stale eviction", async () => {
  using _fetch = stubFetch();
  const fake = createFakeClient("file");
  const store = createMemoryStore(
    new Map([[IMAGE_URL, { fileId: "file-x", expiresAt: new Date(Date.now() + 60_000) }]]),
  );
  const adapter = openAIModel({ model: "gpt-5.6-luna", client: fake.client, fileStore: store });
  fake.failOnce = new Error("404 Files [file-x] were not found");

  // Another stream hit the same miss first, uploaded its own file and stored it.
  const send = fake.client.responses.stream.bind(fake.client.responses);
  fake.client.responses.stream = ((request: never) => {
    store.rows.set(IMAGE_URL, { fileId: "file-y", expiresAt: new Date(Date.now() + 60_000) });
    return send(request);
  }) as typeof fake.client.responses.stream;

  await runStream(adapter, [HISTORY[0]]);

  assertEquals(store.rows.get(IMAGE_URL)?.fileId, "file-y");
  assertEquals(contentOf(fake.requests[1], 0), { type: "input_image", file_id: "file-y", detail: "auto" });
  assertEquals(fake.created, []);
});

Deno.test("a failed upload rejects the stream before any request, keeping the id already stored", async () => {
  using _fetch = stubFetch();
  const fake = createFakeClient("file");
  const store = createMemoryStore();
  const adapter = openAIModel({ model: "gpt-5.6-luna", client: fake.client, fileStore: store });

  const files = fake.client.files!;
  const create = files.create.bind(files);
  files.create = ((body: never, options: never) => {
    if (fake.created.length === 1) return Promise.reject(new Error("upload failed"));
    return create(body, options);
  }) as typeof files.create;

  await assertRejects(() => runStream(adapter), Error, "upload failed");

  assertEquals(fake.requests, []);
  assertEquals(store.rows.get(IMAGE_URL)?.fileId, "file-1");
  assertEquals(store.rows.has(PDF_URL), false);
});

Deno.test("request_start is yielded only once every upload is done", async () => {
  using _fetch = stubFetch();
  const fake = createFakeClient("file");
  const adapter = openAIModel({ model: "gpt-5.6-luna", client: fake.client, fileStore: createMemoryStore() });

  const stream = adapter.stream({
    history: HISTORY,
    instructions: "test",
    tools: [],
    signal: AbortSignal.timeout(5000),
  });
  const first = await stream.next();

  assertEquals(first.done, false);
  assertEquals(first.value, { type: "request_start", index: 0 });
  assertEquals(fake.created.length, 2);
  while (!(await stream.next()).done) { /* drain */ }
});

Deno.test("any error naming a sent id counts as a rejection, whatever its wording", async () => {
  using _fetch = stubFetch();
  const fake = createFakeClient("file");
  const store = createMemoryStore();
  const adapter = openAIModel({ model: "gpt-5.6-luna", client: fake.client, fileStore: store });
  fake.failWith = new Error(
    "400 Invalid 'input[0].content[0].file_id': 'file-1'. Expected an ID that begins with 'assistant'.",
  );

  await assertRejects(() => runStream(adapter), Error, "Expected an ID that begins with 'assistant'.");

  assertEquals(fake.requests.length, 3);
  assertEquals(fake.created.length, 2);
  assertEquals(store.rows.get(IMAGE_URL)?.fileId, "file-1");
  assertEquals(store.rows.get(PDF_URL)?.fileId, "file-2");
});

Deno.test("the agent never sees the recovery, so a single model and one run are enough", async () => {
  using _fetch = stubFetch();
  const store = createMemoryStore();
  const requests: { input: unknown[] }[] = [];
  let uploads = 0;
  const client = {
    responses: {
      stream(request: { input: unknown[] }) {
        requests.push(request);
        const attempt = requests.length;
        const fileId = (request.input[0] as { content: { file_id: string }[] }).content[0].file_id;
        return {
          async *[Symbol.asyncIterator]() {
            if (attempt === 1) throw new Error(`404 Files [${fileId}] were not found`);
            yield { type: "response.output_text.delta", output_index: 0, delta: "done" };
          },
          finalResponse: () => Promise.resolve({ usage: { input_tokens: 1, output_tokens: 1 } }),
        };
      },
    },
    files: {
      create: () => Promise.resolve({ id: `file-${++uploads}` }),
      delete: (fileId: string) => Promise.resolve({ id: fileId, deleted: true }),
    },
  } as unknown as OpenResponsesClient;

  const agent = new Agent({
    model: openAIModel({ model: "gpt-5.6-luna", client, fileStore: store }),
    instructions: "test",
  });

  const items = await withFakeTime(async () => {
    const items: StreamItem[] = [];
    for await (const item of agent.stream([HISTORY[0]])) {
      items.push(item);
    }
    return items;
  });

  assertEquals(items.filter((item) => item.type === "model_switched").length, 0);
  assertEquals(requests.length, 2);
  assertEquals((requests[0].input[0] as { content: { file_id: string }[] }).content[0].file_id, "file-1");
  assertEquals((requests[1].input[0] as { content: { file_id: string }[] }).content[0].file_id, "file-1");
  assertEquals(store.rows.get(IMAGE_URL)?.fileId, "file-1");
  assertEquals(
    items.flatMap((item) => item.type === "delta_output_text" ? [item.delta] : []).join(""),
    "done",
  );
});
