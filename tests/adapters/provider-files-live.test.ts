/**
 * Live end-to-end test of the Files API path, with real uploads, Responses calls and deletes.
 * Every file uploaded here is deleted at the end, even when an assertion fails.
 * @module
 */

import { assert, assertEquals, assertNotEquals } from "@std/assert";
import OpenAI from "openai";
import { Agent, type ProviderFileStore } from "../../mod.ts";
import type { Adapter } from "../../src/adapters/adapter.ts";
import { azureOpenAIModel } from "../../src/adapters/azure_openai/adapter.ts";
import type { OpenResponsesClient } from "../../src/adapters/open_responses/adapter.ts";
import { openAIModel } from "../../src/adapters/openai/adapter.ts";
import type { ChatItem } from "../../src/types.ts";
import { INTEGRATION_TIMEOUT_MS } from "./shared.ts";

const HAS_OPENAI_KEY = Boolean(Deno.env.get("OPENAI_API_KEY"));
const HAS_AZURE_KEYS = Boolean(Deno.env.get("AZURE_OPENAI_API_KEY") && Deno.env.get("AZURE_OPENAI_ENDPOINT"));

const IMAGE_URL = "https://www.w3.org/Icons/w3c_home.png";
// The URL must not give the content away, or the model can answer from the filename alone.
const PDF_URL = "https://arxiv.org/pdf/1706.03762v7";
const PDF_TITLE = /attention is all you need/i;

const HISTORY: ChatItem[] = [
  { type: "input_file", kind: "image/png", content: IMAGE_URL },
  { type: "input_file", kind: "application/pdf", content: PDF_URL },
  {
    type: "input_text",
    content: "What is the exact title of the PDF? Then say in one short sentence what the image shows.",
  },
];

function createCountingClient(real: OpenAI) {
  const counts = { uploads: 0, requests: 0 };
  const uploaded: string[] = [];
  const client = {
    responses: {
      stream(...args: Parameters<OpenAI["responses"]["stream"]>) {
        counts.requests++;
        return real.responses.stream(...args);
      },
    },
    files: {
      async create(...args: Parameters<OpenAI["files"]["create"]>) {
        counts.uploads++;
        const file = await real.files.create(...args);
        uploaded.push(file.id);
        return file;
      },
      delete: (...args: Parameters<OpenAI["files"]["delete"]>) => real.files.delete(...args),
    },
  } as unknown as OpenResponsesClient;
  return { client, counts, uploaded };
}

function createMemoryStore() {
  const rows = new Map<string, { fileId: string; expiresAt: Date }>();
  const store: ProviderFileStore = {
    get: (url) => Promise.resolve(rows.get(url)),
    set(url, fileId, expiresAt) {
      const existing = rows.get(url);
      if (existing && existing.expiresAt > new Date()) return Promise.resolve({ fileId: existing.fileId });
      rows.set(url, { fileId, expiresAt });
      return Promise.resolve({ fileId });
    },
    delete(url, fileId) {
      if (rows.get(url)?.fileId === fileId) rows.delete(url);
      return Promise.resolve();
    },
  };
  return { store, rows };
}

async function runLiveProviderFilesTest(t: Deno.TestContext, options: {
  real: OpenAI;
  purpose: "user_data" | "assistants";
  expiresOnProvider: boolean;
  makeAdapter: (client: OpenResponsesClient, store: ProviderFileStore) => Adapter<unknown, unknown>;
}) {
  const { client, counts, uploaded } = createCountingClient(options.real);
  const { store, rows } = createMemoryStore();
  const agent = new Agent({
    model: options.makeAdapter(client, store),
    instructions: "You are a live integration test assistant. Answer briefly and literally.",
  });
  const run = () => agent.run(HISTORY, { signal: AbortSignal.timeout(INTEGRATION_TIMEOUT_MS) });

  try {
    await t.step("first turn uploads both files and the model reads them", async () => {
      const result = await run();

      assert(PDF_TITLE.test(result.outputText), `model did not read the PDF: ${result.outputText}`);
      assertEquals(counts.uploads, 2);
      assertEquals(counts.requests, 1);
      assertEquals([...rows.keys()].sort(), [IMAGE_URL, PDF_URL].sort());

      for (const [url, row] of rows) {
        const file = await options.real.files.retrieve(row.fileId);
        assertEquals(file.purpose, options.purpose, url);
        if (options.expiresOnProvider) {
          // The provider's own expiry is the backup and must come after our 7 days.
          assertEquals((file.expires_at! - file.created_at) / 86_400, 9, url);
        } else {
          assertEquals(file.expires_at, null, url);
        }
        // Our own expiry is seven days from upload, whatever the provider does.
        const days = (row.expiresAt.getTime() - Date.now()) / 86_400_000;
        assert(days > 6.9 && days <= 7, `expected a 7 day expiry, got ${days}`);
      }
    });

    const firstIds = new Map([...rows].map(([url, row]) => [url, row.fileId]));

    await t.step("second turn reuses the stored ids without uploading", async () => {
      const result = await run();

      assert(PDF_TITLE.test(result.outputText), `model did not read the PDF: ${result.outputText}`);
      assertEquals(counts.uploads, 2);
      assertEquals(counts.requests, 2);
      for (const [url, row] of rows) assertEquals(row.fileId, firstIds.get(url), url);
    });

    await t.step("an expired row is a miss: old file deleted on the provider, new one uploaded", async () => {
      rows.set(IMAGE_URL, { fileId: firstIds.get(IMAGE_URL)!, expiresAt: new Date(Date.now() - 1) });

      const result = await run();

      assert(PDF_TITLE.test(result.outputText), `model did not read the PDF: ${result.outputText}`);
      assertEquals(counts.requests, 3);
      assertEquals(counts.uploads, 3);
      assertNotEquals(rows.get(IMAGE_URL)?.fileId, firstIds.get(IMAGE_URL));
      assertEquals(rows.get(PDF_URL)?.fileId, firstIds.get(PDF_URL));
      const gone = await options.real.files.retrieve(firstIds.get(IMAGE_URL)!).then(() => false, () => true);
      assert(gone, "the expired image file is still on the provider");
    });

    await t.step("a file gone on the provider is re-uploaded and the request resent", async () => {
      const imageIdAfterExpiry = rows.get(IMAGE_URL)?.fileId;
      // Simulate the provider deleting our file on its own.
      await options.real.files.delete(firstIds.get(PDF_URL)!);

      const result = await run();

      assert(PDF_TITLE.test(result.outputText), `model did not read the PDF: ${result.outputText}`);
      // One upload, for the PDF only, whatever the provider does with the fresh id afterwards.
      assertEquals(counts.uploads, 4);
      // The rejected request plus one resend, or up to two more if the fresh id was rejected too.
      console.log(`requests after recovery: ${counts.requests}`);
      assert(counts.requests >= 5 && counts.requests <= 7, `requests: ${counts.requests}`);
      assertNotEquals(rows.get(PDF_URL)?.fileId, firstIds.get(PDF_URL));
      assertEquals(rows.get(IMAGE_URL)?.fileId, imageIdAfterExpiry);
    });
  } finally {
    for (const id of uploaded) {
      await options.real.files.delete(id).catch(() => {});
    }
  }
}

Deno.test({
  name: "OpenAI Files API: upload, reuse, and recover from a provider miss (gpt-5.6-luna)",
  ignore: !HAS_OPENAI_KEY,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    await runLiveProviderFilesTest(t, {
      real: new OpenAI(),
      purpose: "user_data",
      expiresOnProvider: true,
      makeAdapter: (client, fileStore) => openAIModel({ model: "gpt-5.6-luna", effort: "low", client, fileStore }),
    });
  },
});

Deno.test({
  name: "Azure Files API: upload, reuse, and recover from a provider miss (gpt-5.6-luna)",
  ignore: !HAS_AZURE_KEYS,
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    const endpoint = Deno.env.get("AZURE_OPENAI_ENDPOINT")!.replace(/\/$/, "");
    await runLiveProviderFilesTest(t, {
      real: new OpenAI({ apiKey: Deno.env.get("AZURE_OPENAI_API_KEY"), baseURL: `${endpoint}/openai/v1` }),
      purpose: "assistants",
      expiresOnProvider: false,
      makeAdapter: (client, fileStore) => azureOpenAIModel({ model: "gpt-5.6-luna", effort: "low", client, fileStore }),
    });
  },
});
