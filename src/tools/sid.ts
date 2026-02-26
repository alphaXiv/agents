import z from "zod";
import { type ExecuteFunc, Tool } from "../tool.ts";

const sidEmbeddingSearchParameters = z.object({
  query: z
    .string()
    .describe(
      "A short, concept-focused, clear, specific query with singular intent in natural language. Do not just assemble a series of keywords."
    ),
  limit: z
    .number()
    .int()
    .optional()
    .default(5)
    .describe("The maximum number of papers to return. Default is 5."),
});

type SidEmbeddingSearchParams = z.output<typeof sidEmbeddingSearchParameters>;

export class SidEmbeddingSearchTool extends Tool<SidEmbeddingSearchParams, z.input<typeof sidEmbeddingSearchParameters>> {
  constructor({ description, execute, retries }: {
    description: string;
    execute: ExecuteFunc<SidEmbeddingSearchParams>;
    retries?: number;
  }) {
    super({
      name: "search",
      description,
      parameters: sidEmbeddingSearchParameters,
      execute,
      retries
    })
  }
}

const sidTextSearchParameters = z.object({
  query: z.string().describe("Keyword phrase to search for in academic papers."),
  limit: z
    .number()
    .int()
    .optional()
    .default(5)
    .describe("The maximum number of papers to return. Default is 5."),
  minPublicationDate: z
    .string()
    .optional()
    .describe(
      "Only return papers published after this date. ISO 8601 datetime format, e.g. '2025-01-01T00:00:00Z'."
    ),
});

type SidTextSearchParams = z.output<typeof sidTextSearchParameters>;

export class SidTextSearchTool extends Tool<SidTextSearchParams, z.input<typeof sidTextSearchParameters>> {
  constructor({ description, execute, retries }: {
    description: string;
    execute: ExecuteFunc<SidTextSearchParams>;
    retries?: number;
  }) {
    super({
      name: "text_search",
      description,
      parameters: sidTextSearchParameters,
      execute,
      retries
    })
  }
}

const sidReportHelpfulIdsParameters = z.object({
  ids: z.array(z.string()).describe("Document IDs ordered from most helpful to least helpful."),
});

type SidReportHelpfulIdsParams = z.output<typeof sidReportHelpfulIdsParameters>;

export class SidReportHelpfulIdsTool extends Tool<SidReportHelpfulIdsParams, z.input<typeof sidReportHelpfulIdsParameters>> {
  constructor({ execute }: {
    execute: ExecuteFunc<SidReportHelpfulIdsParams>;
  }) {
    super({
      name: "report_helpful_ids",
      description: "Report helpful document IDs in ranked order.",
      parameters: sidReportHelpfulIdsParameters,
      execute,
    })
  }
}
