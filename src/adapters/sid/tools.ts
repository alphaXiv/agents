import z from "zod";
import { type ExecuteFunc, ModelOutput, Tool } from "../../tool.ts";

export class SidEmbeddingSearchTool<zO = unknown, zI = unknown, ModelOutput = unknown>
  extends Tool<zO, zI, ModelOutput> {
  constructor({ description, parameters, execute, retries }: {
    description: string;
    parameters: z.ZodType<zO, zI>;
    execute: ExecuteFunc<zO, ModelOutput>;
    retries?: number;
  }) {
    super({
      name: "search",
      description,
      parameters,
      execute,
      retries,
    });
  }
}

export class SidTextSearchTool<zO = unknown, zI = unknown, ModelOutput = unknown> extends Tool<zO, zI, ModelOutput> {
  constructor({ description, parameters, execute, retries }: {
    description: string;
    parameters: z.ZodType<zO, zI>;
    execute: ExecuteFunc<zO, ModelOutput>;
    retries?: number;
  }) {
    super({
      name: "text_search",
      description,
      parameters,
      execute,
      retries,
    });
  }
}

export class SidReportHelpfulIdsTool extends Tool<
  { ids: string[] },
  { ids: string[] },
  string[]
> {
  constructor() {
    super({
      name: "report_helpful_ids",
      description: "Report helpful document IDs in ranked order.",
      parameters: z.object({
        ids: z.array(z.string()).describe("Document IDs ordered from most helpful to least helpful."),
      }),
      execute: ({ ids }) => new ModelOutput(ids),
    });
  }
}
