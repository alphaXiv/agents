import z from "zod";
import { type ExecuteFunc, ModelOutput, Tool } from "../../tool.ts";

export class SidEmbeddingSearchTool<zO, zI> extends Tool<zO, zI> {
  constructor({ description, parameters, execute, retries }: {
    description: string;
    parameters: z.ZodType<zO, zI>;
    execute: ExecuteFunc<zO>;
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

export class SidTextSearchTool<zO, zI> extends Tool<zO, zI> {
  constructor({ description, parameters, execute, retries }: {
    description: string;
    parameters: z.ZodType<zO, zI>;
    execute: ExecuteFunc<zO>;
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
