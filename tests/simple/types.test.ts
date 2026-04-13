import {
  Agent,
  ModelOutput,
  SidEmbeddingSearchTool,
  SidReportHelpfulIdsTool,
  SidTextSearchTool,
} from "@alphaxiv/agents";
import z from "zod";
import { assertType, type IsExact } from "@std/testing/types";

void async function sid1TypeTests() {
  const textSearch = new SidTextSearchTool({
    description: "search text stuff",
    parameters: z.object({
      query: z.string(),
    }),
    execute: () => "yo mista white",
  });
  const textSearch2 = new SidTextSearchTool({
    description: "search text stuff",
    parameters: z.object({
      query: z.string(),
    }),
    execute: () => new ModelOutput(123),
  });
  const embedding = new SidEmbeddingSearchTool({
    description: "search text stuff",
    parameters: z.object({
      query: z.string(),
    }),
    execute: () => new ModelOutput(true as const),
  });
  const reportHelpfulIds = new SidReportHelpfulIdsTool();
  {
    const agent = new Agent({
      model: "sid:sid-1",
      instructions: "you aact like mrbeast",
      tools: [textSearch, reportHelpfulIds],
    });
    const { output } = await agent.run("fadsfsda");
    assertType<IsExact<typeof output, string[] | undefined>>(true);
  }
  {
    const agent = new Agent({
      model: "sid:sid-1",
      instructions: "you aact like mrbeast",
      tools: [textSearch, textSearch2, reportHelpfulIds],
    });
    const { output } = await agent.run("fadsfsda");
    assertType<IsExact<typeof output, number | string[] | undefined>>(true);
  }
  {
    const agent = new Agent({
      model: "sid:sid-1",
      instructions: "",
      tools: [textSearch, textSearch2, reportHelpfulIds, embedding],
    });
    const { output } = await agent.run("fadsfsda");
    assertType<IsExact<typeof output, number | string[] | undefined | true>>(true);
  }
};
