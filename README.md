# @alphaxiv/agents

A TypeScript library to build agents easily, while not limiting more powerful
uses of advanced LLM features.

## Example

### Deep research agent

```ts
import z from "zod";
import { Agent, Tool, mapAsync } from "@alphaxiv/agents";

const searchGenerator = new Agent({
  model: "anthropic:claude-4.5-sonnet",
  name: "Generating Relevant Search Terms",
  instructions:
    "You will be given a query, and your goal is to generate relevant search terms for the query. For example, for a query like 'SFT vs RL', you should generate 'SFT', 'RL', 'SFT weaknesses', 'SFT strengths', and so on. Your goal will be to generate around 20-30 queries.",
});

const subresearcher = new Agent({
  model: "anthropic:claude-4.5-sonnet",
  name: "Determining article relevancy",
  instructions:
    `You will be given a query and a document. Your task is the following:
1. Determine if the document is relevant to the query – If the document isn't relevant to the query, simply output <irrelevant-document> and do not execute step 2.
2. If the document IS relevant, write up a justification for why it's relevant and what the important tidbits are including a summary of the documents`,
});

const reducer = new Agent({
  model: "anthropic:claude-4.5-sonnet",
  instructions: ``,
  name: (input) => `Summarizing ${input.length} results`,
});

const fetchTool = await Tool({
  name: "Searching for input...",
  input: z.string().describe("Search to pass to the google search API"),
  tool: async (input: string) => {
    return await search(query);
  }
});

const deepResearchWorkflow = await Tool({
  name: "Deep research ",
  input: z.string().describe("Top level deep research query"),
  tool: async (input: string) => {
    const queries = await searchGenerator.run(input);
    const searchResults = await mapAsync(async (query) => {
      const result = await fetchTool.run(query);
      return await subresearcher.run({
        input,
        searchResult,
      });
    }));
    return await reducer.runReduce(searchResults);
  }
});

const agent = new Agent({
  model: "anthropic:claude-4.5-sonnet",
  instructions:
    "You are a helpful assistant capable of performing deep research operations",
  tools: [deepResearchWorkflow],
});

// if (import.meta.main) agent.cli()

// const result = await agent.run("Can you summarize the current debate on SFT vs RL?");

// console.log(result.finalOutput);
```
