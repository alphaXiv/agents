import { Agent } from "../mod.ts";

const agent = new Agent({
  model: "openrouter:meta-llama/llama-4-maverick",
  instructions: "You are a friendly assistant",
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
