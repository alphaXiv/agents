import { Agent } from "../mod.ts";

const agent = new Agent({
  model: "openrouter:meta-llama/llama-4-maverick",
  instructions: "You are a friendly assistant",
});

const result = await agent.run([
  {
    type: "input_file",
    kind: "application/pdf",
    content: "https://arxiv.org/pdf/2511.01419v1",
  },
  {
    type: "input_text",
    content: "Who wrote this paper?",
  },
]);

console.log(result.outputText);
