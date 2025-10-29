import { Agent } from "../mod.ts";

const agent = new Agent({
  model: "google:gemini-2.5-flash",
  instructions: "You are a friendly assistant",
});

const result = await agent.run([
  {
    type: "input_file",
    kind: "application/pdf",
    content: "https://arxiv.org/pdf/2510.18866v1",
  },
  {
    type: "input_text",
    content: "Who wrote this paper?",
  },
]);

console.log(result.outputText);
