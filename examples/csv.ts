import { Agent } from "../mod.ts";

const agent = new Agent({
  model: "openrouter:openai/gpt-oss-20b",
  instructions: "You are a friendly assistant",
});

const result = await agent.run([
  {
    type: "input_file",
    kind: "text/csv",
    content: "https://people.sc.fsu.edu/~jburkardt/data/csv/addresses.csv",
  },
  {
    type: "input_text",
    content: "Who lives where?",
  },
]);

console.log(result.outputText);
