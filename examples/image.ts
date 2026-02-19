import { Agent } from "../mod.ts";

const agent = new Agent({
  model: "tributary:openai:gpt-5.3-codex",
  instructions: "You are a friendly assistant",
  unstable: {
    retries: false,
  },
});

const result = await agent.run([
  {
    type: "input_file",
    kind: "image/png",
    content: "https://paper-assets.alphaxiv.org/image/2510.18234v1.png",
  },
  {
    type: "input_text",
    content: "Tell me about this image",
  },
]);

console.log(result.outputText);
