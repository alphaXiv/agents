import { Agent, GeminiModel } from "../mod.ts";

const agent = new Agent({
  model: new GeminiModel({ model: "gemini-3.1-pro-preview" }),
  //new TributaryModel({ model: "openai:gpt-5.3-codex" }),
  instructions: "You are a friendly assistant",
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
