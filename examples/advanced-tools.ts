import z from "zod";
import { Agent, Tool } from "../mod.ts";

const getImage = new Tool({
  name: "Get image...",
  description: "Get an image",
  parameters: z.string().describe("Query parameter, ignored"),
  execute: () => {
    return [
      {
        type: "tool_result_text",
        content: "Below is the image I was talking about",
      },
      {
        type: "tool_result_file",
        kind: "image/png",
        content: "https://paper-assets.alphaxiv.org/image/2510.18234v1.png",
      },
    ];
  },
});

const agent = new Agent({
  model: "anthropic:claude-haiku-4-5",
  instructions: "You are a friendly assistant",
  tools: [getImage],
});

const result = await agent.run([
  {
    type: "input_text",
    content: "Get the image (using your tool) and tell me about it!",
  },
]);

console.log(result.outputText);
