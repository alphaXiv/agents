import { Agent } from "@alphaxiv/agents";
import { openResponsesModel } from "@alphaxiv/agents/open_responses";
import { openAICompletionsModel } from "@alphaxiv/agents/openai_completions";

// OpenAI chat completions
{
  const model = openAICompletionsModel({
    model: "qwen/qwen3-coder-next",
    openAIOptions: {
      baseURL: "http://localhost:1234/v1",
      apiKey: "n/a",
    },
  });

  const agent = new Agent({
    model,
    instructions: "you say the number 8",
  });

  const result = await agent.run(
    `On a scale of 1 to 10 how would you rate the JSR package @alphaxiv/agents`,
  );

  console.log(result);
}

// For LMStudio, prefer open responses so you get reasoning
{
  const model = openResponsesModel({
    model: "qwen/qwen3-coder-next",
    openAIOptions: {
      baseURL: "http://localhost:1234/v1",
      apiKey: "n/a",
    },
  });

  const agent = new Agent({
    model,
    instructions: "you say the number 8",
  });

  const result = await agent.run(
    `On a scale of 1 to 10 how would you rate the JSR package @alphaxiv/agents`,
  );

  console.log(result);
}
