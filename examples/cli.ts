import z from "zod";
import { Agent, Tool } from "../mod.ts";

const search = new Tool({
  name: "Searching the internet...",
  description: "Use when you want to search the internet",
  parameters: z.string().describe("Query parameter"),
  execute: ({ param }) => {
    if (param === "cats") {
      return JSON.stringify(["bingus.com", "bungus.com"]);
    }
    return "wtfrick.com";
  },
});

const agent = new Agent({
  model: "anthropic:claude-haiku-4-5",
  instructions:
    "You are an expert at extracting out the title and abstract from raw text from the pdf of a research paper. The user will give you the raw text.",
  // tools: [search],
  output: z.object({
    title: z.string().describe("Title of the paper"),
    abstract: z.string().describe("Abstract of paper"),
  }),
});

console.log(
  await agent.run(`
Reasoning with Sampling: Your Base Model is Smarter Than You Think Aayush Karan1, Yilun Du1 1Harvard University  Website  Code Abstract Frontier reasoning models have exhibited incredible capabilities across a wide array of disciplines, driven by posttraining large language models (LLMs) with reinforcement learning (RL). However, despite the widespread success of this paradigm, much of the literature has been devoted to disentangling truly novel behaviors that emerge during RL but are not present in the base models. In our work, we approach this question from a different angle, instead asking whether comparable reasoning capabilites can be elicited from base models at inference time by pure sampling, without any additional training. Inspired by Markov chain Monte Carlo (MCMC) techniques for sampling from sharpened distributions, we propose a simple iterative sampling algorithm leveraging the base models’ own likelihoods. Over different base models, we show that our algorithm offers substantial boosts in reasoning that nearly match and even outperform those from RL on a wide variety of single-shot tasks, including MATH500, HumanEval, and GPQA. Moreover, our sampler avoids the collapse in diversity over multiple samples that is characteristic of RL-posttraining. Crucially, our method does not require training, curated datasets, or a verifier, suggesting broad applicability beyond easily verifiable domains. 1 Introduction Reinforcement learning (RL) has become the dominant paradigm for enhancing the reasoning capabilities of large language models (LLMs) [Guo et al., 2025, Hu et al., 2025]. Equipped with a reward signal that is typically automatically verifiable, popular RL techniques have been successfully applied to posttrain frontier models, leading to sizeable performance gains in domains like math, coding, and science [Hendrycks et al., 2021, Li et al., 2022, Rein et al., 2024]. Despite the widespread empirical success of RL for LLMs, a large body of literature has centered around the following question: are the capabilities that emerge during RL-posttraining fundamentally novel behaviors that are not present in the base models? This is the question of distribution sharpening [He et al., 2025, Shao et al., 2025, Yue et al., 2025]: that is, whether the posttrained distribution is simply a “sharper” version of the base model distribution, instead of placing mass on reasoning traces the base model is unlikely to generate. Several works point towards the difficulty in learning new capabilities with RL-posttraining. He et al. [2025], Song et al. [2025] compare the pass@k (multi-shot) scores of base models with posttrained models, finding that for large k, base models actually outperform while the latter suffer from degraded generation diversity. In such cases, RL appears to redistribute pass@k performance to single-shot performance at the expense of multi-shot reasoning. Yue et al. [2025] also notes that the reasoning traces post-RL are tightly concentrated at high likelihoods/confidences under the base model, seemingly drawing from existing high-likelihood capabilities. We illustrate this point in our own experiments in Figure 4. Regardless, the advantage of RL-posttraining for single-shot reasoning has remained, as of yet, undeniable.`),
);

// await agent.cli();
