import z from "zod";
import { Agent } from "../mod.ts";
import { SidEmbeddingSearchTool, SidReportHelpfulIdsTool, SidTextSearchTool } from "../src/adapters/sid/tools.ts";

const currentDate = new Date().toISOString().slice(0, 10);

const systemPrompt =
  `You are an expert research assistant that is given a question and must use the provided search tools to find all documents needed to answer the question.

Steps:
1. Reflect on what information is needed to answer the question and use the search tools to find documents. Each document has an id.
2. Repeat step 1 until all documents necessary and sufficient to answer the question have been found. Take up to 10 turns -- you can make multiple searches per turn! Most questions will require multiple turns. Most questions require at least 5-8 search requests. Many will need more.
3. Use report_helpful_ids tool to report the most helpful document ids. List the most helpful document ids first (important!).

The interaction ends once report_helpful_ids is called. You will be scored based on whether you have found all the documents (recall: the fraction of ground-truth relevant documents that appear in your submission).

You have access to the following tools:

- search: Searches for academic papers using embeddings via the retrieval API. Returns paper titles, publication dates, votes, and a short relevant snippet from the abstract (not the full abstract) from papers matching the embedding query.
  - Arguments: query (required), limit (optional, default 5)
  - Query guidance: A short, concept-focused, clear, specific query with singular intent in natural language. Do not just assemble a series of keywords.
  - Example queries: "Methods for table extraction from scanned documents", "Multi-modal models for chart understanding", "Low-resource language text recognition approaches"
- text_search: Searches for academic papers using keyword phrases via the retrieval API. Returns paper titles, publication dates, votes, and relevant snippets from papers matching the keyword query.
  - Arguments: query (required), limit (optional, default 5), minPublicationDate (optional, ISO 8601 datetime e.g. '2025-01-01T00:00:00Z')
- report_helpful_ids: report helpful document IDs in order (most helpful first)
  - Arguments: ids (required, list of strings)

To use a tool, enclose it within <tool_call> tags with a Python dictionary containing "name" and "arguments". For example:

<tool_call>
{"name": "search", "arguments": {"query": "Methods for table extraction from scanned documents", "limit": 5}}
</tool_call>


<tool_call>
{"name": "text_search", "arguments": {"query": "document layout analysis", "limit": 5, "minPublicationDate": "2024-01-01T00:00:00Z"}}
</tool_call>

After you've received the tool responses, you can report the helpful document IDs:

<tool_call>
{"name": "report_helpful_ids", "arguments": {"ids": ["placeholder_1", "placeholder_2", "placeholder_3"]}}
</tool_call>

The current date is ${currentDate}.`;

// -------------------- Fake paper database --------------------

interface FakePaper {
  id: string;
  title: string;
  publicationDate: string;
  votes: number;
  abstract: string;
  keywords: string[];
}

const FAKE_PAPERS: FakePaper[] = [
  {
    id: "2308.13418",
    title: "Nougat: Neural Optical Understanding for Academic Documents",
    publicationDate: "2023-08-25",
    votes: 187,
    abstract:
      "Scientific knowledge is predominantly stored in books and scientific journals, often in PDF format. However, the PDF format leads to a loss of semantic information, especially for mathematical expressions. We propose Nougat, a Visual Transformer model that performs OCR on scientific documents, outputting academic markup language directly.",
    keywords: ["ocr", "pdf", "scientific", "transformer", "nougat", "document"],
  },
  {
    id: "2310.16809",
    title: "GOT: General OCR Theory — Towards OCR-2.0 via a Unified End-to-End Model",
    publicationDate: "2023-10-25",
    votes: 243,
    abstract:
      "We propose General OCR Theory (GOT), aiming to push the limits of OCR systems. GOT unifies scene text, document text, formula, chart, sheet music, and geometric shapes recognition into a single end-to-end model. Our 580M model demonstrates strong performance across diverse OCR benchmarks.",
    keywords: [
      "ocr",
      "unified",
      "end-to-end",
      "scene text",
      "document",
      "formula",
      "got",
    ],
  },
  {
    id: "2305.15393",
    title: "DocPedia: Unleashing the Power of Large Multimodal Model for Document Understanding",
    publicationDate: "2023-05-24",
    votes: 64,
    abstract:
      "This paper presents DocPedia, a novel large multimodal model for document understanding. Unlike existing approaches that rely on external OCR engines, DocPedia directly processes high-resolution document images up to 2560x2560 and performs various document understanding tasks without OCR preprocessing.",
    keywords: [
      "document",
      "multimodal",
      "ocr-free",
      "understanding",
      "high-resolution",
    ],
  },
  {
    id: "2403.02460",
    title: "TextMonkey: An OCR-Free Large Multimodal Model for Understanding Document",
    publicationDate: "2024-03-04",
    votes: 98,
    abstract:
      "We present TextMonkey, an OCR-free large multimodal model for document understanding that can process high-resolution inputs through a Shifted Window Attention mechanism. TextMonkey achieves state-of-the-art on multiple benchmarks including DocVQA, InfoVQA, and ChartQA.",
    keywords: ["ocr-free", "document", "multimodal", "textmonkey", "docvqa"],
  },
  {
    id: "2111.15664",
    title: "TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models",
    publicationDate: "2021-11-30",
    votes: 312,
    abstract:
      "Text recognition is a long-standing research problem for document digitalization. We present TrOCR, an end-to-end text recognition approach with pre-trained image Transformer and text Transformer models leveraging large-scale synthetic data and pre-training.",
    keywords: [
      "ocr",
      "transformer",
      "text recognition",
      "trocr",
      "pre-trained",
    ],
  },
  {
    id: "2304.08485",
    title: "LLaVAR: Enhanced Visual Instruction Tuning for Text-Rich Image Understanding",
    publicationDate: "2023-04-17",
    votes: 73,
    abstract:
      "Instruction-tuned large language models with visual capabilities have demonstrated impressive performance. However, these models struggle with images containing rich text. LLaVAR enhances visual instruction tuning by collecting text-rich images and generating instruction-following data.",
    keywords: ["text-rich", "instruction tuning", "visual", "llavar", "ocr"],
  },
  {
    id: "2409.01704",
    title: "OmniParser: A Unified Framework for Text Spotting, Key Information Extraction and Table Recognition",
    publicationDate: "2024-09-03",
    votes: 55,
    abstract:
      "OmniParser proposes a unified framework that jointly handles text spotting, key information extraction, and table recognition in document images. The framework uses a shared vision encoder with task-specific heads, significantly reducing computational overhead compared to pipeline approaches.",
    keywords: [
      "document",
      "ocr",
      "table recognition",
      "text spotting",
      "unified",
      "omniparser",
    ],
  },
  {
    id: "2206.01062",
    title: "Donut: Document Understanding Transformer without OCR",
    publicationDate: "2022-06-02",
    votes: 278,
    abstract:
      "Understanding document images (e.g., invoices) is a core but challenging task since it requires complex functions such as reading text and a holistic understanding of the document. We propose Donut, a model that does not require off-the-shelf OCR engines or APIs, yet shows state-of-the-art performance.",
    keywords: ["document", "ocr-free", "transformer", "donut", "understanding"],
  },
];

function searchPapers(
  query: string,
  limit: number,
  minDate?: string,
): FakePaper[] {
  const q = query.toLowerCase();
  return FAKE_PAPERS
    .filter((p) => {
      if (minDate && p.publicationDate < minDate.slice(0, 10)) return false;
      return p.keywords.some((k) => q.includes(k)) ||
        p.title.toLowerCase().includes(q) ||
        p.abstract.toLowerCase().split(" ").some((w) => q.includes(w) && w.length > 3);
    })
    .slice(0, limit);
}

function formatPaperXml(papers: FakePaper[]): string {
  if (papers.length === 0) return "No papers found matching the search query.";
  return papers
    .map(
      (p) =>
        `<doc id="${p.id}" title="${p.title}" publicationDate="${p.publicationDate}" votes="${p.votes}">\n${
          p.abstract.substring(0, 300)
        }\n</doc>`,
    )
    .join("\n");
}

// -------------------- Tools --------------------
const embeddingSearch = new SidEmbeddingSearchTool({
  description:
    "Searches for academic papers using embeddings via the retrieval API. Returns paper titles, publication dates, votes, and a short relevant snippet from the abstract.",
  parameters: z.object({
    query: z.string().describe(
      "A short, concept-focused, clear, specific query with singular intent in natural language.",
    ),
    limit: z.number().int().optional().default(5).describe(
      "The maximum number of papers to return. Default is 5.",
    ),
  }),
  execute: ({ query, limit }) => formatPaperXml(searchPapers(query, limit)),
});

const textSearch = new SidTextSearchTool({
  description:
    "Searches for academic papers using keyword phrases via the retrieval API. Returns paper titles, publication dates, votes, and relevant snippets from papers matching the keyword query.",
  parameters: z.object({
    query: z.string().describe(
      "Keyword phrase to search for in academic papers.",
    ),
    limit: z.number().int().optional().default(5).describe(
      "The maximum number of papers to return. Default is 5.",
    ),
    minPublicationDate: z.string().optional().describe(
      "Only return papers published after this date. ISO 8601 datetime format, e.g. '2025-01-01T00:00:00Z'.",
    ),
  }),
  execute: ({ query, limit, minPublicationDate }) => formatPaperXml(searchPapers(query, limit, minPublicationDate)),
});

const reportHelpfulIds = new SidReportHelpfulIdsTool();

// -------------------- The actual run --------------------
const agent = new Agent({
  model: "sid:sid-1",
  instructions: systemPrompt,
  tools: [embeddingSearch, textSearch, reportHelpfulIds],
});

const { output } = await agent.run("Find me OCR papers");
console.log("OCR paper ids:", output);
