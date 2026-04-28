import type Anthropic from "@anthropic-ai/sdk";
import { isStructuredOutputRetryFeedback } from "../../constants.ts";
import { normalizeToolName } from "../../tool.ts";
import type { ChatItem } from "../../types.ts";
import type { AnthropicToolMap } from "./utils.ts";

const supportedImageMimeTypes = ["image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp"];

// TODO: drop signature after 10 minutes or whatever
// Mapping between thinking response and signature since signature is meaningless cross-provider and we technically only need to include thinking for the one step
const signatureMap = new Map<string, string>();

export function rememberAnthropicReasoningSignature(content: string, signature: string) {
  signatureMap.set(content, signature);
}

export async function getAnthropicHistory(options: {
  history: ChatItem[];
  normalizedTools: AnthropicToolMap[];
  signal: AbortSignal;
}): Promise<Anthropic.Messages.MessageParam[]> {
  const anthropicHistory: Anthropic.Messages.MessageParam[] = [];
  let anthropicToolFileBuffer: Anthropic.Messages.MessageParam[] = [];

  // Put all of the history in place
  for (const historyItem of options.history) {
    switch (historyItem.type) {
      case "input_text": {
        // first, flush tool buffer
        anthropicHistory.push(...anthropicToolFileBuffer);
        anthropicToolFileBuffer = [];

        // next, append message
        anthropicHistory.push({
          role: "user",
          content: [{ type: "text", text: historyItem.content }],
        });
        break;
      }
      case "output_text": {
        // first, flush tool buffer
        anthropicHistory.push(...anthropicToolFileBuffer);
        anthropicToolFileBuffer = [];

        // next, append message
        anthropicHistory.push({
          role: isStructuredOutputRetryFeedback(historyItem.content) ? "user" : "assistant",
          content: [{ type: "text", text: historyItem.content }],
        });
        break;
      }
      case "context_summary": {
        anthropicHistory.push(...anthropicToolFileBuffer);
        anthropicToolFileBuffer = [];

        anthropicHistory.push({
          role: "user",
          content: [{ type: "text", text: historyItem.content }],
        });
        break;
      }
      case "tool_use": {
        const tool = options.normalizedTools.find((tool) => tool.original.name === historyItem.kind);
        const content = historyItem.content ? JSON.parse(historyItem.content) : {};
        anthropicHistory.push({
          role: "assistant",
          content: [{
            type: "tool_use",
            id: historyItem.tool_use_id,
            name: tool?.anthropic.name ?? normalizeToolName(historyItem.kind),
            input: tool?.compatibility ? tool.compatibility.toProvider(content) : content,
          }],
        });
        break;
      }
      case "tool_result_text": {
        anthropicHistory.push({
          role: "user",
          content: [{
            type: "tool_result",
            tool_use_id: historyItem.tool_use_id,
            content: historyItem.content,
            is_error: historyItem.content.startsWith("Error: "),
          }],
        });
        break;
      }
      case "output_reasoning": {
        // first, flush tool buffer
        anthropicHistory.push(...anthropicToolFileBuffer);
        anthropicToolFileBuffer = [];

        // next append reasoning
        const signature = signatureMap.get(historyItem.content);
        if (signature) {
          anthropicHistory.push({
            role: "assistant",
            content: [{
              type: "thinking",
              thinking: historyItem.content,
              signature,
            }],
          });
        } else {
          // no-op :( nothing we can do
        }
        break;
      }
      case "input_file":
      case "tool_result_file": {
        const pushBuffer = historyItem.type === "input_file" ? anthropicHistory : anthropicToolFileBuffer;
        if (supportedImageMimeTypes.includes(historyItem.kind)) {
          pushBuffer.push({
            role: "user",
            content: [{
              type: "image",
              source: {
                type: "url",
                url: historyItem.content,
              },
            }],
          });
        } else if (historyItem.kind === "application/pdf") {
          pushBuffer.push({
            role: "user",
            content: [
              {
                type: "document",
                source: {
                  type: "url",
                  url: historyItem.content,
                },
              },
            ],
          });
        } else if (historyItem.kind.startsWith("text/")) {
          const req = await fetch(historyItem.content, { signal: options.signal });
          const text = await req.text();

          pushBuffer.push({
            role: "user",
            content: [
              {
                type: "text",
                text: `<ant-file>${text}</ant-file>`,
              },
            ],
          });
        } else {
          throw new Error(`Anthropic models don't support the following media type: ${historyItem.kind}`);
        }
        break;
      }
      default:
        historyItem satisfies never;
    }
  }

  // Flush remaining toolFileBuffer
  anthropicHistory.push(...anthropicToolFileBuffer);
  anthropicToolFileBuffer = [];

  return anthropicHistory;
}
