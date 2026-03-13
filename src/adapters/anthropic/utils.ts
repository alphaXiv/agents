import type Anthropic from "@anthropic-ai/sdk";
import z from "zod";
import type { Tool as AnthropicTool } from "@anthropic-ai/sdk/resources/messages/messages";
import type { ChatItem } from "../../types.ts";
import type { Tool } from "../../tool.ts";

export type AnthropicToolMap = {
  original: Tool<unknown, unknown, unknown>;
  anthropic: AnthropicTool;
  /** Anthropic doesn't allow non-objects at the top level but we want to. We therefore wrap the tool input with a wrapper object which need to unwrap at the output */
  wrapperObject: boolean;
  /** No parameter specified */
  isVoid: boolean;
};

const supportedImageMimeTypes = [
  "image/jpeg",
  "image/jpg",
  "image/png",
  "image/gif",
  "image/webp",
];

// TODO: drop signature after 10 minutes or whatever
// Mapping between thinking response and signature since signature is meaningless cross-provider and we technically only need to include thinking for the one step
export const signatureMap = new Map<string, string>();

export async function getAnthropicHistory(
  history: ChatItem[],
  normalizedTools: AnthropicToolMap[],
  signal: AbortSignal,
) {
  const anthropicHistory: Anthropic.Messages.MessageParam[] = [];
  let anthropicToolFileBuffer: Anthropic.Messages.MessageParam[] = [];

  // Put all of the history in place
  for (const historyItem of history) {
    if (historyItem.type === "input_text") {
      // first, flush tool buffer
      anthropicHistory.push(...anthropicToolFileBuffer);
      anthropicToolFileBuffer = [];

      // next, append message
      anthropicHistory.push({
        role: "user",
        content: [{ type: "text", text: historyItem.content }],
      });
    } else if (historyItem.type === "output_text") {
      // first, flush tool buffer
      anthropicHistory.push(...anthropicToolFileBuffer);
      anthropicToolFileBuffer = [];

      // next, append message
      anthropicHistory.push({
        role: "assistant",
        content: [{ type: "text", text: historyItem.content }],
      });
    } else if (historyItem.type === "tool_use") {
      const tool = normalizedTools.find((tool) =>
        tool.original.name === historyItem.kind
      );
      const content = historyItem.content
        ? JSON.parse(historyItem.content)
        : {};
      anthropicHistory.push({
        role: "assistant",
        content: [{
          type: "tool_use",
          id: historyItem.tool_use_id,
          name: tool?.anthropic.name ?? historyItem.kind,
          input: tool?.wrapperObject ? { content } : content,
        }],
      });
    } else if (historyItem.type === "tool_result_text") {
      anthropicHistory.push({
        role: "user",
        content: [{
          type: "tool_result",
          tool_use_id: historyItem.tool_use_id,
          content: historyItem.content,
          is_error: historyItem.content.startsWith("Error: "),
        }],
      });
    } else if (historyItem.type === "output_reasoning") {
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
    } else if (
      historyItem.type === "input_file" ||
      historyItem.type === "tool_result_file"
    ) {
      const pushBuffer = historyItem.type === "input_file"
        ? anthropicHistory
        : anthropicToolFileBuffer;
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
        const req = await fetch(historyItem.content, { signal });
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
        throw new Error(
          "Anthropic models don't support the following media type: " +
            historyItem.kind,
        );
      }
    } else {
      historyItem satisfies never;
    }
  }

  // Flush remaining toolFileBuffer
  anthropicHistory.push(...anthropicToolFileBuffer);
  anthropicToolFileBuffer = [];

  return anthropicHistory;
}

export function normalizeAnthropicTools(
  tools: Tool<unknown, unknown, unknown>[],
): AnthropicToolMap[] {
  return tools.map((tool) => {
    let name = tool.name.toLowerCase().replaceAll(" ", "_").replace(
      /[^a-zA-Z0-9_-]/g,
      "",
    );
    if (!/^[a-zA-Z_]/.test(name)) {
      name = "_" + name; // Ensure name starts with letter or underscore
    }
    name = name.slice(0, 64); // Limit to 64 characters

    const isVoid = tool.parameters instanceof z.ZodVoid;
    const wrapperObject = !isVoid &&
      !(tool.parameters instanceof z.ZodObject);

    return {
      original: tool,
      anthropic: {
        name,
        input_schema: isVoid ? { type: "object" } : z.toJSONSchema(
          wrapperObject
            ? z.object({ content: tool.parameters })
            : tool.parameters,
          // deno-lint-ignore no-explicit-any
        ) as any,
        description: tool.description,
      },
      wrapperObject,
      isVoid,
    };
  });
}
