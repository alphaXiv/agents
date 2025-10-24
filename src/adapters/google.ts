import {
  type Content,
  type FunctionDeclaration,
  GoogleGenAI,
} from "@google/genai";
import z from "zod";
import { assert } from "@std/assert";
import type { Tool } from "../tool.ts";
import type { ChatItem } from "../types.ts";
import { crossPlatformEnv, removeDollarSchema } from "../util.ts";

// TODO: support both output schema and tools in same agent
export class GoogleAdapter<zO, zI> {
  #client: GoogleGenAI;
  #model: string;
  #output?: z.ZodType<zO, zI>;
  #normalizedTools: {
    original: Tool<unknown, unknown>;
    google: FunctionDeclaration;
    /** Google silently doesn't allow non-objects at the top level but we want to. We therefore wrap the tool input with a wrapper object which need to unwrap at the output */
    wrapperObject: boolean;
  }[];

  constructor(
    { model, output, tools }: {
      model: string;
      output?: z.ZodType<zO, zI>;
      tools: Tool<unknown, unknown>[];
    },
  ) {
    this.#model = model;
    this.#output = output;
    this.#normalizedTools = tools.map((tool) => {
      let name = tool.name.toLowerCase().replaceAll(" ", "_").replace(
        /[^a-zA-Z0-9_-]/g,
        "",
      );
      if (!/^[a-zA-Z_]/.test(name)) {
        name = "_" + name; // Ensure name starts with letter or underscore
      }
      name = name.slice(0, 64); // Limit to 64 characters

      const wrapperObject = !(tool.parameters instanceof z.ZodObject);

      return {
        original: tool,
        google: {
          name,
          parameters: z.toJSONSchema(
            wrapperObject
              ? z.object({ content: tool.parameters })
              : tool.parameters,
          ) as any,
          description: tool.description,
        },
        wrapperObject,
      };
    });
    this.#client = new GoogleGenAI({
      apiKey: crossPlatformEnv("GEMINI_API_KEY"),
    });
  }

  async run({ history, systemPrompt }: {
    systemPrompt: string;
    history: ChatItem[];
  }): Promise<ChatItem[]> {
    const googleHistory: Content[] = [];
    for (const historyItem of history) {
      if (historyItem.type === "input_text") {
        googleHistory.push({
          role: "user",
          parts: [{ text: historyItem.text }],
        });
      } else if (historyItem.type === "output_text") {
        googleHistory.push({
          role: "model",
          parts: [{ text: historyItem.text }],
        });
      } else if (historyItem.type === "tool_use") {
        const tool = this.#normalizedTools.find((tool) =>
          tool.original.name === historyItem.name
        );
        assert(tool);
        const content = JSON.parse(historyItem.input);
        googleHistory.push({
          role: "model",
          parts: [{
            functionCall: {
              id: historyItem.id,
              name: tool.google.name,
              args: tool.wrapperObject ? { content } : content,
            },
          }],
        });
      } else if (historyItem.type === "tool_result") {
        const toolCall = history.find((item) =>
          item.type === "tool_use" && item.id === historyItem.tool_use_id
        );
        assert(toolCall);
        googleHistory.push({
          role: "user",
          parts: [{
            functionResponse: {
              id: historyItem.tool_use_id,
              name: toolCall.type === "tool_use" ? toolCall.name : undefined, // TODO: figure out why type narrowing is trolling me here
              response: { content: historyItem.content },
            },
          }],
        });
      } else if (historyItem.type === "output_reasoning") {
        // no-op, don't propagate reasoning
      } else {
        historyItem satisfies never;
      }
    }

    const isReasoningModel = true; // TODO: implement

    const response = await this.#client.models.generateContent({
      model: this.#model,
      contents: googleHistory,
      config: {
        tools: this.#normalizedTools
          ? [{
            functionDeclarations: this.#normalizedTools.map(({ google }) =>
              google
            ),
          }]
          : undefined,
        systemInstruction: systemPrompt,
        thinkingConfig: isReasoningModel
          ? { includeThoughts: true }
          : undefined,
        responseMimeType: this.#output ? "application/json" : undefined,
        responseSchema: this.#output
          ? removeDollarSchema(z.toJSONSchema(this.#output))
          : undefined,
      },
    });

    const output: ChatItem[] = [];

    const candidate = response.candidates?.[0];
    assert(candidate);

    for (const part of candidate.content?.parts ?? []) {
      if (part.text) {
        if (part.thought) {
          output.push({
            type: "output_reasoning",
            text: part.text,
          });
        } else {
          output.push({
            type: "output_text",
            text: part.text,
          });
        }
      } else if (part.functionCall) {
        const func = part.functionCall;
        const funcId = func.id ?? crypto.randomUUID();
        console.log(func);
        assert(func.name && func.args);
        const tool = this.#normalizedTools.find((tool) =>
          tool.google.name === func.name
        );
        assert(tool);
        output.push({
          type: "tool_use",
          id: funcId,
          name: tool.original.name,
          input: JSON.stringify(
            tool.wrapperObject ? func.args.content : func.args,
          ),
        });
      }
    }

    return output;
  }
}
