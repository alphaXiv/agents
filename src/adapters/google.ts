import {
  ApiError,
  type Content,
  type DeleteFileResponse,
  type FunctionDeclaration,
  GoogleGenAI,
} from "@google/genai";
import z from "zod";
import { assert } from "@std/assert";
import type { Tool } from "../tool.ts";
import type {
  AsyncStreamItemGenerator,
  ChatItem,
  ReasoningEffort,
} from "../types.ts";
import { crossPlatformEnv, hashString, removeDollarSchema } from "../util.ts";

const BASE_URL = "https://generativelanguage.googleapis.com";

/** No way to specify url for google api :( */
async function ensureFileUploaded(
  gemini: GoogleGenAI,
  url: string,
  mimeType: string,
  abortSignal: AbortSignal,
): Promise<string> {
  // Create a safe filename by hashing the URL
  const safeFileName = (await hashString(url)).slice(0, 40);

  try {
    // Try to get the file first to see if it exists
    await gemini.files.get({
      name: safeFileName,
      config: { abortSignal },
    });
    return safeFileName;
  } catch (err) {
    if (abortSignal.aborted) {
      throw err;
    }
    const response = await fetch(url);
    const blob = await response.blob();

    try {
      await gemini.files.upload({
        file: blob,
        config: {
          name: `files/${safeFileName}`,
          mimeType,
          abortSignal,
        },
      });
    } catch (err) {
      if (err instanceof ApiError) {
        if (
          err.message.includes(
            "generativelanguage.googleapis.com/file_storage_bytes",
          ) && err.message.includes(
            "429",
          )
        ) {
          // We've run out of file storage as cache. Delete some files, then try uploading again.
          const fileList = await gemini.files.list({
            config: { pageSize: 100, abortSignal },
          });
          let toDelete = 1000; // How many files we want to delete before attempting to upload one
          const deletionPromises: Promise<DeleteFileResponse>[] = [];
          for await (const file of fileList) {
            // Check if the file was created > 30 minutes ago
            if (
              file.name && (
                !file.createTime ||
                new Date(file.createTime).getTime() <
                  new Date().getTime() - (30 * 60 * 1000)
              )
            ) {
              // If so, delete it
              deletionPromises.push(
                gemini.files.delete({
                  name: file.name,
                  config: { abortSignal },
                }),
              );
              toDelete--;
              if (toDelete <= 0) {
                break;
              }
            }
          }
          await Promise.all(deletionPromises);
          return await ensureFileUploaded(gemini, url, mimeType, abortSignal);
        }
      }
      throw err;
    }

    return safeFileName;
  }
}

export async function getGoogleHistory(
  history: ChatItem[],
  gemini: GoogleGenAI,
  toolMap: GoogleToolMap[],
  signal: AbortSignal,
) {
  const googleHistory: Content[] = [];
  for (const historyItem of history) {
    if (historyItem.type === "input_text") {
      googleHistory.push({
        role: "user",
        parts: [{ text: historyItem.content }],
      });
    } else if (historyItem.type === "output_text") {
      googleHistory.push({
        role: "model",
        parts: [{ text: historyItem.content }],
      });
    } else if (historyItem.type === "tool_use") {
      const tool = toolMap.find((tool) =>
        tool.original.name === historyItem.kind
      );
      const content = historyItem.content
        ? JSON.parse(historyItem.content)
        : undefined;
      googleHistory.push({
        role: "model",
        parts: [{
          functionCall: {
            id: historyItem.tool_use_id,
            name: tool?.google.name ?? historyItem.kind,
            args: tool?.wrapperObject ? { content } : content,
          },
        }],
      });
    } else if (historyItem.type === "tool_result_text") {
      const toolCall = history.find((item) =>
        item.type === "tool_use" &&
        item.tool_use_id === historyItem.tool_use_id
      );
      assert(toolCall);
      googleHistory.push({
        role: "user",
        parts: [{
          functionResponse: {
            id: historyItem.tool_use_id,
            name: toolCall.type === "tool_use" ? toolCall.kind : undefined, // TODO: figure out why type narrowing is trolling me here
            response: { content: historyItem.content },
          },
        }],
      });
    } else if (
      historyItem.type === "input_file" ||
      historyItem.type === "tool_result_file"
    ) {
      const fileName = await ensureFileUploaded(
        gemini,
        historyItem.content,
        historyItem.kind,
        signal,
      ); // TODO: make this strategy configurable
      googleHistory.push({
        role: "user",
        parts: [{
          fileData: {
            fileUri: `${BASE_URL}/v1beta/files/${fileName}`,
            mimeType: historyItem.kind,
          },
        }],
      });
    } else if (historyItem.type === "output_reasoning") {
      // no-op, don't propagate reasoning
    } else {
      historyItem satisfies never;
    }
  }

  return googleHistory;
}

// TODO: ensure this list is complete
const nonReasoningModels = [
  "gemini-2.0-flash",
  "gemini-2.0-flash-001",
  "gemini-2.0-flash-lite",
  "gemini-2.0-flash-lite-001",
  "gemini-2.5-flash-image",
  "gemini-2.5-flash-image-preview",
];

// TODO: ensure this list is complete
const alwaysReasoningModels = [
  "gemini-2.5-pro",
];

type GoogleToolMap = {
  original: Tool<unknown, unknown>;
  google: FunctionDeclaration;
  /** Google silently doesn't allow non-objects at the top level but we want to. We therefore wrap the tool input with a wrapper object which need to unwrap at the output */
  wrapperObject: boolean;
  /** No parameter specified */
  isVoid: boolean;
};

// TODO: support both output schema and tools in same agent
export class GoogleAdapter<zO, zI> {
  #client: GoogleGenAI;
  #model: string;
  #output?: z.ZodType<zO, zI>;
  #normalizedTools: GoogleToolMap[];
  #reasoningEffort: ReasoningEffort;

  constructor(
    { model, output, tools, reasoningEffort }: {
      model: string;
      output?: z.ZodType<zO, zI>;
      tools: Tool<unknown, unknown>[];
      reasoningEffort: ReasoningEffort;
    },
  ) {
    this.#model = model;
    this.#output = output;
    this.#reasoningEffort = reasoningEffort;
    this.#normalizedTools = tools.map((tool) => {
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
        google: {
          name,
          parameters: isVoid ? undefined : z.toJSONSchema(
            wrapperObject
              ? z.object({ content: tool.parameters })
              : tool.parameters,
          ) as any,
          description: tool.description,
        },
        wrapperObject,
        isVoid,
      };
    });
    this.#client = new GoogleGenAI({
      apiKey: crossPlatformEnv("GEMINI_API_KEY"),
    });
  }

  async run({ history, systemPrompt, signal }: {
    systemPrompt: string;
    history: ChatItem[];
    signal: AbortSignal;
  }): Promise<ChatItem[]> {
    const googleHistory = await getGoogleHistory(
      history,
      this.#client,
      this.#normalizedTools,
      signal,
    );

    const isReasoningModel = !nonReasoningModels.includes(this.#model);

    const response = await this.#client.models.generateContent({
      model: this.#model,
      contents: googleHistory,
      config: {
        tools: this.#normalizedTools.length
          ? [{
            functionDeclarations: this.#normalizedTools.map(({ google }) =>
              google
            ),
          }]
          : undefined,
        systemInstruction: systemPrompt,
        thinkingConfig: isReasoningModel
          ? {
            includeThoughts: true,
            thinkingBudget: this.#reasoningEffort === "minimal" &&
                !alwaysReasoningModels.includes(this.#model)
              ? 0
              : undefined,
          }
          : undefined,
        responseMimeType: this.#output ? "application/json" : undefined,
        responseSchema: this.#output
          ? removeDollarSchema(z.toJSONSchema(this.#output))
          : undefined,
        abortSignal: signal,
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
            content: part.text,
          });
        } else {
          output.push({
            type: "output_text",
            content: part.text,
          });
        }
      } else if (part.functionCall) {
        const func = part.functionCall;
        const funcId = func.id ?? crypto.randomUUID();
        assert(func.name && func.args);
        const tool = this.#normalizedTools.find((tool) =>
          tool.google.name === func.name
        );
        output.push({
          type: "tool_use",
          tool_use_id: funcId,
          kind: tool?.original.name ?? func.name,
          content: tool?.isVoid ? undefined : JSON.stringify(
            tool?.wrapperObject ? func.args.content : func.args,
          ),
        });
      }
    }

    return output;
  }

  async *stream({ history, systemPrompt, signal }: {
    systemPrompt: string;
    history: ChatItem[];
    signal: AbortSignal;
  }): AsyncStreamItemGenerator {
    const googleHistory = await getGoogleHistory(
      history,
      this.#client,
      this.#normalizedTools,
      signal,
    );

    const isReasoningModel = !nonReasoningModels.includes(this.#model);

    const response = await this.#client.models.generateContentStream({
      model: this.#model,
      contents: googleHistory,
      config: {
        tools: this.#normalizedTools.length
          ? [{
            functionDeclarations: this.#normalizedTools.map(({ google }) =>
              google
            ),
          }]
          : undefined,
        systemInstruction: systemPrompt,
        thinkingConfig: isReasoningModel
          ? {
            includeThoughts: true,
            thinkingBudget: this.#reasoningEffort === "minimal" &&
                !alwaysReasoningModels.includes(this.#model)
              ? 0
              : undefined,
          }
          : undefined,
        abortSignal: signal,
      },
    });

    let lastType = "";
    let lastIndex = -1;
    for await (const item of response) {
      const parts = item?.candidates?.[0]?.content?.parts;
      if (!parts) continue;
      for (const part of parts) {
        if (part.text) {
          if (part.thought) {
            if (lastType !== "reasoning") {
              lastType = "reasoning";
              lastIndex++;
            }
            yield {
              type: "delta_output_reasoning",
              delta: part.text,
              index: lastIndex,
            };
          } else {
            if (lastType !== "text") {
              lastType = "text";
              lastIndex++;
            }
            yield {
              type: "delta_output_text",
              delta: part.text,
              index: lastIndex,
            };
          }
        } else if (part.functionCall) {
          lastType = "tool_use";
          lastIndex++;

          const func = part.functionCall;
          const funcId = func.id ?? crypto.randomUUID();
          assert(func.name && func.args);
          const tool = this.#normalizedTools.find((tool) =>
            tool.google.name === func.name
          );
          yield {
            type: "tool_use",
            tool_use_id: funcId,
            kind: tool?.original.name ?? func.name,
            content: tool?.isVoid
              ? undefined
              : (tool?.wrapperObject
                ? JSON.stringify(part.functionCall.args!.content)
                : JSON.stringify(part.functionCall.args)),
            index: lastIndex,
          };
        }
      }
    }
  }
}
