/**
 * Adapter implementation for Gemini APIs using `@google/genai`.
 * ```ts
 * const adapter = googleAdapter({
 *   name: "google", // display name according to who is running the API
 *   url: "https://generativelanguage.googleapis.com",
 *   apiKey: process.env.GEMINI_API_KEY,
 * });
 *
 * const agent = new Agent({
 *   adapter,
 *   model: "gemini-2.0-flash"
 * });
 * ```
 * If you are using the default provider and API key environment variable, you
 * can omit the `adapter` property and use a unified model string like
 * `model: "google:gemini-2.0-flash"`.
 * @module
 */
import {
  ApiError,
  type Content,
  type DeleteFileResponse,
  type FunctionDeclaration,
  GoogleGenAI,
} from "@google/genai";
import z from "zod";
import { assert } from "@std/assert";
import type {
  Adapter,
  AdapterStreamOptions,
  AdapterTypeOptions,
} from "../adapters.ts";
import type { Tool } from "../tool.ts";
import type {
  AdapterStreamIterator,
  ChatItem,
  ChatItemToolUse,
  ReasoningEffort,
} from "../types.ts";
import { hashString, removeDollarSchema } from "../util.ts";

// TODO: drop signature after 10 minutes or whatever
// Mapping between function call and signature since signature is meaningless cross-provider and we technically only need to include thinking for the one step
const signatureMap = new Map<string, string>();

function getGoogleFileBaseUrl(url: string) {
  return new URL(
    "v1beta/files/",
    url.endsWith("/") ? url : `${url}/`,
  ).toString();
}

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

async function getGoogleHistory(
  history: ChatItem[],
  gemini: GoogleGenAI,
  apiBaseUrl: string,
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
      const thoughtSignature = signatureMap.get(historyItem.tool_use_id) ??
        "context_engineering_is_the_way_to_go"; // https://ai.google.dev/gemini-api/docs/gemini-3?thinking=high#migrating_from_other_models
      googleHistory.push({
        role: "model",
        parts: [{
          functionCall: {
            id: historyItem.tool_use_id,
            name: tool?.google.name ?? historyItem.kind,
            args: tool?.wrapperObject ? { content } : content,
          },
          thoughtSignature,
        }],
      });
    } else if (historyItem.type === "tool_result_text") {
      const toolCall = history.find((item) =>
        item.type === "tool_use" &&
        item.tool_use_id === historyItem.tool_use_id
      ) as ChatItemToolUse | undefined;
      assert(toolCall);
      const definition = toolMap.find((x) => x.original.name === toolCall.kind);
      assert(definition);
      googleHistory.push({
        role: "user",
        parts: [{
          functionResponse: {
            id: historyItem.tool_use_id,
            name: definition.google.name,
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
            fileUri: new URL(fileName, getGoogleFileBaseUrl(apiBaseUrl))
              .toString(),
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
  "gemini-3-pro-preview",
  "gemini-3-pro",
];

type GoogleToolMap = {
  original: Tool<unknown, unknown, unknown>;
  google: FunctionDeclaration;
  /** Google silently doesn't allow non-objects at the top level but we want to. We therefore wrap the tool input with a wrapper object which need to unwrap at the output */
  wrapperObject: boolean;
  /** No parameter specified */
  isVoid: boolean;
};

function normalizeGoogleTools(
  tools: Tool<unknown, unknown, unknown>[],
): GoogleToolMap[] {
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
      google: {
        name,
        parameters: isVoid ? undefined : z.toJSONSchema(
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

function getGoogleThinking(
  model: string,
  reasoningEffort: ReasoningEffort,
) {
  const isReasoningModel = !nonReasoningModels.includes(model);
  return isReasoningModel
    ? {
      includeThoughts: true,
      thinkingBudget: reasoningEffort === "minimal" &&
          !alwaysReasoningModels.includes(model)
        ? 0
        : undefined,
    }
    : undefined;
}

export interface GoogleAdapterOptions {
  name: string;
  url: string;
  apiKey: string;
}

export function googleAdapter<Models extends string>(
  options: GoogleAdapterOptions & AdapterTypeOptions<Models>,
): Adapter<Models> {
  const google = new GoogleGenAI({
    apiKey: options.apiKey,
    httpOptions: {
      baseUrl: options.url,
    },
  });

  async function* stream<zO, zI>({
    model,
    output,
    tools,
    reasoningEffort,
    systemPrompt,
    history,
    signal,
  }: AdapterStreamOptions<zO, zI, Models>): AdapterStreamIterator {
    const normalizedTools = normalizeGoogleTools(tools);
    const googleHistory = await getGoogleHistory(
      history,
      google,
      options.url,
      normalizedTools,
      signal,
    );

    const response = await google.models.generateContentStream({
      model,
      contents: googleHistory,
      config: {
        tools: normalizedTools.length
          ? [{
            functionDeclarations: normalizedTools.map(({ google }) => google),
          }]
          : undefined,
        systemInstruction: systemPrompt,
        thinkingConfig: getGoogleThinking(model, reasoningEffort),
        responseMimeType: output ? "application/json" : undefined,
        responseSchema: output
          ? removeDollarSchema(z.toJSONSchema(output))
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
          assert(func.name);
          const tool = normalizedTools.find((tool) =>
            tool.google.name === func.name
          );

          if (part.thoughtSignature) {
            signatureMap.set(funcId, part.thoughtSignature);
          }

          yield {
            type: "tool_use",
            tool_use_id: funcId,
            kind: tool?.original.name ?? func.name,
            content: tool?.isVoid ? undefined : JSON.stringify(
              tool?.wrapperObject ? func.args?.content : func.args,
            ),
            index: lastIndex,
          };
        }
      }
    }
  }

  return { name: "google", stream };
}
