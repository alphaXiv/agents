import type {
  Content,
  DeleteFileResponse,
  GenerateContentResponseUsageMetadata,
  GoogleGenAI,
  ThinkingConfig,
} from "@google/genai";
import { assert } from "@std/assert";
import type { AdapterStreamIterator, ChatItem, ChatItemToolUse } from "../../types.ts";
import { hashString } from "../../util.ts";
import { Adapter, type AdapterStreamOptions } from "../adapter.ts";
import type { GoogleModels } from "./models.ts";
import { type GoogleToolMap, normalizeGoogleTools } from "./tools.ts";

// TODO: drop signature after 10 minutes or whatever
// Mapping between function call and signature since signature is meaningless cross-provider and we technically only need to include thinking for the one step
const signatureMap = new Map<string, string>();

function getGoogleFileBaseUrl(url: string) {
  return new URL("v1beta/files/", url.endsWith("/") ? url : `${url}/`).toString();
}

export interface GoogleGenAiAdapterOptions<TModel extends GoogleModels> {
  model: TModel;
  thinkingConfig: ThinkingConfig;
  baseUrl?: string;
  client: GoogleGenAI;
}

export abstract class GoogleGenAiAdapter<TModel extends GoogleModels> extends Adapter<TModel> {
  #client: GoogleGenAI;
  #baseUrl: string;
  #thinkingConfig: ThinkingConfig;

  constructor(options: GoogleGenAiAdapterOptions<TModel>) {
    super(options);
    this.#client = options.client;
    this.#baseUrl = options.baseUrl ?? "https://generativelanguage.googleapis.com";
    this.#thinkingConfig = options.thinkingConfig;
  }

  async ensureFileUploaded(url: string, mimeType: string, abortSignal: AbortSignal): Promise<string> {
    // Create a safe filename by hashing the URL
    const safeFileName = (await hashString(url)).slice(0, 40);

    // Try to get the file first to see if it exists
    try {
      await this.#client.files.get({
        name: safeFileName,
        config: { abortSignal },
      });
      return safeFileName;
    } catch (error) {
      if (abortSignal.aborted) {
        throw error;
      }
    }

    // If the file doesn't exist, upload it
    try {
      const response = await fetch(url);
      const blob = await response.blob();

      await this.#client.files.upload({
        file: blob,
        config: {
          name: `files/${safeFileName}`,
          mimeType,
          abortSignal,
        },
      });
    } catch (error) {
      if (
        !(error instanceof Error) || (
          error.message.includes("generativelanguage.googleapis.com/file_storage_bytes") &&
          error.message.includes("429")
        )
      ) {
        throw error;
      }

      // We've run out of file storage as cache. Delete some files, then try uploading again.
      const fileList = await this.#client.files.list({
        config: { pageSize: 100, abortSignal },
      });

      let toDelete = 1000; // How many files we want to delete before attempting to upload one
      const deletionPromises: Promise<DeleteFileResponse>[] = [];
      for await (const file of fileList) {
        // Check if the file was created > 30 minutes ago
        if (file.name && (!file.createTime || new Date(file.createTime).getTime() < Date.now() - (30 * 60 * 1000))) {
          // If so, delete it
          deletionPromises.push(
            this.#client.files.delete({
              name: file.name,
              config: { abortSignal },
            }),
          );
          if (--toDelete <= 0) break;
        }
      }
      await Promise.all(deletionPromises);
      return await this.ensureFileUploaded(url, mimeType, abortSignal);
    }

    return safeFileName;
  }

  async getHistory(history: ChatItem[], toolMap: GoogleToolMap[], signal: AbortSignal): Promise<Content[]> {
    const googleHistory: Content[] = [];
    for (const item of history) {
      switch (item.type) {
        case "input_text":
          googleHistory.push({ role: "user", parts: [{ text: item.content }] });
          break;
        case "output_text":
          googleHistory.push({ role: "model", parts: [{ text: item.content }] });
          break;
        case "context_summary":
          googleHistory.push({ role: "user", parts: [{ text: item.content }] });
          break;
        case "tool_use": {
          const tool = toolMap.find((tool) => tool.original.normalizedName === item.kind);
          const content = item.content ? JSON.parse(item.content) : undefined;
          // Magic word comes from https://ai.google.dev/gemini-api/docs/gemini-3?thinking=high#migrating_from_other_models
          const thoughtSignature = signatureMap.get(item.tool_use_id) ?? "context_engineering_is_the_way_to_go";

          googleHistory.push({
            role: "model",
            parts: [{
              functionCall: {
                id: item.tool_use_id,
                name: tool?.google.name ?? item.kind,
                args: tool?.wrapperObject ? { content } : content,
              },
              thoughtSignature,
            }],
          });
          break;
        }
        case "tool_result_text": {
          const toolCall = history.find((candidate): candidate is ChatItemToolUse =>
            candidate.type === "tool_use" &&
            candidate.tool_use_id === item.tool_use_id
          );
          assert(toolCall);

          const definition = toolMap.find((x) => x.original.normalizedName === toolCall.kind);
          assert(definition);

          googleHistory.push({
            role: "user",
            parts: [{
              functionResponse: {
                id: item.tool_use_id,
                name: definition.google.name,
                response: { content: item.content },
              },
            }],
          });
          break;
        }
        case "input_file":
        case "tool_result_file": {
          // TODO: make this strategy configurable
          const fileName = await this.ensureFileUploaded(item.content, item.kind, signal);
          googleHistory.push({
            role: "user",
            parts: [{
              fileData: {
                fileUri: new URL(fileName, getGoogleFileBaseUrl(this.#baseUrl)).toString(),
                mimeType: item.kind,
              },
            }],
          });
          break;
        }
        case "output_reasoning":
          // no-op, don't propagate reasoning
          break;
        default:
          item satisfies never;
      }
    }

    return googleHistory;
  }

  async *stream<zO, zI>({
    history,
    instructions,
    tools,
    signal,
    output,
  }: AdapterStreamOptions<zO, zI>): AdapterStreamIterator {
    const normalizedTools = normalizeGoogleTools(tools);
    const googleHistory = await this.getHistory(history, normalizedTools, signal);

    const stream = await this.#client.models.generateContentStream({
      model: this.model,
      contents: googleHistory,
      config: {
        systemInstruction: instructions,
        thinkingConfig: this.#thinkingConfig,
        tools: normalizedTools.length > 0
          ? [{ functionDeclarations: normalizedTools.map((tool) => tool.google) }]
          : undefined,
        responseMimeType: output && "application/json",
        responseJsonSchema: output && output.toJSONSchema(),
        abortSignal: signal,
      },
    });

    let lastIndex = -1;
    let lastType = "";
    const advanceIndex = (type: string) => {
      if (lastType !== type) {
        lastType = type;
        lastIndex++;
      }
      return lastIndex;
    };

    let usageMetadata: GenerateContentResponseUsageMetadata | undefined;
    for await (const item of stream) {
      usageMetadata = item.usageMetadata;

      const parts = item?.candidates?.[0]?.content?.parts;
      if (!parts?.length) continue;

      for (const part of parts) {
        if (part.text) {
          const isReasoning = !!part.thought;
          yield {
            type: isReasoning ? "delta_output_reasoning" : "delta_output_text",
            delta: part.text,
            index: advanceIndex(isReasoning ? "reasoning" : "text"),
          };
        } else if (part.functionCall) {
          const func = part.functionCall;
          const funcId = func.id ?? crypto.randomUUID();
          assert(func.name);
          const tool = normalizedTools.find((tool) => tool.google.name === func.name);

          if (part.thoughtSignature) {
            signatureMap.set(funcId, part.thoughtSignature);
          }

          const index = advanceIndex(funcId);

          // TODO: investigate if we can get this earlier
          yield {
            type: "tool_use_start",
            index,
            kind: tool?.original.normalizedName ?? func.name,
            tool_use_id: funcId,
          };
          yield {
            type: "tool_use",
            tool_use_id: funcId,
            kind: tool?.original.normalizedName ?? func.name,
            content: tool?.isVoid ? undefined : JSON.stringify(
              tool?.wrapperObject ? func.args?.content : func.args,
            ),
            index,
          };
        }
      }
    }

    return {
      inputTokens: usageMetadata?.promptTokenCount ?? null,
      outputTokens: usageMetadata?.totalTokenCount != null && usageMetadata?.promptTokenCount != null
        ? usageMetadata.totalTokenCount - usageMetadata.promptTokenCount
        : null,
    };
  }
}
