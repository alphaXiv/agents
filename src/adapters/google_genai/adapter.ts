import {
  type DeleteFileResponse,
  type GenerateContentResponseUsageMetadata,
  GoogleGenAI,
  type GoogleGenAIOptions,
  type ThinkingConfig,
} from "@google/genai";
import { assert } from "@std/assert";
import type { AdapterStreamIterator } from "../../types.ts";
import { hashString } from "../../util.ts";
import type { Adapter, AdapterStreamOptions } from "../adapter.ts";
import { splitCacheInclusiveUsage } from "../shared/usage.ts";
import { getGoogleGenerateContentAPIHistory, rememberGoogleThoughtSignature } from "./history.ts";
import { normalizeGoogleTools } from "./tools.ts";

// Re-exported so wrappers building their own googleGenerateContentAPIModel
// (e.g. a Vertex fallback with inline credentials) can map thinking levels
// the same way geminiModel and vertexAIModel do.
export { getThinkingConfig, type GoogleModels, type SupportedThinkingLevel } from "./models.ts";

export interface GoogleGenerateContentAPIModelOptions {
  googleGenAIOptions?: GoogleGenAIOptions;
  thinkingConfig?: ThinkingConfig;
  model: string;
  provider?: string;
}

/** Generic adapter over a google GenerateContentStream compatible API */
export function googleGenerateContentAPIModel<zO, zI>(
  options: GoogleGenerateContentAPIModelOptions,
): Adapter<zO, zI> {
  const client = new GoogleGenAI(options?.googleGenAIOptions ?? {});
  const baseUrl = options.googleGenAIOptions?.httpOptions?.baseUrl ?? "https://generativelanguage.googleapis.com";
  const thinkingConfig = options.thinkingConfig;

  async function ensureFileUploaded(url: string, mimeType: string, abortSignal: AbortSignal): Promise<string> {
    // Create a safe filename by hashing the URL
    const safeFileName = (await hashString(url)).slice(0, 40);

    // Try to get the file first to see if it exists
    try {
      await client.files.get({
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

      await client.files.upload({
        file: blob,
        config: {
          name: `files/${safeFileName}`,
          mimeType,
          abortSignal,
        },
      });
    } catch (error) {
      if (
        !(error instanceof Error) ||
        !(
          error.message.includes("generativelanguage.googleapis.com/file_storage_bytes") &&
          error.message.includes("429")
        )
      ) {
        throw error;
      }

      // We've run out of file storage as cache. Delete some files, then try uploading again.
      const fileList = await client.files.list({
        config: { pageSize: 100, abortSignal },
      });

      let toDelete = 1000; // How many files we want to delete before attempting to upload one
      const deletionPromises: Promise<DeleteFileResponse>[] = [];
      for await (const file of fileList) {
        // Check if the file was created > 30 minutes ago
        if (file.name && (!file.createTime || new Date(file.createTime).getTime() < Date.now() - (30 * 60 * 1000))) {
          // If so, delete it
          deletionPromises.push(
            client.files.delete({
              name: file.name,
              config: { abortSignal },
            }),
          );
          if (--toDelete <= 0) break;
        }
      }
      await Promise.all(deletionPromises);
      return await ensureFileUploaded(url, mimeType, abortSignal);
    }

    return safeFileName;
  }

  return {
    provider: options.provider ?? "GoogleGenerateContentAPI",
    model: options.model,
    stream: async function* stream<zO, zI>({
      history,
      instructions,
      tools,
      signal,
      output,
    }: AdapterStreamOptions<zO, zI>): AdapterStreamIterator {
      const normalizedTools = normalizeGoogleTools(tools);
      const googleHistory = await getGoogleGenerateContentAPIHistory({
        history,
        toolMap: normalizedTools,
        signal,
        baseUrl,
        ensureFileUploaded,
        // Vertex AI does not support the Files API, so file history is sent as inline data.
        inlineFiles: options.googleGenAIOptions?.vertexai === true,
      });

      const stream = await client.models.generateContentStream({
        model: options.model,
        contents: googleHistory,
        config: {
          systemInstruction: instructions,
          thinkingConfig,
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
            assert(func.name, "Function calls must have a name");
            const tool = normalizedTools.find((tool) => tool.google.name === func.name);

            if (part.thoughtSignature) {
              rememberGoogleThoughtSignature(funcId, part.thoughtSignature);
            }

            const index = advanceIndex(funcId);

            // TODO: investigate if we can get this earlier
            yield {
              type: "tool_use_start",
              index,
              kind: tool?.original.name ?? func.name,
              tool_use_id: funcId,
            };
            yield {
              type: "tool_use",
              tool_use_id: funcId,
              kind: tool?.original.name ?? func.name,
              content: tool?.isVoid ? undefined : JSON.stringify(
                tool?.wrapperObject ? func.args?.content : func.args,
              ),
              index,
            };
          }
        }
      }

      return {
        ...splitCacheInclusiveUsage(usageMetadata?.promptTokenCount, usageMetadata?.cachedContentTokenCount),
        // promptTokenCount is cache-inclusive, so this stays the correct output count.
        outputTokens: usageMetadata?.totalTokenCount != null && usageMetadata?.promptTokenCount != null
          ? usageMetadata.totalTokenCount - usageMetadata.promptTokenCount
          : null,
      };
    },
  };
}
