import { encodeBase64 } from "@std/encoding";

export const IMAGE_MIME_TYPES = [
  "image/jpeg",
  "image/jpg",
  "image/png",
  "image/gif",
  "image/webp",
] as const;

export const PDF_MIME_TYPE = "application/pdf";

export const TEXTLIKE_MIME_TYPES = [
  "text/*",
  "application/json",
  "application/*+json",
  "application/xml",
  "application/*+xml",
  "application/yaml",
  "application/*+yaml",
  "application/x-yaml",
  "application/toml",
  "application/javascript",
  "application/ecmascript",
  "application/typescript",
  "application/x-typescript",
] as const;

export const DEFAULT_SUPPORTED_MIME_TYPES = [
  ...IMAGE_MIME_TYPES,
  PDF_MIME_TYPE,
  ...TEXTLIKE_MIME_TYPES,
];

function mimeTypeMatches(pattern: string, mimeType: string): boolean {
  if (pattern === mimeType) return true;

  const escapedPattern = pattern.replace(/[|\\{}()[\]^$+?.]/g, "\\$&").replaceAll("*", ".*");
  return new RegExp(`^${escapedPattern}$`).test(mimeType);
}

export function supportsMimeType(mimeType: string, supportedMimeTypes: string[]) {
  return supportedMimeTypes.some((pattern) => mimeTypeMatches(pattern, mimeType));
}

export function isTextLikeMimeType(mimeType: string) {
  return TEXTLIKE_MIME_TYPES.some((pattern) => mimeTypeMatches(pattern, mimeType));
}

export function unsupportedMediaTypeError(model: string, mimeType: string) {
  return new Error(`Model ${JSON.stringify(model)} does not support media type ${JSON.stringify(mimeType)}`);
}

export function getFileNameFromUrl(url: string): string | undefined {
  try {
    const pathname = new URL(url).pathname;
    const fileName = pathname.split("/").filter(Boolean).pop();
    return fileName ? decodeURIComponent(fileName) : undefined;
  } catch {
    return url.split("/").filter(Boolean).pop();
  }
}

export async function getContentLength(url: string, signal: AbortSignal) {
  const headResponse = await fetch(url, {
    method: "HEAD",
    signal,
  });
  const contentLength = headResponse.headers.get("Content-Length");
  if (contentLength) {
    return parseInt(contentLength, 10);
  }

  const response = await fetch(url, { signal });
  return (await response.arrayBuffer()).byteLength;
}

export async function fetchTextLikeFileAsTaggedText(url: string, mimeType: string, signal: AbortSignal) {
  const response = await fetch(url, { signal });
  const text = await response.text();
  return `<file mime-type="${mimeType}">${text}</file>`;
}

export async function fetchPdfAsText(url: string, signal: AbortSignal) {
  const response = await fetch(url, { signal });
  const { default: parsePdf } = await import("@lino/pdf-parse");
  const pdfText = await parsePdf(await response.arrayBuffer());
  return pdfText.text.join("\n");
}

export async function fetchRemoteFileAsDataUrl(url: string, mimeType: string, signal: AbortSignal) {
  const response = await fetch(url, { signal });
  const buffer = await response.arrayBuffer();
  return `data:${mimeType};base64,${encodeBase64(buffer)}`;
}
