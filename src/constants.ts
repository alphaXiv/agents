export const RETRY_RESUMABILITY_PROMPT =
  "<system-message>You were interrupted by a provider outage, streaming will now resume where you left off. Do not acknowledge this message and continue streaming as if nothing happened. Your message will be appended to the last assistant message. DO NOT REPEAT YOURSELF.</system-message>";

const STRUCTURED_OUTPUT_RETRY_FEEDBACK_PREFIX = "Sorry, my output has an error:\n";
const STRUCTURED_OUTPUT_RETRY_FEEDBACK_SUFFIX =
  "\nI will try again to produce a JSON response that conforms to the expected schema.";

export function createStructuredOutputRetryFeedback(errorMessage: string): string {
  return `${STRUCTURED_OUTPUT_RETRY_FEEDBACK_PREFIX}${errorMessage}${STRUCTURED_OUTPUT_RETRY_FEEDBACK_SUFFIX}`;
}

export function isStructuredOutputRetryFeedback(content: string): boolean {
  return content.startsWith(STRUCTURED_OUTPUT_RETRY_FEEDBACK_PREFIX) &&
    content.endsWith(STRUCTURED_OUTPUT_RETRY_FEEDBACK_SUFFIX);
}
