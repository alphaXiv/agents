import { TestingAdapter } from "./adapters/__testing.ts";

export const ADAPTERS: Record<string, typeof TestingAdapter> = {
  "__testing": TestingAdapter,
};
