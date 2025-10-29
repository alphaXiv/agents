import { AsyncLocalStorage } from "node:async_hooks";

export const signalAsyncLocalStorage = new AsyncLocalStorage<AbortSignal>();
