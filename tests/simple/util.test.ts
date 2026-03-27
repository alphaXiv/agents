import { assertEquals, assertRejects } from "@std/assert";
import { delay } from "@std/async/delay";
import { iteratePromiseArray } from "../../src/util.ts";

Deno.test("iteratePromiseArray - empty array yields nothing", async () => {
  const results = await Array.fromAsync(iteratePromiseArray([]));
  assertEquals(results, []);
});

Deno.test("iteratePromiseArray - single promise yields its value", async () => {
  const results = await Array.fromAsync(
    iteratePromiseArray([Promise.resolve(42)]),
  );
  assertEquals(results, [42]);
});

Deno.test("iteratePromiseArray - yields all values from multiple promises", async () => {
  const first = Promise.withResolvers<string>();
  const second = Promise.withResolvers<string>();
  const third = Promise.withResolvers<string>();
  const iterPromise = Array.fromAsync(
    iteratePromiseArray([first.promise, second.promise, third.promise]),
  );
  first.resolve("first");
  second.resolve("second");
  third.resolve("third");
  const results = await iterPromise;
  assertEquals(results.sort(), ["first", "second", "third"]);
});

Deno.test("iteratePromiseArray - all promises already resolved", async () => {
  const promises = [
    Promise.resolve(1),
    Promise.resolve(2),
    Promise.resolve(3),
  ];

  const results = await Array.fromAsync(iteratePromiseArray(promises));
  assertEquals(results.sort((a, b) => a - b), [1, 2, 3]);
});

Deno.test("iteratePromiseArray - slower promise is still yielded", async () => {
  const slow = Promise.withResolvers<string>();
  const fast = Promise.withResolvers<string>();

  const iterPromise = Array.fromAsync(
    iteratePromiseArray([slow.promise, fast.promise]),
  );

  fast.resolve("fast");
  await delay(10);
  slow.resolve("slow");

  const results = await iterPromise;
  assertEquals(results.sort(), ["fast", "slow"]);
});

Deno.test("iteratePromiseArray - single rejection", async () => {
  const good = Promise.withResolvers<number>();
  const bad = Promise.withResolvers<number>();

  const err = new Error("fail");
  const iterPromise = Array.fromAsync(
    iteratePromiseArray([good.promise, bad.promise]),
  );

  bad.reject(err);
  good.resolve(1);

  await assertRejects(() => iterPromise, Error, "fail");
});

Deno.test("iteratePromiseArray - two rejections throws AggregateError", async () => {
  const a = Promise.withResolvers<number>();
  const b = Promise.withResolvers<number>();
  const c = Promise.withResolvers<number>();

  const err1 = new Error("one");
  const err2 = new Error("two");

  // Sequence: resolve c, then reject a and b so both errors are Array.fromAsynced
  const iterPromise = Array.fromAsync(
    iteratePromiseArray([a.promise, b.promise, c.promise]),
  );

  c.resolve(99);
  await delay(10);

  a.reject(err1);
  b.reject(err2);

  await assertRejects(
    () => iterPromise,
    AggregateError,
  );
});
