import assert from "node:assert/strict";
import { setImmediate } from "node:timers/promises";
import { test } from "node:test";
import { pollUntil, startPolling } from "../lib/polling.ts";

test("polls only after the preceding request finishes", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  let calls = 0;
  let finish;
  const pending = new Promise((resolve) => { finish = resolve; });
  const stop = startPolling(async () => {
    calls += 1;
    await pending;
  }, 5000, assert.fail);

  t.mock.timers.tick(0);
  assert.equal(calls, 1);
  t.mock.timers.tick(15000);
  assert.equal(calls, 1);
  finish();
  await setImmediate();
  t.mock.timers.tick(4999);
  assert.equal(calls, 1);
  t.mock.timers.tick(1);
  assert.equal(calls, 2);
  stop();
});

test("cleanup cancels startup and aborts requests without scheduling another poll", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  let calls = 0;
  const stopBeforeStart = startPolling(async () => { calls += 1; }, 10, assert.fail);
  stopBeforeStart();
  t.mock.timers.tick(20);
  assert.equal(calls, 0);

  let signal;
  let finish;
  const stop = startPolling(async (requestSignal) => {
    calls += 1;
    signal = requestSignal;
    await new Promise((resolve) => { finish = resolve; });
  }, 10, assert.fail);
  t.mock.timers.tick(0);
  stop();
  assert.equal(signal.aborted, true);
  finish();
  await setImmediate();
  t.mock.timers.tick(100);
  assert.equal(calls, 1);
});

test("reports request failures and continues polling", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const errors = [];
  let calls = 0;
  const stop = startPolling(async () => {
    calls += 1;
    throw new Error("offline");
  }, 10, (error) => errors.push(error.message));
  t.mock.timers.tick(0);
  await setImmediate();
  assert.deepEqual(errors, ["offline"]);
  t.mock.timers.tick(10);
  await setImmediate();
  assert.equal(calls, 2);
  stop();
});

test("pollUntil resolves with the first done value on the third poll", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  let calls = 0;
  const result = pollUntil(async () => (calls += 1), (value) => value === 3, 1000, new AbortController().signal);

  await setImmediate();
  assert.equal(calls, 1);
  t.mock.timers.tick(999);
  await setImmediate();
  assert.equal(calls, 1);
  t.mock.timers.tick(1);
  await setImmediate();
  assert.equal(calls, 2);
  t.mock.timers.tick(1000);
  assert.equal(await result, 3);
  assert.equal(calls, 3);
});

test("pollUntil rejects with AbortError when aborted between polls and schedules nothing more", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const controller = new AbortController();
  let calls = 0;
  const result = pollUntil(async () => (calls += 1), () => false, 1000, controller.signal);
  await setImmediate();
  assert.equal(calls, 1);

  controller.abort();
  await assert.rejects(result, { name: "AbortError" });
  t.mock.timers.tick(10000);
  await setImmediate();
  assert.equal(calls, 1);
});

test("pollUntil rejects with AbortError when aborted during a request", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const controller = new AbortController();
  let calls = 0;
  const result = pollUntil(async () => {
    calls += 1;
    controller.abort();
    return calls;
  }, () => false, 1000, controller.signal);
  await assert.rejects(result, { name: "AbortError" });
  t.mock.timers.tick(10000);
  await setImmediate();
  assert.equal(calls, 1);
});

test("pollUntil rejects when a request fails", async () => {
  await assert.rejects(
    pollUntil(async () => { throw new Error("offline"); }, () => true, 1000, new AbortController().signal),
    /offline/,
  );
});
