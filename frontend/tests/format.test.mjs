import assert from "node:assert/strict";
import { test } from "node:test";
import { formatDelta, formatScore, formatWhen, isTerminal, kappaLabel, riskTone, statusTone } from "../lib/format.ts";

test("running is the only non-terminal status", () => {
  assert.equal(isTerminal("running"), false);
  for (const status of ["completed", "failed", "cancelled", "interrupted"]) assert.equal(isTerminal(status), true);
});

test("each status maps to a tone", () => {
  assert.equal(statusTone("running"), "live");
  assert.equal(statusTone("completed"), "ok");
  assert.equal(statusTone("failed"), "bad");
  assert.equal(statusTone("interrupted"), "warn");
  assert.equal(statusTone("cancelled"), "warn");
  assert.equal(statusTone("something-new"), "idle");
});

test("risk tones go from ok to bad and an unmeasured risk is neutral", () => {
  assert.equal(riskTone("low"), "ok");
  assert.equal(riskTone("moderate"), "warn");
  assert.equal(riskTone("high"), "bad");
  assert.equal(riskTone(null), "idle");
});

test("scores show as percentages with one decimal", () => {
  assert.equal(formatScore(0.5), "50.0%");
  assert.equal(formatScore(1), "100.0%");
  assert.equal(formatScore(null), "n/a");
});

test("deltas carry their sign", () => {
  assert.equal(formatDelta(0.25), "+25.0 pts");
  assert.equal(formatDelta(-0.1), "-10.0 pts");
  assert.equal(formatDelta(0), "0.0 pts");
});

test("timestamps format in the viewer's locale and a missing one says so", () => {
  assert.equal(formatWhen(undefined), "unknown");
  assert.notEqual(formatWhen("2026-10-12T10:00:00+00:00"), "unknown");
});

test("kappa says why it is missing: no judge, or a rater that never varied", () => {
  assert.equal(kappaLabel(0.4123, true), "0.41");
  assert.equal(kappaLabel(null, false), "No judge configured");
  assert.equal(kappaLabel(null, true), "Not measured: a rater never varied");
  assert.equal(kappaLabel(undefined, true), "Not measured: a rater never varied");
});
