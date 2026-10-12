import assert from "node:assert/strict";
import { test } from "node:test";
import { ApiError, apiErrorFrom, describeError } from "../lib/api.ts";

test("a FastAPI detail string becomes the error message", () => {
  const error = apiErrorFrom(503, JSON.stringify({ detail: "RUNPOD_API_KEY is not configured." }));
  assert.ok(error instanceof ApiError);
  assert.equal(error.status, 503);
  assert.equal(error.message, "RUNPOD_API_KEY is not configured.");
});

test("validation errors list each field and its problem", () => {
  const body = { detail: [{ loc: ["body", "k"], msg: "Input should be greater than 0" }] };
  assert.equal(apiErrorFrom(422, JSON.stringify(body)).message, "k: Input should be greater than 0");
});

test("a body that is not JSON is shown as returned", () => {
  assert.equal(apiErrorFrom(502, "Bad Gateway").message, "Bad Gateway");
});

test("an empty body falls back to the HTTP status", () => {
  assert.equal(apiErrorFrom(500, "").message, "Request failed with HTTP 500");
});

test("a network failure names the API it could not reach", () => {
  const message = describeError(new TypeError("fetch failed"), "http://localhost:7860");
  assert.match(message, /Cannot reach the Autotune API at http:\/\/localhost:7860/);
  assert.match(message, /autotune api/);
});

test("an API error is described by its message", () => {
  assert.equal(describeError(new ApiError(409, "Job is completed"), "x"), "Job is completed");
});
