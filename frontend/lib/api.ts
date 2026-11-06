const isDev = process.env.NODE_ENV !== "production";

/** Base URL of the Autotune API (`autotune api`). Set NEXT_PUBLIC_API_URL to override. */
export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? (isDev ? "http://localhost:7860" : "");

/** An HTTP error from the API, carrying the server's own explanation. */
export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

type ValidationIssue = { loc?: (string | number)[]; msg?: string };

/** Turn a failed response body into an ApiError whose message is FastAPI's `detail`. */
export function apiErrorFrom(status: number, body: string): ApiError {
  let detail: unknown = body.trim();
  try {
    detail = (JSON.parse(body) as { detail?: unknown }).detail ?? detail;
  } catch {
    // Not JSON: show the body as the server sent it.
  }
  if (Array.isArray(detail)) {
    detail = (detail as ValidationIssue[])
      .map((issue) => `${(issue.loc ?? []).filter((part) => part !== "body").join(".")}: ${issue.msg ?? "invalid"}`)
      .join("; ");
  }
  const message = typeof detail === "string" && detail ? detail : `Request failed with HTTP ${status}`;
  return new ApiError(status, message);
}

/** What to tell the person when a request failed. */
export function describeError(error: unknown, base: string = API_BASE): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof TypeError) {
    return `Cannot reach the Autotune API at ${base}. Start it with \`autotune api\` (or ./run.sh).`;
  }
  return error instanceof Error ? error.message : String(error);
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { cache: "no-store", ...init });
  const body = await response.text();
  if (!response.ok) throw apiErrorFrom(response.status, body);
  return (body ? JSON.parse(body) : null) as T;
}

export function apiGet<T>(path: string, signal?: AbortSignal): Promise<T> {
  return request<T>(path, { signal });
}

export function apiPost<T>(path: string, payload?: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: payload === undefined ? undefined : { "Content-Type": "application/json" },
    body: payload === undefined ? undefined : JSON.stringify(payload),
  });
}

export function apiUpload<T>(path: string, file: File): Promise<T> {
  const form = new FormData();
  form.append("file", file);
  return request<T>(path, { method: "POST", body: form });
}
