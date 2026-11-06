export type Tone = "live" | "ok" | "bad" | "warn" | "idle";

const TONES: Record<string, Tone> = {
  running: "live",
  starting: "live",
  online: "ok",
  completed: "ok",
  failed: "bad",
  cancelled: "warn",
  interrupted: "warn",
  stopped: "idle",
};

/** A run that is no longer running will not change again, so polling can stop. */
export function isTerminal(status: string): boolean {
  return status !== "running";
}

export function statusTone(status: string): Tone {
  return TONES[status] ?? "idle";
}

/** Risk chips reuse the status colors. A risk that was not measured is neutral. */
export function riskTone(risk: string | null | undefined): Tone {
  if (risk === "high") return "bad";
  if (risk === "moderate") return "warn";
  if (risk === "low") return "ok";
  return "idle";
}

export function formatScore(value: number | null | undefined): string {
  return value === null || value === undefined ? "n/a" : `${(value * 100).toFixed(1)}%`;
}

export function formatDelta(value: number): string {
  const points = (value * 100).toFixed(1);
  return value > 0 ? `+${points} pts` : `${points} pts`;
}

export function formatWhen(iso: string | undefined): string {
  if (!iso) return "unknown";
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? "unknown" : date.toLocaleString();
}
