"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { ErrorNotice, PageHero, Panel, StatusBadge } from "@/components/ui";
import { apiPost, describeError } from "@/lib/api";
import { formatDelta, formatScore, formatWhen } from "@/lib/format";
import type { BenchmarkSummary, ModelEndpoint } from "@/lib/types";
import { usePolling } from "@/lib/usePolling";

type Mode = "dataset" | "harbor";

const EMPTY_ENDPOINT: ModelEndpoint = { base_url: "", model: "", api_key_env: "" };

function cleanEndpoint(endpoint: ModelEndpoint): ModelEndpoint {
  return { base_url: endpoint.base_url.trim(), model: endpoint.model.trim(), api_key_env: endpoint.api_key_env?.trim() || null };
}

export default function BenchmarkPage() {
  const router = useRouter();
  const runs = usePolling<BenchmarkSummary[]>("/api/benchmark", 5000, () => true);
  const [mode, setMode] = useState<Mode>("dataset");
  const [path, setPath] = useState("");
  const [k, setK] = useState(3);
  const [maxTokens, setMaxTokens] = useState(4096);
  const [base, setBase] = useState<ModelEndpoint>({ ...EMPTY_ENDPOINT });
  const [trained, setTrained] = useState<ModelEndpoint>({ ...EMPTY_ENDPOINT });
  const [useJudge, setUseJudge] = useState(false);
  const [judge, setJudge] = useState<ModelEndpoint>({ ...EMPTY_ENDPOINT });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function launch() {
    setBusy(true);
    setError(null);
    try {
      const { job_id } = await apiPost<{ job_id: string }>("/api/benchmark", {
        mode,
        path: path.trim(),
        k,
        max_tokens: maxTokens,
        base: cleanEndpoint(base),
        trained: cleanEndpoint(trained),
        judge: mode === "dataset" && useJudge ? cleanEndpoint(judge) : null,
      });
      router.push(`/benchmark/${job_id}`);
    } catch (cause) {
      setError(describeError(cause));
      setBusy(false);
    }
  }

  return (
    <>
      <PageHero eyebrow="02 / Benchmark" title={<>Did training <em>help</em>?</>}>
        Measure a base and a trained model on the same held-out data or Harbor tasks. Every number in the report comes from recorded samples.
      </PageHero>
      <ErrorNotice message={error} />

      <Panel index="A" title="New benchmark">
        <div className="panel__body">
          <div className="segmented" role="radiogroup" aria-label="Mode">
            {(["dataset", "harbor"] as Mode[]).map((option) => (
              <label key={option}>
                <input type="radio" name="mode" value={option} checked={mode === option} onChange={() => setMode(option)} />
                {option === "dataset" ? "Held-out dataset" : "Harbor tasks"}
              </label>
            ))}
          </div>
          <label className="field">
            <span className="field__label">
              {mode === "dataset" ? "Held-out dataset path" : "Harbor task directory"}
              <small>on the API server</small>
            </span>
            <input
              className="input input--mono"
              placeholder={mode === "dataset" ? "exports/held_out.jsonl" : "tasks/hello-world"}
              value={path}
              onChange={(e) => setPath(e.target.value)}
            />
          </label>
          <div className="field-row">
            <EndpointFields title="Base model" value={base} onChange={setBase} />
            <EndpointFields title="Trained model" value={trained} onChange={setTrained} />
          </div>
          <div className="field-row">
            <label className="field">
              <span className="field__label">Attempts per task (k)</span>
              <input className="input input--mono" type="number" min={1} max={50} value={k} onChange={(e) => setK(Number(e.target.value))} />
            </label>
            {mode === "dataset" ? (
              <label className="field">
                <span className="field__label">Answer token budget</span>
                <input className="input input--mono" type="number" min={1} value={maxTokens} onChange={(e) => setMaxTokens(Number(e.target.value))} />
              </label>
            ) : null}
          </div>
          {mode === "dataset" ? (
            <>
              <label className="check">
                <input type="checkbox" checked={useJudge} onChange={(e) => setUseJudge(e.target.checked)} />
                Also score answers with an LLM judge
              </label>
              {useJudge ? <EndpointFields title="Judge model" value={judge} onChange={setJudge} /> : null}
            </>
          ) : null}
          <button className="btn btn--block" disabled={busy} onClick={() => void launch()}>
            {busy ? "Starting…" : "Run benchmark"}
          </button>
        </div>
      </Panel>

      <Panel index="B" title="Runs" aside={runs.data ? `${runs.data.length} total` : undefined}>
        <ErrorNotice message={runs.error} title="Could not load runs" />
        {runs.data?.length ? (
          <div className="table-wrap">
            <table className="table">
              <thead><tr><th>Run</th><th>Mode</th><th>Base → trained</th><th>pass@1</th><th>Δ</th><th>Status</th><th>Started</th></tr></thead>
              <tbody>
                {runs.data.map((run) => (
                  <tr key={run.job_id}>
                    <td><Link href={`/benchmark/${run.job_id}`} className="mono">{run.job_id}</Link></td>
                    <td className="mono">{run.mode}{run.k ? ` · k=${run.k}` : ""}</td>
                    <td className="mono">{run.base_model} → {run.trained_model}</td>
                    <td className="mono">{run.headline ? `${formatScore(run.headline.base_pass_at_1)} → ${formatScore(run.headline.trained_pass_at_1)}` : "—"}</td>
                    <td className={`mono ${run.headline && run.headline.delta_pass_at_1 > 0 ? "num-pos" : run.headline && run.headline.delta_pass_at_1 < 0 ? "num-neg" : ""}`}>
                      {run.headline ? formatDelta(run.headline.delta_pass_at_1) : "—"}
                    </td>
                    <td><StatusBadge status={run.status} /></td>
                    <td>{formatWhen(run.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="panel__empty">No benchmarks yet.</p>
        )}
      </Panel>
    </>
  );
}

function EndpointFields({ title, value, onChange }: { title: string; value: ModelEndpoint; onChange: (value: ModelEndpoint) => void }) {
  return (
    <fieldset className="space-y-2 border border-foreground/20 p-3">
      <legend className="px-1 text-xs font-semibold uppercase tracking-wider">{title}</legend>
      <input className="input input--mono" placeholder="http://localhost:11434/v1" aria-label={`${title} base URL`} value={value.base_url} onChange={(e) => onChange({ ...value, base_url: e.target.value })} />
      <input className="input input--mono" placeholder="model name" aria-label={`${title} model`} value={value.model} onChange={(e) => onChange({ ...value, model: e.target.value })} />
      <input className="input input--mono" placeholder="API key env var (optional)" aria-label={`${title} API key env var`} value={value.api_key_env ?? ""} onChange={(e) => onChange({ ...value, api_key_env: e.target.value })} />
    </fieldset>
  );
}
