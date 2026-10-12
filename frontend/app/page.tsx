"use client";

import Link from "next/link";
import { ErrorNotice, PageHero, Panel, StatusBadge } from "@/components/ui";
import { formatDelta, formatWhen } from "@/lib/format";
import type { BenchmarkSummary, CredentialStatus, ServingState, SystemStatus, TrainingSummary } from "@/lib/types";
import { usePolling } from "@/lib/usePolling";

const always = () => true;

export default function HomePage() {
  const system = usePolling<SystemStatus>("/api/system/status", 30000, always);
  const credentials = usePolling<CredentialStatus[]>("/api/settings/credentials", 30000, always);
  const training = usePolling<TrainingSummary[]>("/api/training", 5000, always);
  const benchmarks = usePolling<BenchmarkSummary[]>("/api/benchmark", 5000, always);
  const serving = usePolling<ServingState>("/api/serve/status", 5000, always);
  const error = system.error ?? training.error ?? benchmarks.error ?? serving.error ?? credentials.error;
  const missing = (credentials.data ?? []).filter((credential) => !credential.configured);

  return (
    <>
      <PageHero
        eyebrow="Autotune / control room"
        title={<>Train, measure, <em>serve</em>.</>}
        readout={[
          { label: "Device", value: system.data?.device ?? "…" },
          { label: "VRAM", value: system.data ? `${system.data.vram_gb} GB` : "…" },
          { label: "Quantization", value: system.data?.recommended_quantization ?? "…" },
          { label: "Serving", value: serving.data ? <StatusBadge status={serving.data.status} /> : "…" },
        ]}
      >
        Upload a dataset, fine-tune a model on it, benchmark the result against its base, and serve it.
      </PageHero>

      <ErrorNotice message={error} title="API unavailable" />
      {missing.length > 0 ? (
        <div className="notice notice--warn">
          <strong>Credentials not configured</strong>
          {missing.map((credential) => credential.name).join(", ")}. See <Link className="underline" href="/settings">Settings</Link>.
        </div>
      ) : null}

      <div className="grid-2">
        <Panel index="01" title="Recent training" aside={<Link href="/training">All runs →</Link>}>
          {training.data?.length ? (
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th>Run</th><th>Model</th><th>Status</th><th>Started</th></tr></thead>
                <tbody>
                  {training.data.slice(0, 5).map((run) => (
                    <tr key={run.job_id}>
                      <td><Link href={`/training/${run.job_id}`} className="mono">{run.job_id}</Link></td>
                      <td className="mono">{run.base_model ?? "—"}</td>
                      <td><StatusBadge status={run.status} /></td>
                      <td>{formatWhen(run.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="panel__empty">No training runs yet. <Link className="underline" href="/training">Start one</Link>.</p>
          )}
        </Panel>

        <Panel index="02" title="Recent benchmarks" aside={<Link href="/benchmark">All runs →</Link>}>
          {benchmarks.data?.length ? (
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th>Run</th><th>Models</th><th>Δ pass@1</th><th>Status</th></tr></thead>
                <tbody>
                  {benchmarks.data.slice(0, 5).map((run) => (
                    <tr key={run.job_id}>
                      <td><Link href={`/benchmark/${run.job_id}`} className="mono">{run.job_id}</Link></td>
                      <td className="mono">{run.base_model} → {run.trained_model}</td>
                      <td className="mono">{run.headline ? formatDelta(run.headline.delta_pass_at_1) : "—"}</td>
                      <td><StatusBadge status={run.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="panel__empty">No benchmarks yet. <Link className="underline" href="/benchmark">Run one</Link>.</p>
          )}
        </Panel>
      </div>
    </>
  );
}
