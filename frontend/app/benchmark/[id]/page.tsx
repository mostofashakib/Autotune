"use client";

import Link from "next/link";
import { use } from "react";
import { ErrorNotice, LogConsole, PageHero, Panel, StatusBadge } from "@/components/ui";
import { formatDelta, formatScore, formatWhen, isTerminal } from "@/lib/format";
import type { BenchmarkRun, ModelScore } from "@/lib/types";
import { usePolling } from "@/lib/usePolling";

function deltaClass(value: number): string {
  return value > 0 ? "num-pos" : value < 0 ? "num-neg" : "";
}

export default function BenchmarkRunPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const run = usePolling<BenchmarkRun>(`/api/benchmark/${id}`, 2000, (data) => !isTerminal(data.status));
  const data = run.data;
  const report = data?.report ?? null;

  return (
    <>
      <PageHero
        eyebrow={<><Link href="/benchmark">02 / Benchmark</Link> / {id}</>}
        title={data ? <>{data.base_model} → <em>{data.trained_model}</em></> : id}
        readout={[
          { label: "Status", value: data ? <StatusBadge status={data.status} /> : "…" },
          { label: "Mode", value: data ? `${data.mode}${data.k ? ` · k=${data.k}` : ""}` : "…" },
          { label: "Δ pass@1", value: report ? <span className={deltaClass(report.delta_pass_at_1)}>{formatDelta(report.delta_pass_at_1)}</span> : "—" },
          { label: "Δ pass@k", value: report ? <span className={deltaClass(report.delta_pass_at_k)}>{formatDelta(report.delta_pass_at_k)}</span> : "—" },
        ]}
      >
        {data ? `${data.path ?? ""} · started ${formatWhen(data.created_at)}` : null}
      </PageHero>

      <ErrorNotice message={run.error} title="Could not load this benchmark" />
      <ErrorNotice message={data?.error ?? null} title={`Benchmark ${data?.status ?? ""}`} />

      {report ? (
        <>
          <Panel index="A" title="Scores">
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th>Model</th><th>Samples</th><th>Passed</th><th>Errors</th><th>pass@1</th><th>pass@{report.k}</th><th>Mean score</th><th>Judge</th></tr></thead>
                <tbody>
                  <ScoreRow label={`Base · ${report.base_endpoint.model}`} score={report.base} />
                  <ScoreRow label={`Trained · ${report.trained_endpoint.model}`} score={report.trained} />
                </tbody>
              </table>
            </div>
          </Panel>
          <Panel index="B" title="Per task" aside={`${report.tasks.length} tasks`}>
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th>Task</th><th>Base pass@1</th><th>Trained pass@1</th><th>Δ pass@1</th><th>Δ mean score</th></tr></thead>
                <tbody>
                  {report.tasks.map((task) => (
                    <tr key={task.task_id}>
                      <td className="mono">{task.task_id}</td>
                      <td className="mono">{formatScore(task.base.pass_at_1)}</td>
                      <td className="mono">{formatScore(task.trained.pass_at_1)}</td>
                      <td className={`mono ${deltaClass(task.delta_pass_at_1)}`}>{formatDelta(task.delta_pass_at_1)}</td>
                      <td className={`mono ${deltaClass(task.delta_mean_score)}`}>{formatDelta(task.delta_mean_score)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        </>
      ) : null}

      <LogConsole title="Benchmark log" lines={data?.logs ?? []} idle="Waiting for the benchmark…" />
    </>
  );
}

function ScoreRow({ label, score }: { label: string; score: ModelScore }) {
  return (
    <tr>
      <td className="mono">{label}</td>
      <td className="mono">{score.samples}</td>
      <td className="mono">{score.passed}</td>
      <td className="mono">{score.errors}</td>
      <td className="mono">{formatScore(score.pass_at_1)}</td>
      <td className="mono">{formatScore(score.pass_at_k)}</td>
      <td className="mono">{formatScore(score.mean_score)}</td>
      <td className="mono">{formatScore(score.mean_judge_score)}</td>
    </tr>
  );
}
