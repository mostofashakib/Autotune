"use client";

import Link from "next/link";
import { use, useState } from "react";
import { ErrorNotice, LogConsole, LossChart, PageHero, Panel, StatusBadge } from "@/components/ui";
import { apiPost, describeError } from "@/lib/api";
import { formatWhen, isTerminal } from "@/lib/format";
import type { TrainingRun } from "@/lib/types";
import { usePolling } from "@/lib/usePolling";

export default function TrainingRunPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const run = usePolling<TrainingRun>(`/api/training/status/${id}`, 2000, (data) => !isTerminal(data.status));
  const [cancelError, setCancelError] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const data = run.data;

  async function cancel() {
    setCancelling(true);
    setCancelError(null);
    try {
      await apiPost(`/api/training/cancel/${id}`);
      run.refresh();
    } catch (cause) {
      setCancelError(describeError(cause));
    } finally {
      setCancelling(false);
    }
  }

  return (
    <>
      <PageHero
        eyebrow={<><Link href="/training">01 / Training</Link> / {id}</>}
        title={data?.base_model ?? id}
        readout={[
          { label: "Status", value: data ? <StatusBadge status={data.status} /> : "…" },
          { label: "Progress", value: data ? `${data.progress_percent}%` : "…" },
          { label: "Step", value: data ? `${data.current_step}${data.total_steps ? ` / ${data.total_steps}` : ""}` : "…" },
          { label: "Loss", value: data?.current_loss?.toFixed(4) ?? "n/a" },
        ]}
      >
        {data ? `${data.objective?.toUpperCase() ?? ""} on ${data.dataset_name ?? "dataset"}, started ${formatWhen(data.created_at)}.` : null}
      </PageHero>

      <ErrorNotice message={run.error} title="Could not load this run" />
      <ErrorNotice message={data?.error ?? null} title={`Run ${data?.status ?? ""}`} />
      <ErrorNotice message={cancelError} title="Cancel failed" />
      {data?.checkpoint_dir ? (
        <div className="notice notice--ok">
          <strong>Checkpoint saved</strong>
          <code>{data.checkpoint_dir}</code>. <Link className="underline" href={`/serving?model=${encodeURIComponent(data.checkpoint_dir)}`}>Serve it</Link>
        </div>
      ) : null}

      <div className="grid-2">
        <Panel index="A" title="Loss" aside={data ? `${data.loss_history.length} logged steps` : undefined}>
          <div className="panel__body">
            {data ? (
              <>
                <div className="progress" aria-label="Progress"><div style={{ width: `${data.progress_percent}%` }} /></div>
                <LossChart values={data.loss_history} />
              </>
            ) : null}
            {data?.status === "running" ? (
              <button className="btn btn--ghost" disabled={cancelling} onClick={() => void cancel()}>
                {cancelling ? "Cancelling…" : "Cancel run"}
              </button>
            ) : null}
          </div>
        </Panel>
        <LogConsole title="Trainer log" lines={data?.logs ?? []} idle="Waiting for the trainer…" />
      </div>
    </>
  );
}
