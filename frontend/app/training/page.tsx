"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { ErrorNotice, PageHero, Panel, StatusBadge } from "@/components/ui";
import { apiPost, apiUpload, describeError } from "@/lib/api";
import { formatWhen } from "@/lib/format";
import type { Recipe, TrainingSummary, UploadedDataset } from "@/lib/types";
import { usePolling } from "@/lib/usePolling";

type Overrides = Recipe & { output_dir: string };

function overridesFrom(dataset: UploadedDataset): Overrides {
  const name = dataset.file_name.replace(/\.[^.]+$/, "");
  return { ...dataset.recipe, output_dir: `./checkpoints/${name}` };
}

export default function TrainingPage() {
  const router = useRouter();
  const runs = usePolling<TrainingSummary[]>("/api/training", 5000, () => true);
  const [dataset, setDataset] = useState<UploadedDataset | null>(null);
  const [form, setForm] = useState<Overrides | null>(null);
  const [busy, setBusy] = useState<"upload" | "start" | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function upload(file: File) {
    setBusy("upload");
    setError(null);
    try {
      const uploaded = await apiUpload<UploadedDataset>("/api/dataset/upload", file);
      setDataset(uploaded);
      setForm(overridesFrom(uploaded));
    } catch (cause) {
      setDataset(null);
      setError(describeError(cause));
    } finally {
      setBusy(null);
    }
  }

  async function start() {
    if (!dataset || !form) return;
    setBusy("start");
    setError(null);
    try {
      const { job_id } = await apiPost<{ job_id: string }>("/api/training/start", {
        dataset_name: dataset.file_name,
        ...form,
      });
      router.push(`/training/${job_id}`);
    } catch (cause) {
      setError(describeError(cause));
      setBusy(null);
    }
  }

  function set<K extends keyof Overrides>(key: K, value: Overrides[K]) {
    setForm((current) => (current ? { ...current, [key]: value } : current));
  }

  return (
    <>
      <PageHero eyebrow="01 / Training" title={<>Fine-tune on <em>your data</em>.</>}>
        Upload a dataset. Autotune detects its format and picks a recipe for your hardware. Adjust anything, then start the run.
      </PageHero>
      <ErrorNotice message={error} />

      <div className="grid-2">
        <Panel index="A" title="Dataset" aside="JSONL, JSON, CSV, Parquet">
          <div className="panel__body">
            <label className="field">
              <span className="field__label">File <small>saved to the data directory</small></span>
              <input
                className="input"
                type="file"
                accept=".jsonl,.json,.csv,.parquet"
                disabled={busy !== null}
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) void upload(file);
                }}
              />
            </label>
            {busy === "upload" ? <p className="text-sm text-muted-foreground">Inspecting…</p> : null}
            {dataset ? (
              <>
                <div className="facts">
                  <div><span>Format</span><strong>{dataset.task_type}</strong></div>
                  <div><span>Records</span><strong>{dataset.total_records}</strong></div>
                  <div><span>Train / val</span><strong>{dataset.train_records} / {dataset.val_records}</strong></div>
                  <div><span>Dropped</span><strong>{dataset.dropped_count}</strong></div>
                </div>
                {dataset.preview.map((row, index) => (
                  <div key={index} className="answer">
                    <h3>Prompt</h3>
                    <p>{row.prompt}</p>
                    <h3 className="mt-3">Completion</h3>
                    <p>{row.completion}</p>
                  </div>
                ))}
              </>
            ) : null}
          </div>
        </Panel>

        <Panel index="B" title="Recipe" aside={dataset ? "auto-selected, editable" : "upload a dataset first"}>
          {form ? (
            <div className="panel__body">
              <label className="field">
                <span className="field__label">Base model</span>
                <input className="input input--mono" value={form.base_model} onChange={(e) => set("base_model", e.target.value)} />
              </label>
              <div className="field-row">
                <label className="field">
                  <span className="field__label">Objective</span>
                  <select className="input" value={form.objective} onChange={(e) => set("objective", e.target.value as Recipe["objective"])}>
                    <option value="sft">SFT</option>
                    <option value="dpo">DPO</option>
                    <option value="grpo">GRPO</option>
                  </select>
                </label>
                <label className="field">
                  <span className="field__label">Quantization</span>
                  <select className="input" value={form.quantization} onChange={(e) => set("quantization", e.target.value as Recipe["quantization"])}>
                    <option value="qlora_4bit_nf4">QLoRA 4-bit</option>
                    <option value="lora_16bit">LoRA 16-bit</option>
                    <option value="none">None</option>
                  </select>
                </label>
              </div>
              <div className="field-row--3">
                <NumberField label="LoRA r" value={form.lora_r} onChange={(v) => set("lora_r", v)} />
                <NumberField label="LoRA alpha" value={form.lora_alpha} onChange={(v) => set("lora_alpha", v)} />
                <NumberField label="Learning rate" value={form.learning_rate} step="0.00001" onChange={(v) => set("learning_rate", v)} />
                <NumberField label="Epochs" value={form.epochs} onChange={(v) => set("epochs", v)} />
                <NumberField label="Batch size" value={form.batch_size} onChange={(v) => set("batch_size", v)} />
              </div>
              <label className="field">
                <span className="field__label">Checkpoint directory</span>
                <input className="input input--mono" value={form.output_dir} onChange={(e) => set("output_dir", e.target.value)} />
              </label>
              <button className="btn btn--block" disabled={busy !== null} onClick={() => void start()}>
                {busy === "start" ? "Starting…" : "Start training"}
              </button>
            </div>
          ) : (
            <p className="panel__empty">The recipe appears after Autotune has inspected your dataset.</p>
          )}
        </Panel>
      </div>

      <Panel index="C" title="Runs" aside={runs.data ? `${runs.data.length} total` : undefined}>
        <ErrorNotice message={runs.error} title="Could not load runs" />
        {runs.data?.length ? (
          <div className="table-wrap">
            <table className="table">
              <thead><tr><th>Run</th><th>Dataset</th><th>Model</th><th>Objective</th><th>Progress</th><th>Status</th><th>Started</th></tr></thead>
              <tbody>
                {runs.data.map((run) => (
                  <tr key={run.job_id}>
                    <td><Link href={`/training/${run.job_id}`} className="mono">{run.job_id}</Link></td>
                    <td className="mono">{run.dataset_name ?? "—"}</td>
                    <td className="mono">{run.base_model ?? "—"}</td>
                    <td className="mono uppercase">{run.objective ?? "—"}</td>
                    <td className="mono">{run.progress_percent}%</td>
                    <td><StatusBadge status={run.status} /></td>
                    <td>{formatWhen(run.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="panel__empty">No runs yet.</p>
        )}
      </Panel>
    </>
  );
}

function NumberField({ label, value, step, onChange }: { label: string; value: number; step?: string; onChange: (value: number) => void }) {
  return (
    <label className="field">
      <span className="field__label">{label}</span>
      <input className="input input--mono" type="number" step={step ?? "1"} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  );
}
