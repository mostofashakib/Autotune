"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { ErrorNotice, PageHero, Panel, StatusBadge } from "@/components/ui";
import { apiGet, apiPost, apiUpload, describeError } from "@/lib/api";
import { formatWhen } from "@/lib/format";
import type { CatalogModel, Recipe, TrainingSummary, UploadedDataset } from "@/lib/types";
import { usePolling } from "@/lib/usePolling";

type Overrides = Recipe & { output_dir: string };

function overridesFrom(dataset: UploadedDataset): Overrides {
  const name = dataset.file_name.replace(/\.[^.]+$/, "");
  return { ...dataset.recipe, output_dir: `./checkpoints/${name}` };
}

export default function TrainingPage() {
  const router = useRouter();
  const runs = usePolling<TrainingSummary[]>("/api/training", 5000, () => true);
  const catalog = usePolling<CatalogModel[]>("/api/models", 30000, () => true);
  const [customModel, setCustomModel] = useState(false);
  const [dataset, setDataset] = useState<UploadedDataset | null>(null);
  const [form, setForm] = useState<Overrides | null>(null);
  const [busy, setBusy] = useState<"upload" | "start" | "recommend" | null>(null);
  const [intensity, setIntensity] = useState<"quick" | "balanced" | "thorough">("balanced");
  const [presets, setPresets] = useState<UploadedDataset["presets"] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [ingestionProgress, setIngestionProgress] = useState<Record<string, number>>({});

  async function upload(file: File) {
    setBusy("upload");
    setError(null);
    try {
      setIngestionProgress({});
      const { job_id } = await apiUpload<{ job_id: string }>("/api/dataset/upload?background=true", file);
      let uploaded: UploadedDataset;
      for (;;) {
        const state = await apiGet<{ status: string; progress: Record<string, number>; error: string | null; result?: UploadedDataset }>(`/api/dataset/status/${job_id}`);
        setIngestionProgress(state.progress);
        if (state.status === "completed" && state.result) { uploaded = state.result; break; }
        if (["failed", "interrupted"].includes(state.status)) throw new Error(state.error ?? "Dataset ingestion failed");
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }
      setDataset(uploaded);
      setForm(overridesFrom(uploaded));
      setPresets(uploaded.presets);
      setIntensity("balanced");
      setCustomModel(false);
    } catch (cause) {
      setDataset(null);
      setForm(null);
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
        intensity,
      });
      router.push(`/training/${job_id}`);
    } catch (cause) {
      setError(describeError(cause));
      setBusy(null);
    }
  }

  async function recommend(base_model: string, objective: Recipe["objective"]) {
    if (!dataset) return;
    setBusy("recommend");
    setError(null);
    try {
      const result = await apiPost<{ recipe: Recipe; presets: UploadedDataset["presets"] }>("/api/recipe/recommend", { dataset_name: dataset.file_name, base_model, objective });
      setPresets(result.presets);
      setForm((current) => current ? { ...current, ...result.presets[intensity] } : current);
    } catch (cause) { setError(describeError(cause)); }
    finally { setBusy(null); }
  }

  function mark(key: keyof Recipe) {
    return form && dataset && form[key] !== dataset.recipe[key] ? " · changed" : "";
  }

  function reset() {
    if (!dataset) return;
    setForm(overridesFrom(dataset));
    setPresets(dataset.presets);
    setIntensity("balanced");
    setCustomModel(false);
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
        <Panel index="A" title="Dataset" aside="Structured data or documents (ZIP for a folder)">
          <div className="panel__body">
            <label className="field">
              <span className="field__label">File <small>saved to the data directory</small></span>
              <input
                className="input"
                type="file"
                accept=".jsonl,.json,.csv,.tsv,.parquet,.pq,.pdf,.docx,.txt,.md,.html,.zip"
                disabled={busy !== null}
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) void upload(file);
                }}
              />
            </label>
            {busy === "upload" ? <p className="text-sm text-muted-foreground">Inspecting… {Object.entries(ingestionProgress).map(([key, value]) => `${key.replaceAll("_", " ")}: ${value}`).join(" · ")}</p> : null}
            {dataset ? (
              <>
                {dataset.warnings?.map((warning) => <p key={warning} role="status">{warning}</p>)}
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
              <h3>Basic</h3>
              <label className="field">
                <span className="field__label">Base model{mark("base_model")}</span>
                <select className="input" disabled={busy !== null} value={customModel ? "custom" : form.base_model} onChange={(e) => {
                  if (e.target.value === "custom") { setCustomModel(true); return; }
                  setCustomModel(false);
                  void recommend(e.target.value, form.objective);
                }}>
                  {[{ label: "Small · up to 4B", min: 0, max: 4 }, { label: "Medium · 4B to 9B", min: 4, max: 9 }, { label: "Large · 9B to 15B", min: 9, max: 15 }].map((group) => (
                    <optgroup key={group.label} label={group.label}>
                      {catalog.data?.filter((model) => model.parameters_b > group.min && model.parameters_b <= group.max).map((model) => (
                        <option key={model.hf_id} value={model.hf_id} disabled={!model.fits}>
                          {model.display_name}{model.gated ? " · HF_TOKEN required" : ""}{model.fits ? "" : ` · ${model.reason}`}
                        </option>
                      ))}
                    </optgroup>
                  ))}
                  <option value="custom">Custom Hugging Face model</option>
                </select>
                {customModel ? <input aria-label="Custom Hugging Face model ID" className="input input--mono" value={form.base_model} onChange={(e) => set("base_model", e.target.value)} /> : null}
                <small>{catalog.data?.find((model) => model.hf_id === form.base_model)?.license}</small>
                {catalog.data?.find((model) => model.hf_id === form.base_model)?.gated ? <small>Set HF_TOKEN in .env and accept this model’s license on Hugging Face.</small> : null}
                <ErrorNotice message={catalog.error} title="Could not load model catalog" />
              </label>
              {customModel ? <button className="btn" disabled={busy !== null} onClick={() => void recommend(form.base_model, form.objective)}>Recommend for custom model</button> : null}
              <div className="field-row">
                <label className="field">
                  <span className="field__label">Objective{mark("objective")}</span>
                  <select className="input" disabled={busy !== null} value={form.objective} onChange={(e) => void recommend(form.base_model, e.target.value as Recipe["objective"])}>
                    <option value="sft">SFT</option>
                    <option value="dpo">DPO</option>
                    <option value="grpo">GRPO</option>
                  </select>
                </label>
                <label className="field">
                  <span className="field__label">Training intensity{intensity !== "balanced" ? " · changed" : ""}</span>
                  <select className="input" disabled={busy !== null || !presets} value={intensity} onChange={(e) => {
                    const next = e.target.value as typeof intensity;
                    setIntensity(next);
                    if (presets) setForm((current) => current ? { ...current, epochs: presets[next].epochs, learning_rate: presets[next].learning_rate } : current);
                  }}>
                    <option value="quick">Quick</option>
                    <option value="balanced">Balanced</option>
                    <option value="thorough">Thorough</option>
                  </select>
                  <small>{form.epochs} epochs · learning rate {form.learning_rate}</small>
                </label>
              </div>
              <details>
                <summary>Advanced</summary>
                <label className="field">
                  <span className="field__label">Quantization{mark("quantization")}</span>
                  <select className="input" value={form.quantization} onChange={(e) => set("quantization", e.target.value as Recipe["quantization"])}>
                    <option value="qlora_4bit_nf4">QLoRA 4-bit</option>
                    <option value="lora_16bit">LoRA 16-bit</option>
                    <option value="none">None</option>
                  </select>
                </label>
              <div className="field-row--3">
                <NumberField label={"LoRA r" + mark("lora_r")} value={form.lora_r} onChange={(v) => set("lora_r", v)} />
                <NumberField label={"LoRA alpha" + mark("lora_alpha")} value={form.lora_alpha} onChange={(v) => set("lora_alpha", v)} />
                <NumberField label={"Learning rate" + mark("learning_rate")} value={form.learning_rate} step="0.00001" onChange={(v) => set("learning_rate", v)} />
                <NumberField label={"Epochs" + mark("epochs")} value={form.epochs} onChange={(v) => set("epochs", v)} />
                <NumberField label={"Batch size" + mark("batch_size")} value={form.batch_size} onChange={(v) => set("batch_size", v)} />
                <NumberField label={"Gradient accumulation" + mark("gradient_accumulation_steps")} value={form.gradient_accumulation_steps} onChange={(v) => set("gradient_accumulation_steps", v)} />
                <NumberField label={"Max sequence length" + mark("max_seq_length")} value={form.max_seq_length} onChange={(v) => set("max_seq_length", v)} />
              </div>
              <label className="field">
                <span className="field__label">Checkpoint directory{dataset && form.output_dir !== overridesFrom(dataset).output_dir ? " · changed" : ""}</span>
                <input className="input input--mono" value={form.output_dir} onChange={(e) => set("output_dir", e.target.value)} />
              </label>
              </details>
              <button className="btn btn--block" disabled={busy !== null} onClick={reset}>Reset to recommended</button>
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
