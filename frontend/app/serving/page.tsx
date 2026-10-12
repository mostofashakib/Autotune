"use client";

import { useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";
import { ErrorNotice, LogConsole, PageHero, Panel, StatusBadge } from "@/components/ui";
import { apiPost, describeError } from "@/lib/api";
import type { ModelEndpoint, ServingState } from "@/lib/types";
import { usePolling } from "@/lib/usePolling";

export default function ServingPage() {
  return (
    <Suspense>
      <Serving />
    </Suspense>
  );
}

function Serving() {
  const params = useSearchParams();
  const serving = usePolling<ServingState>("/api/serve/status", 3000, () => true);
  const [modelPath, setModelPath] = useState(params.get("model") ?? "");
  const [port, setPort] = useState(8000);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const state = serving.data;

  async function call(path: string, payload?: unknown) {
    setBusy(true);
    setError(null);
    try {
      await apiPost(path, payload);
    } catch (cause) {
      setError(describeError(cause));
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <PageHero
        eyebrow="03 / Serving"
        title={<>Put a checkpoint <em>online</em>.</>}
        readout={[
          { label: "Server", value: state ? <StatusBadge status={state.status} /> : "…" },
          { label: "Endpoint", value: state?.base_url ? `${state.base_url}/v1` : "—" },
        ]}
      >
        Serve a checkpoint with vLLM as an OpenAI-compatible endpoint, then ask it and its base model the same prompt.
      </PageHero>
      <ErrorNotice message={error ?? serving.error} />

      <div className="grid-2">
        <Panel index="A" title="vLLM server">
          <div className="panel__body">
            <label className="field">
              <span className="field__label">Checkpoint or model <small>path on the API server</small></span>
              <input className="input input--mono" placeholder="./checkpoints/my-model" value={modelPath} onChange={(e) => setModelPath(e.target.value)} />
            </label>
            <label className="field">
              <span className="field__label">Port</span>
              <input className="input input--mono" type="number" value={port} onChange={(e) => setPort(Number(e.target.value))} />
            </label>
            <div className="flex flex-wrap gap-3">
              <button className="btn" disabled={busy || !modelPath.trim()} onClick={() => void call("/api/serve/start", { model_path: modelPath.trim(), port })}>
                Start server
              </button>
              <button className="btn btn--ghost" disabled={busy || state?.status === "stopped"} onClick={() => void call("/api/serve/stop")}>
                Stop server
              </button>
            </div>
          </div>
        </Panel>
        <LogConsole title="vLLM log" lines={state?.log_tail ? state.log_tail.split("\n") : []} idle="No server output yet." />
      </div>

      <Playground served={state?.status === "online" && state.base_url ? { base_url: `${state.base_url}/v1`, model: modelPath } : null} />
    </>
  );
}

type Answer = { text: string | null; error: string | null; loading: boolean };
const IDLE: Answer = { text: null, error: null, loading: false };

function Playground({ served }: { served: ModelEndpoint | null }) {
  const [prompt, setPrompt] = useState("");
  const [maxTokens, setMaxTokens] = useState(1024);
  const [base, setBase] = useState<ModelEndpoint>({ base_url: "", model: "" });
  const [trained, setTrained] = useState<ModelEndpoint>({ base_url: "", model: "" });
  const [answers, setAnswers] = useState<Record<"base" | "trained", Answer>>({ base: IDLE, trained: IDLE });

  async function ask(which: "base" | "trained", model: ModelEndpoint) {
    setAnswers((current) => ({ ...current, [which]: { text: null, error: null, loading: true } }));
    try {
      const { response } = await apiPost<{ response: string }>("/api/playground/generate", {
        prompt, max_tokens: maxTokens, model: { base_url: model.base_url.trim(), model: model.model.trim() },
      });
      setAnswers((current) => ({ ...current, [which]: { text: response, error: null, loading: false } }));
    } catch (cause) {
      setAnswers((current) => ({ ...current, [which]: { text: null, error: describeError(cause), loading: false } }));
    }
  }

  const ready = prompt.trim() && base.base_url.trim() && base.model.trim() && trained.base_url.trim() && trained.model.trim();

  return (
    <Panel index="B" title="Playground" aside="same prompt, two models">
      <div className="panel__body">
        <div className="field-row">
          <EndpointInputs title="Base model" value={base} onChange={setBase} />
          <div className="space-y-2">
            <EndpointInputs title="Trained model" value={trained} onChange={setTrained} />
            {served ? (
              <button className="btn btn--ghost" onClick={() => setTrained(served)}>Use the running server</button>
            ) : null}
          </div>
        </div>
        <label className="field">
          <span className="field__label">Prompt</span>
          <textarea className="input" value={prompt} onChange={(e) => setPrompt(e.target.value)} />
        </label>
        <div className="flex flex-wrap items-end gap-3">
          <label className="field w-40">
            <span className="field__label">Max tokens</span>
            <input className="input input--mono" type="number" min={1} value={maxTokens} onChange={(e) => setMaxTokens(Number(e.target.value))} />
          </label>
          <button className="btn" disabled={!ready} onClick={() => { void ask("base", base); void ask("trained", trained); }}>
            Ask both
          </button>
        </div>
        <div className="answers">
          {(["base", "trained"] as const).map((which) => (
            <div key={which} className="answer">
              <h3>{which}</h3>
              {answers[which].loading ? <p className="text-muted-foreground">Waiting for the model…</p> : null}
              {answers[which].error ? <ErrorNotice message={answers[which].error} title="Model call failed" /> : null}
              {answers[which].text !== null ? <p>{answers[which].text}</p> : null}
            </div>
          ))}
        </div>
      </div>
    </Panel>
  );
}

function EndpointInputs({ title, value, onChange }: { title: string; value: ModelEndpoint; onChange: (value: ModelEndpoint) => void }) {
  return (
    <fieldset className="space-y-2 border border-foreground/20 p-3">
      <legend className="px-1 text-xs font-semibold uppercase tracking-wider">{title}</legend>
      <input className="input input--mono" placeholder="http://localhost:11434/v1" aria-label={`${title} base URL`} value={value.base_url} onChange={(e) => onChange({ ...value, base_url: e.target.value })} />
      <input className="input input--mono" placeholder="model name" aria-label={`${title} model`} value={value.model} onChange={(e) => onChange({ ...value, model: e.target.value })} />
    </fieldset>
  );
}
