"use client";

import { useState } from "react";
import { ErrorNotice, PageHero, Panel, StatusBadge } from "@/components/ui";
import { apiPost, describeError } from "@/lib/api";
import { formatWhen } from "@/lib/format";
import type { Gpu, Pod } from "@/lib/types";
import { usePolling } from "@/lib/usePolling";

export default function CloudPage() {
  const gpus = usePolling<Gpu[]>("/api/cloud/gpus", 60000, () => false);
  const pods = usePolling<Pod[]>("/api/cloud/pods", 10000, () => true);
  const [gpuId, setGpuId] = useState("");
  const [podName, setPodName] = useState("autotune-trainer");
  const [diskGb, setDiskGb] = useState(50);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const selected = gpuId || gpus.data?.[0]?.id || "";

  async function act(key: string, path: string, payload?: unknown) {
    setBusy(key);
    setError(null);
    try {
      await apiPost(path, payload);
      pods.refresh();
    } catch (cause) {
      setError(describeError(cause));
    } finally {
      setBusy(null);
    }
  }

  return (
    <>
      <PageHero eyebrow="04 / Cloud" title={<>Rent a <em>GPU</em>.</>}>
        Launch a RunPod pod for training runs too large for this machine. Prices and pods come straight from RunPod.
      </PageHero>
      <ErrorNotice message={error} title="RunPod request failed" />

      <div className="grid-2">
        <Panel index="A" title="Launch a pod">
          <div className="panel__body">
            <ErrorNotice message={gpus.error} title="GPU catalog unavailable" />
            <label className="field">
              <span className="field__label">GPU</span>
              <select className="input" value={selected} disabled={!gpus.data?.length} onChange={(e) => setGpuId(e.target.value)}>
                {(gpus.data ?? []).map((gpu) => (
                  <option key={gpu.id} value={gpu.id}>{gpu.name} · {gpu.memory_gb} GB · ${gpu.community_price.toFixed(2)}/hr</option>
                ))}
              </select>
            </label>
            <div className="field-row">
              <label className="field">
                <span className="field__label">Pod name</span>
                <input className="input input--mono" value={podName} onChange={(e) => setPodName(e.target.value)} />
              </label>
              <label className="field">
                <span className="field__label">Disk (GB)</span>
                <input className="input input--mono" type="number" min={10} value={diskGb} onChange={(e) => setDiskGb(Number(e.target.value))} />
              </label>
            </div>
            <button
              className="btn btn--block"
              disabled={busy !== null || !selected || !podName.trim()}
              onClick={() => void act("launch", "/api/cloud/launch", { gpu_type: selected, pod_name: podName.trim(), disk_gb: diskGb })}
            >
              {busy === "launch" ? "Launching…" : "Launch pod"}
            </button>
          </div>
        </Panel>

        <Panel index="B" title="Your pods" aside="launched from Autotune">
          <ErrorNotice message={pods.error} title="Could not load pods" />
          {pods.data?.length ? (
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th>Pod</th><th>GPU</th><th>Cost</th><th>Status</th><th /></tr></thead>
                <tbody>
                  {pods.data.map((pod) => (
                    <tr key={pod.pod_id}>
                      <td>
                        <span className="mono block">{pod.name}</span>
                        <span className="mono text-muted-foreground">{pod.pod_id} · {formatWhen(pod.launched_at)}</span>
                        {pod.ssh_ip ? <span className="mono block">ssh root@{pod.ssh_ip} -p {pod.ssh_port}</span> : null}
                        {pod.live_error ? <span className="mt-1 block text-red-800">Live status unavailable: {pod.live_error}</span> : null}
                      </td>
                      <td className="mono">{pod.gpu_name}</td>
                      <td className="mono">{pod.cost_per_hr ? `$${pod.cost_per_hr.toFixed(2)}/hr` : "—"}</td>
                      <td><StatusBadge status={pod.status.toLowerCase()} /></td>
                      <td>
                        {pod.status !== "TERMINATED" ? (
                          <button className="btn btn--ghost" disabled={busy !== null} onClick={() => void act(pod.pod_id, `/api/cloud/stop/${pod.pod_id}`)}>
                            {busy === pod.pod_id ? "Stopping…" : "Terminate"}
                          </button>
                        ) : null}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="panel__empty">No pods launched from Autotune.</p>
          )}
        </Panel>
      </div>
    </>
  );
}
