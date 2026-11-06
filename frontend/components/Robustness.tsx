import { formatScore, riskTone } from "@/lib/format";
import type { RobustnessScore } from "@/lib/types";

const WIDTH = 260;
const HEIGHT = 130;
const PAD = { left: 38, right: 10, top: 10, bottom: 24 };
const MUTED = "var(--muted-foreground)";

function RiskChip({ label, value }: { label: string; value: string | null | undefined }) {
  return (
    <span className={`status status--${riskTone(value)}`}>
      <i aria-hidden="true" />
      {label}: {value ?? "Not measured"}
    </span>
  );
}

function CurveChart({ score }: { score: RobustnessScore }) {
  const { k_labels: ks, pass_at_k: atK, pass_pow_k: powK } = score;
  const plotW = WIDTH - PAD.left - PAD.right;
  const plotH = HEIGHT - PAD.top - PAD.bottom;
  const x = (i: number) => PAD.left + (plotW * i) / (ks.length - 1);
  const y = (value: number) => PAD.top + plotH * (1 - value);
  const line = (values: number[]) => values.map((v, i) => `${x(i)},${y(v)}`).join(" ");
  return (
    <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label="pass@k and pass^k by k" className="w-full max-w-sm">
      {[0, 0.5, 1].map((tick) => (
        <g key={tick}>
          <line x1={PAD.left} x2={WIDTH - PAD.right} y1={y(tick)} y2={y(tick)} stroke="var(--border)" strokeWidth="0.5" />
          <text x={PAD.left - 5} y={y(tick) + 3} textAnchor="end" fontSize="8" fill={MUTED}>{tick * 100}%</text>
        </g>
      ))}
      {ks.map((k, i) => (
        <text key={k} x={x(i)} y={HEIGHT - 12} textAnchor="middle" fontSize="8" fill={MUTED}>{k}</text>
      ))}
      <text x={WIDTH - PAD.right} y={HEIGHT - 2} textAnchor="end" fontSize="7" fill={MUTED}>k (samples drawn)</text>
      <polyline points={line(atK)} fill="none" stroke="var(--primary)" strokeWidth="1.5" />
      <polyline points={line(powK)} fill="none" stroke="var(--foreground)" strokeWidth="1.5" strokeDasharray="3 2" />
    </svg>
  );
}

function Curves({ score }: { score: RobustnessScore }) {
  if (score.k_labels.length < 2) {
    return (
      <p className="mono">
        k={score.k_labels[0]} · pass@k {formatScore(score.pass_at_k[0])} · pass^k {formatScore(score.pass_pow_k[0])}
      </p>
    );
  }
  return (
    <>
      <CurveChart score={score} />
      <p className="mono">
        <span style={{ color: "var(--primary)" }}>solid: pass@k</span> · dashed: pass^k
      </p>
    </>
  );
}

export function RobustnessCard({ label, score }: { label: string; score: RobustnessScore }) {
  const kappa = score.grader_judge_kappa;
  return (
    <div className="space-y-3">
      <h3 className="mono">{label}</h3>
      <Curves score={score} />
      <p className="mono">Grader/judge kappa: {typeof kappa === "number" ? kappa.toFixed(2) : "No judge configured"}</p>
      <div className="flex flex-wrap gap-2">
        <RiskChip label="Reward hacking" value={score.reward_hacking_risk} />
        <RiskChip label="Memorization" value={score.memorization_risk} />
        <RiskChip label="Contamination" value={score.contamination_risk} />
      </div>
    </div>
  );
}
