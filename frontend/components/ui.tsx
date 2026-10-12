import type { ReactNode } from "react";
import { statusTone } from "@/lib/format";

export function StatusBadge({ status }: { status: string }) {
  return (
    <span className={`status status--${statusTone(status)}`}>
      <i aria-hidden="true" />
      {status}
    </span>
  );
}

export function ErrorNotice({ message, title = "Request failed" }: { message: string | null; title?: string }) {
  if (!message) return null;
  return (
    <div className="notice notice--error" role="alert">
      <strong>{title}</strong>
      {message}
    </div>
  );
}

export function PageHero({
  eyebrow,
  title,
  children,
  readout,
}: {
  eyebrow: ReactNode;
  title: ReactNode;
  children?: ReactNode;
  readout?: { label: string; value: ReactNode }[];
}) {
  return (
    <section className="page-hero">
      <div className="page-hero__copy">
        <p className="page-hero__eyebrow">{eyebrow}</p>
        <h1>{title}</h1>
        {children ? <p>{children}</p> : null}
      </div>
      {readout ? (
        <div className="readout">
          {readout.map((item) => (
            <div key={item.label}>
              <span>{item.label}</span>
              <strong>{item.value}</strong>
            </div>
          ))}
        </div>
      ) : null}
    </section>
  );
}

export function Panel({
  index,
  title,
  aside,
  children,
}: {
  index?: string;
  title: string;
  aside?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="panel">
      <header className="panel__heading">
        <div>
          {index ? <span>{index}</span> : null}
          <h2>{title}</h2>
        </div>
        {aside ? <p>{aside}</p> : null}
      </header>
      {children}
    </section>
  );
}

export function LogConsole({ title, lines, idle }: { title: string; lines: string[]; idle: string }) {
  return (
    <section className="console" aria-label={title}>
      <div className="console__bar">{title}</div>
      <div className="console__output">
        {lines.length ? lines.map((line, index) => <p key={index}>{line}</p>) : <p className="console__idle">{idle}</p>}
      </div>
    </section>
  );
}

/** Measured loss per logged step. Nothing is drawn until the trainer reports a value. */
export function LossChart({ values }: { values: number[] }) {
  if (values.length < 2) {
    return <p className="panel__empty">The loss curve appears once the trainer has logged two steps.</p>;
  }
  const width = 600;
  const height = 160;
  const max = Math.max(...values);
  const min = Math.min(...values);
  const span = max - min || 1;
  const points = values
    .map((value, index) => `${(index / (values.length - 1)) * width},${height - ((value - min) / span) * height}`)
    .join(" ");
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="h-40 w-full" role="img" aria-label="Training loss by logged step">
      <polyline points={points} fill="none" stroke="var(--primary)" strokeWidth="2.5" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}
