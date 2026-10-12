/** A dial with its needle past center: tuning toward a target. */
export default function AutotuneMark() {
  return (
    <svg viewBox="0 0 32 32" fill="none" stroke="currentColor" strokeWidth="2.4" aria-hidden="true">
      <path d="M5 22a11 11 0 1 1 22 0" strokeLinecap="square" />
      <path d="M16 22l6-8" strokeLinecap="square" stroke="var(--primary)" />
      <path d="M16 22h.01" strokeLinecap="round" strokeWidth="4" />
    </svg>
  );
}
