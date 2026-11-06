"use client";

import { useEffect, useState } from "react";
import { apiGet, describeError } from "@/lib/api";
import { startPolling } from "@/lib/polling";

/**
 * Poll `path` every `intervalMs` while `keepPolling(data)` holds, and once more on mount.
 * Errors are reported, not hidden, and polling continues so a restarted API recovers.
 */
export function usePolling<T>(path: string | null, intervalMs: number, keepPolling: (data: T) => boolean) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);

  useEffect(() => {
    if (!path || done) return;
    return startPolling(
      async (signal) => {
        const next = await apiGet<T>(path, signal);
        setData(next);
        setError(null);
        if (!keepPolling(next)) setDone(true);
      },
      intervalMs,
      (cause) => setError(describeError(cause)),
    );
    // keepPolling is a pure predicate supplied inline by callers.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, intervalMs, done]);

  return { data, error, refresh: () => setDone(false) };
}
