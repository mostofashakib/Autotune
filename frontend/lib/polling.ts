/** Schedule each request after the previous one settles, and abort on cleanup. */
export function startPolling(
  task: (signal: AbortSignal) => Promise<void>,
  intervalMs: number,
  onError: (cause: unknown) => void,
): () => void {
  const controller = new AbortController();
  let timer: ReturnType<typeof setTimeout>;

  async function run() {
    try {
      await task(controller.signal);
    } catch (cause) {
      if (!controller.signal.aborted) onError(cause);
    } finally {
      if (!controller.signal.aborted) timer = setTimeout(run, intervalMs);
    }
  }

  timer = setTimeout(run, 0);
  return () => {
    controller.abort();
    clearTimeout(timer);
  };
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    function onAbort() {
      clearTimeout(timer);
      reject(signal.reason);
    }
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

/**
 * Request until `isDone` holds, waiting `intervalMs` after each request settles.
 * Resolves with the first done value. Rejects with the request error, or with an
 * AbortError once `signal` aborts, and never schedules another request after that.
 */
export async function pollUntil<T>(
  fetchOnce: (signal: AbortSignal) => Promise<T>,
  isDone: (value: T) => boolean,
  intervalMs: number,
  signal: AbortSignal,
): Promise<T> {
  for (;;) {
    signal.throwIfAborted();
    const value = await fetchOnce(signal);
    signal.throwIfAborted();
    if (isDone(value)) return value;
    await sleep(intervalMs, signal);
  }
}
