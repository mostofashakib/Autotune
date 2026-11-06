"""Failures the core domain raises for callers to handle."""


class MissingExtraError(RuntimeError):
    """A feature needs an optional dependency group that is not installed."""


class ProviderError(RuntimeError):
    """An external service (RunPod, vLLM) failed or rejected a request."""


class HardwareUnavailable(RuntimeError):
    """The machine lacks hardware a feature needs, such as a CUDA GPU."""
