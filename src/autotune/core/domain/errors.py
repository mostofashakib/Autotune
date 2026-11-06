"""Failures the core domain raises for callers to handle."""


class MissingExtraError(RuntimeError):
    """A feature needs an optional dependency group that is not installed."""
