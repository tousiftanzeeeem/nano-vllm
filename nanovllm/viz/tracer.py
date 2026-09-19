"""Event tracer for the visualization layer.

Dependency-free on purpose: the engine imports this module, so it must not pull
in torch, transformers or anything else heavy.

Tracing is off by default and costs one `is None` check per call site.
"""

from contextlib import contextmanager


class Tracer:

    def __init__(self):
        self.events: list[dict] = []
        self.step = 0

    def emit(self, kind: str, **fields):
        self.events.append({"step": self.step, "kind": kind, **fields})

    def drain(self) -> list[dict]:
        events, self.events = self.events, []
        return events


_TRACER: Tracer | None = None


def tracer() -> Tracer | None:
    """Return the active tracer, or None when tracing is off."""
    return _TRACER


@contextmanager
def tracing():
    global _TRACER
    previous, _TRACER = _TRACER, Tracer()
    try:
        yield _TRACER
    finally:
        _TRACER = previous
