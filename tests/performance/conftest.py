"""Performance test fixtures — timing utilities and SLA assertions."""

import time

import pytest


class SLATimer:
    """Context manager for measuring execution time against SLA."""

    def __init__(self, sla_ms: float, label: str = ""):
        self.sla_ms = sla_ms
        self.label = label
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
        return False

    def assert_within_sla(self):
        assert self.elapsed_ms < self.sla_ms, (
            f"{self.label}: {self.elapsed_ms:.2f}ms exceeds SLA of {self.sla_ms}ms"
        )


@pytest.fixture
def sla_timer():
    """Provide SLA timer factory for performance assertions."""

    def _timer(sla_ms: float, label: str = ""):
        return SLATimer(sla_ms, label)

    return _timer
