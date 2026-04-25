from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class StepTiming:
    name: str
    start: float = 0.0
    end: float = 0.0

    @property
    def elapsed_ms(self) -> float:
        if self.end > 0:
            return (self.end - self.start) * 1000
        return 0.0


class TimingTracker:
    def __init__(self) -> None:
        self._steps: List[StepTiming] = []
        self._current: Optional[StepTiming] = None
        self._lock = threading.Lock()

    def start_step(self, name: str) -> None:
        with self._lock:
            if self._current is not None:
                self._current.end = time.perf_counter()
                self._steps.append(self._current)
            self._current = StepTiming(name=name, start=time.perf_counter())

    def finish_step(self) -> StepTiming | None:
        with self._lock:
            if self._current is None:
                return None
            self._current.end = time.perf_counter()
            step = self._current
            self._steps.append(step)
            self._current = None
            return step

    def finish_all(self) -> None:
        with self._lock:
            if self._current is not None:
                self._current.end = time.perf_counter()
                self._steps.append(self._current)
                self._current = None

    def get_results(self) -> Dict[str, float]:
        self.finish_all()
        return {s.name: round(s.elapsed_ms, 2) for s in self._steps}

    def get_total_ms(self) -> float:
        self.finish_all()
        if not self._steps:
            return 0.0
        return round(
            (self._steps[-1].end - self._steps[0].start) * 1000, 2
        )

    def reset(self) -> None:
        with self._lock:
            self._steps.clear()
            self._current = None

    def summary(self) -> str:
        results = self.get_results()
        total = self.get_total_ms()
        lines = [f"总耗时: {total:.1f}ms"]
        for name, ms in results.items():
            pct = (ms / total * 100) if total > 0 else 0
            lines.append(f"  {name}: {ms:.1f}ms ({pct:.1f}%)")
        return "\n".join(lines)
