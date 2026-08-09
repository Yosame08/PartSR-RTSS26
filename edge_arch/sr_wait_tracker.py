import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Optional


@dataclass(frozen=True)
class PendingSRTask:
    number: int
    predicted_seconds: float


class SRWaitTracker:
    """Tracks one FIFO SR worker without sharing mutable state across processes."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._pending: Deque[PendingSRTask] = deque()
        self._total_predicted = 0.0
        self._active_number: Optional[int] = None
        self._active_started_at: Optional[float] = None

    def reserve(self, number: int, predicted_seconds: float) -> None:
        if number <= 0:
            raise ValueError(f"task number must be positive, got {number}")
        if not math.isfinite(predicted_seconds) or predicted_seconds < 0:
            raise ValueError(f"predicted_seconds must be finite and non-negative, got {predicted_seconds}")
        if self._pending and number <= self._pending[-1].number:
            raise ValueError(f"task numbers must increase, got {number}")
        self._pending.append(PendingSRTask(number, predicted_seconds))
        self._total_predicted += predicted_seconds

    def start(self, number: int) -> None:
        if not self._pending or self._pending[0].number != number:
            expected = self._pending[0].number if self._pending else None
            raise RuntimeError(f"SR task {number} started out of order; expected {expected}")
        if self._active_number is not None:
            raise RuntimeError(f"SR task {self._active_number} is already active")
        self._active_number = number
        self._active_started_at = self._clock()

    def finish(self, number: int) -> None:
        if self._active_number != number:
            raise RuntimeError(f"SR task {number} finished while active task is {self._active_number}")
        task = self._pending.popleft()
        self._total_predicted = max(0.0, self._total_predicted - task.predicted_seconds)
        self._active_number = None
        self._active_started_at = None

    def cancel(self, number: int) -> None:
        """Remove a task that has not started running in the worker."""
        if self._active_number == number:
            raise RuntimeError(f"SR task {number} is already active")
        for index, task in enumerate(self._pending):
            if task.number == number:
                del self._pending[index]
                self._total_predicted = max(0.0, self._total_predicted - task.predicted_seconds)
                return
        raise RuntimeError(f"unknown SR task {number}")

    def estimated_wait(self) -> float:
        if not self._pending:
            return 0.0
        if self._active_started_at is None:
            return self._total_predicted
        elapsed = max(0.0, self._clock() - self._active_started_at)
        remaining_active = max(0.0, self._pending[0].predicted_seconds - elapsed)
        queued = self._total_predicted - self._pending[0].predicted_seconds
        return max(0.0, remaining_active + queued)

    @property
    def pending_count(self) -> int:
        return len(self._pending)
