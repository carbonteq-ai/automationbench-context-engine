"""Recorded episodes as the environment stores them (one JSON line per episode).

`iter_episodes` never yields an excluded task. `Episode` is a thin read-only view with the
accessors the record builder needs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Episode:
    record: dict
    path: str | None = None

    @property
    def trace(self) -> dict:
        return self.record["traces"][0]

    @property
    def id(self) -> str:
        return str(self.record.get("id") or self.trace.get("id"))

    @property
    def task(self) -> str:
        return self.trace["task"]["data"]["name"]

    @property
    def task_data(self) -> dict:
        return self.trace["task"]["data"]

    @property
    def messages(self) -> list[dict]:
        return [n["message"] for n in self.trace["nodes"] if n.get("message")]

    @property
    def nodes(self) -> list[dict]:
        return self.trace["nodes"]

    @property
    def events(self) -> list[dict]:
        return self.trace.get("tool_execution_events") or []

    @property
    def native_results(self) -> list[dict]:
        return ((self.trace.get("info") or {}).get("automationbench") or {}).get("assertions") or []

    @property
    def metrics(self) -> dict:
        return self.trace.get("metrics") or {}

    @property
    def success(self) -> bool:
        return self.metrics.get("task_completed_correctly") == 1.0

    @property
    def assessment_batches(self) -> list[dict]:
        return self.trace.get("assessment_batches") or self.record.get("assessment_batches") or []

    @property
    def stop_condition(self) -> str | None:
        return self.trace.get("stop_condition")

    @property
    def turn_progress(self) -> list[dict]:
        return (self.trace.get("info") or {}).get("automationbench_turn_progress") or []
