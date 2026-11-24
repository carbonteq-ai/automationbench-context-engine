"""The teacher contexts of the 50.6% SDPO run are reproduced exactly.

The fixture holds every round-1 attempt of two training tasks of that run (16 attempts: their episode records,
conversations and scored outcomes) and the teacher context the run built for each. The engine must give the same
context, located turn, forbidden flag and refusal reason, character for character.
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path

import pytest

pytest.importorskip(
    "automationbench_v1", reason="needs the AutomationBench environment on PYTHONPATH"
)

from abce.teacher import teacher_context  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "teacher"


def read_jsonl_gz(name: str) -> list[dict]:
    with gzip.open(FIXTURE / name, "rt") as fh:
        return [json.loads(line) for line in fh]


def task_name(record: dict) -> str:
    return record["traces"][0]["task"]["data"]["task_name"]


def cases() -> list[tuple[dict, list[dict], dict, dict, dict]]:
    records = {r["id"]: r for r in read_jsonl_gz("episodes.jsonl.gz")}
    chains = {c["episode_id"]: c for c in read_jsonl_gz("chains.jsonl.gz")}
    outcomes = {o["episode_id"]: o for o in json.loads((FIXTURE / "outcomes.json").read_text())}
    expected = {
        e["episode_id"]: e
        for e in map(json.loads, (FIXTURE / "expected.jsonl").read_text().splitlines())
    }
    siblings = defaultdict(list)
    for record in records.values():
        siblings[task_name(record)].append(record)
    return [
        (records[i], siblings[task_name(records[i])], chains[i], outcomes[i], expected[i])
        for i in sorted(expected)
    ]


@pytest.mark.parametrize("record, siblings, chain, outcome, expected", cases(), ids=lambda v: None)
def test_teacher_context_matches_the_trained_run(record, siblings, chain, outcome, expected):
    got = teacher_context(record, siblings, chain, outcome)
    assert json.loads(json.dumps(got.context)) == expected.get("context")
    assert got.last_turn == expected.get("last")
    assert got.forbidden == expected.get("forbidden")
    assert got.reason == expected.get("why")


def test_solved_attempts_get_only_the_outcome_line():
    solved = [c for c in cases() if c[3].get("success") == 1]
    assert solved
    for record, siblings, chain, outcome, _ in solved:
        got = teacher_context(record, siblings, chain, outcome)
        assert got.context["notes"] == {} and "it succeeded" in got.context["first_message"]
        assert got.last_turn is None
