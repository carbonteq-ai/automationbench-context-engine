"""The teacher context for one AutomationBench attempt: the entry point of the feedback engine.

`teacher_context` gives the text the SDPO teacher reads with an attempt: for a solved attempt an outcome line, for a
failed one the engine's feedback in design `S1FUHWNLGOJYIV` (the design the 50.6% SDPO adapter was trained with),
plus the first turn the feedback locates, which bounds the tokens a failed attempt is trained on.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from abce.teacher.attempt import make_engine, point_turn
from abce.teacher.engine import EngineContext
from abce.teacher.text import sc_live
from abce.teacher.versions import SHOWN_ENGINE

DESIGN = "S1FUHWNLGOJYIV"

HEAD = "The following is feedback from your earlier attempt at this task:"

TAIL = "Correctly solve the original question."

SUCCESS_LINE = "it succeeded; every check of the task passed."

ENGINE_OPTIONS = {
    "line_budget": None,
    "char_budget": None,
    "not_done_source": "benchmark",
    "final_note": True,
}


@dataclass(frozen=True)
class TeacherContext:
    """What the teacher reads with an attempt, or why there is nothing to read."""

    context: (
        dict | None
    )  # {"first_message": str, "notes": {turn: str}, "final_note": str | None, ...}
    last_turn: (
        int | None
    )  # first located wrong decision; a failed attempt is trained up to this turn
    forbidden: bool  # the note at last_turn says the task forbids the action
    reason: str | None  # why there is no context, e.g. "no_engine_feedback"


def task_text(chain: dict) -> str:
    """The task's user message, as the student saw it."""
    return next(m["content"] for m in chain["messages0"] if m["role"] == "user")


def outcome_context(task: str, success: bool, outcome: dict | None = None) -> dict:
    """The outcome-only context: the task, one outcome line and the closing instruction."""
    if success:
        line = SUCCESS_LINE
    else:
        passed, total = (
            (outcome or {}).get("required_passed"),
            (outcome or {}).get("required_total"),
        )
        if passed is not None and total:
            line = f"it failed; {int(total - passed)} of the task's {int(total)} required checks were not met."
        else:
            line = "it failed; some of the task's checks were not met."
    return {"first_message": f"{task}\n\n{HEAD} {line}\n\n{TAIL}", "notes": {}, "final_note": None}


@lru_cache(maxsize=1)
def engines() -> tuple[EngineContext, EngineContext]:
    """(hidden, shown): the engine with expected values hidden, and the values-shown view its pointers use."""
    shown = EngineContext(
        expected_values=True,
        pool=False,
        fix_hints=list(SHOWN_ENGINE["fix_hints"]),
        options=dict(ENGINE_OPTIONS),
        withhold_untrue=True,
    )
    return make_engine("T-A2"), shown


def teacher_context(
    record: dict, siblings: list[dict], chain: dict, outcome: dict
) -> TeacherContext:
    """The teacher context of one scored attempt.

    record: the attempt's episode record; siblings: every record of the same task in the round (the attempt included);
    chain: the attempt's conversation (messages0, segments); outcome: its scored outcome (success, required checks).
    """
    task = task_text(chain)
    if outcome.get("success") == 1:
        return TeacherContext(outcome_context(task, True), None, False, None)
    hidden, shown = engines()
    engine = hidden(record, siblings, "A", "both")
    if engine is None:
        return TeacherContext(None, None, False, "no_engine_feedback")
    last = point_turn(engine)
    forbidden = "forbids" in (engine.get("notes") or {}).get(last, "") if last else False
    text = sc_live(DESIGN, engine, shown(record, siblings, "A", "both"), record, chain)
    return TeacherContext(text, last, forbidden, None)
