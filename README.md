# AutomationBench context engine

Reads one attempt at an [AutomationBench](https://github.com/zapier/AutomationBench) task and writes located
feedback on it. The feedback names the checks the attempt failed (required actions not taken, forbidden actions
taken, wrong values) and puts each note at the turn where the attempt went wrong. Values the attempt never saw stay
hidden. Its job is to be the privileged context of a self-distillation (SDPO) teacher. The student never sees it.

Version 1.0.0 is the engine the [carbonteq/qwen3.5-4b-automationbench-sdpo](https://huggingface.co/carbonteq/qwen3.5-4b-automationbench-sdpo)
adapter was trained with (50.6% on AutomationBench test120, from 42.4% for its SFT start). On all 624 round-1 attempts
of that training run, it gives the same teacher context, character for character, as the copy the run used. That copy
is vendored in posttrain branch `feat/automationbench-sdpo` under `scripts/qualification/abce_vendor`.

## How it works

```
attempt (episode record + conversation)      task (manifest, grader checks, tool contracts)
                 \                                  /
                  record: calls, observed values, checks met and failed, located at turns
                                      |
                  checks: truth (every line is true of the attempt), leak (no unseen value is shown)
                                      |
                  render: plain-language lines, with values hidden or shown
                                      |
                  teacher: the outcome line, the first message and the per-turn notes the teacher reads
```

| Package | What it does |
| --- | --- |
| `abce.record` | reads an episode into located items: calls, values observed, the grader's checks and their results |
| `abce.tasks`, `abce.paths` | load task manifests and contracts from the AutomationBench environment checkout |
| `abce.why`, `abce.fix`, `abce.solution` | why a check failed, and what would satisfy it, from the attempt's own records |
| `abce.checks` | truth and leak checks on every line before it is used |
| `abce.render` | turns items into plain lines (`plain`) or the raw grader form (`raw`) |
| `abce.context` | the located feedback for one attempt |
| `abce.teacher` | the teacher context: `teacher_context(record, siblings, chain, outcome)` |

## Setup

The engine reads task data from a checkout of
[verifiers-environments](https://github.com/carbonteq-ai/verifiers-environments) that has the task fixes (version 1.1;
see carbonteq-ai/verifiers-environments#7). It imports `automationbench_v1` from that checkout.

```bash
uv sync
export ABCE_ENVIRONMENT=../verifiers-environments/environments/automationbench_v1
export PYTHONPATH=$ABCE_ENVIRONMENT/src
```

| Variable | Default | What it is |
| --- | --- | --- |
| `ABCE_ENVIRONMENT` | `../verifiers-environments/environments/automationbench_v1` | the environment package; task contracts are read from its `src/automationbench_v1/contracts` |
| `ABCE_CACHE` | `.cache` | cache of parsed task manifests |

If `ABCE_ENVIRONMENT` points at the wrong checkout, every task contract silently comes back empty. Check the path
before reading any output.

## Usage

```python
from abce.teacher import teacher_context

# record: one episode record from the evaluation runner's episodes.jsonl.gz
# siblings: every record of the same task in the round (the attempt included)
# chain: the attempt's conversation from chains.jsonl.gz; outcome: its entry in outcomes.json
result = teacher_context(record, siblings, chain, outcome)
result.context      # {"first_message": ..., "notes": {turn: note}, "final_note": ...}, or None
result.last_turn    # the first turn the feedback locates; a failed attempt is trained up to this turn
result.reason       # why there is no context, e.g. "no_engine_feedback"
```

A solved attempt gets one outcome line. A failed attempt gets the feedback in design `S1FUHWNLGOJYIV`. Each
letter of a design name switches on one wording rule in `abce.teacher.text`; this design is the one that was trained.

## Checks

```bash
uv run ruff check src tests
uv run pytest
```

`tests/test_teacher_context.py` runs every round-1 attempt of two training tasks of the 50.6% run, 16 attempts in
all. It requires the same teacher context the run built for each.

## Known limits

These are recorded here, not fixed: this release stays identical to the engine that was trained with.
- **Lines say what is needed, not why.** The grader's checks carry no reasons. In readings of solved teacher
  retries, some solves did what the feedback named but skipped the step the task needed.
- **Found after the training run, fixed only in a later engine that was never trained:**
  - some requirements were silently dropped from the feedback;
  - wording was unreadable on apps the training never ran;
  - a narrow "this field has to stay" line could sit beside a line forbidding any change to the whole record.
- **Wording gap.** On `sales.zoom_crm_meeting`, a check that accepts any value combines with a hidden topic. The only
  true wording reverses the check's meaning, so such attempts get no feedback.
- **Coverage.** Every line passes the truth and leak checks. Usefulness was read by hand for a sample of check kinds
  only.
- **Size.** `abce.render.plain`, `abce.checks.truth`, `abce.fix` and `abce.teacher.text` are large modules. Splitting
  them without changing behaviour can be checked with `tests/test_teacher_context.py`.
