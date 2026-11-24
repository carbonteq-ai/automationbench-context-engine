"""`EngineContext`: runs the context engine on one attempt and its sibling attempts of the same task, and returns
the located feedback (first message, notes per turn, final note) with expected values hidden or shown.
"""

from collections import Counter, defaultdict

# Correction r0: settings of the corrected engine's `build_context` that the runner may pass (EngineContext `options`)
ENGINE_OPTIONS = ("line_budget", "char_budget", "not_done_source", "final_note")

NOT_DONE_SOURCES = ("all", "benchmark")


def check_engine_options(options):
    """The given engine options, checked (unknown names and impossible values refused before any engine call)."""
    unknown = sorted(set(options) - set(ENGINE_OPTIONS))
    if unknown:
        raise ValueError(f"unknown engine options {unknown}")
    for name in ("line_budget", "char_budget"):
        if name in options:
            value = options[name]
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(
                    f"{name} must be a positive integer or None (no limit), not {value!r}"
                )
    if "not_done_source" in options and options["not_done_source"] not in NOT_DONE_SOURCES:
        raise ValueError(
            f"not_done_source must be one of {NOT_DONE_SOURCES}, not {options['not_done_source']!r}"
        )
    if "final_note" in options and not isinstance(options["final_note"], bool):
        raise ValueError(f"final_note must be True or False, not {options['final_note']!r}")
    return dict(options)


def load_engine():
    """(feedback_for_action, build_record, Episode) from the engine."""
    from abce.context import feedback_for_action
    from abce.record import build_record
    from abce.record.episode import Episode

    return feedback_for_action, build_record, Episode


def load_context_engine():
    """(build_context, build_record, Episode, truth check_episode, leak finder) from the engine."""
    load_engine()
    from abce.checks.leak import leaks
    from abce.checks.truth import check_episode
    from abce.context import build_context
    from abce.record import build_record
    from abce.record.episode import Episode

    return build_context, build_record, Episode, check_episode, leaks


WITHHOLD_ROUNDS = 3


class EngineContext:
    """`build_context` replacement for `train_arm` (D44 per-tool-call scheme): the engine's plain context for one
    episode (first message + per-turn notes, expected values hidden; arm B adds the engine's verified sibling
    solution). Every context is checked by the engine's truth test (and, with values hidden, its leak test) before
    use; a context that fails raises ValueError, which `train_arm` records as an exclusion (no stop). Returns None when the engine has nothing to say.
    Signature matches `sdpo_ab_context.build_context(record, siblings, arm, placement, contract, variant=...)`.
    Recipe V: `expected_values=True` shows the expected values (the truth check still runs on every context; the
    leak test is skipped, since showing the values is the point); `fix_hints` is a list of the engine's fix-hint
    kind names (abce.fix.FIX_HINT_KINDS; the study approved only "call_fix") passed to the engine as is, and needs
    values shown; True (all kinds) is refused, so every kind is named. The defaults (hidden, no hints) are recipe U's
    behaviour.
    Correction r0: `options` are further `build_context` settings of the corrected engine (ENGINE_OPTIONS:
    line_budget and char_budget, a positive int or None for no limit; not_done_source "all" | "benchmark";
    final_note True | False), passed only when given, so a call without them is exactly the earlier one."""

    def __init__(
        self,
        expected_values=False,
        pool=False,
        fix_hints=False,
        options=None,
        withhold_untrue=False,
    ):
        if fix_hints is True or isinstance(fix_hints, str):
            raise ValueError(f"fix_hints must be a list of kind names, not {fix_hints!r}")
        fix_hints = list(fix_hints or [])
        if fix_hints and not expected_values:
            raise ValueError("fix hints need the expected values shown (--engine-values shown)")
        self.expected_values = expected_values
        self.fix_hints = fix_hints
        # sdpo-weekend (plan v4, 14 Nov): with withhold_untrue, an item whose line the truth checker (or, values hidden, the
        # leak test) rejects is removed from a copy of the Record, the context rebuilt and re-checked against that same
        # Record (at most WITHHOLD_ROUNDS times); the untrue line is never shown, the rest of the context is kept, and each
        # withheld item is listed in coverage["withheld"] with its reason. Off (the default): a rejection refuses the whole
        # context, exactly as before.
        self.withhold_untrue = withhold_untrue
        # passed only when on, so the default call to the engine is exactly recipe U's
        self.hint_option = {"fix_hints": list(fix_hints)} if fix_hints else {}
        self.options = check_engine_options(options or {})
        self.hint_option = {**self.hint_option, **self.options}
        self.build_context, self.build_record, self.Episode, self.check, self.leaks = (
            load_context_engine()
        )
        if fix_hints:
            load_engine()
            from abce.fix import selected_kinds

            selected_kinds(fix_hints)  # unknown kind names refused before any engine call
        self.stats = Counter()
        # D47 recipe P: successful attempts of earlier batches (and given folders), per task. A failed attempt whose
        # own batch offers no verified solution is offered these as further siblings (arm B only).
        self.pool = defaultdict(list) if pool else None

    def add_to_pool(self, records):
        if self.pool is None:
            return
        for record in records:
            episode = self.Episode(record=record)
            if episode.success and all(r["id"] != record["id"] for r in self.pool[episode.task]):
                self.pool[episode.task].append(record)
                self.stats["pool_records"] += 1

    def __call__(self, record, siblings, arm, placement, contract=None, variant=None):
        if placement != "both":
            raise ValueError(f"engine contexts are built with placement both, not {placement!r}")
        episode = self.Episode(record=record)
        others = [self.Episode(record=r) for r in siblings if r["id"] != record["id"]]
        engine_record = self.build_record(episode, probe=True, manifest=True)
        self.stats["records"] += 1
        self.stats["manifest_items"] += sum(
            getattr(i, "origin", None) == "manifest" for i in engine_record.items
        )
        for note in engine_record.notes:
            self.stats["record_note:" + " ".join(str(note).split()[:4])] += 1
        context = self.build_context(
            engine_record,
            others,
            style="plain",
            placement="both",
            expected_values=self.expected_values,
            episode=episode,
            solution="sibling" if arm == "B" else "none",
            **self.hint_option,
        )
        if (
            self.pool is not None
            and arm == "B"
            and not episode.success
            and not context.get("solution_episode")
        ):
            known = {r["id"] for r in siblings} | {record["id"]}
            extra = [
                self.Episode(record=r)
                for r in self.pool.get(episode.task, [])
                if r["id"] not in known
            ]
            if extra:
                pooled = self.build_context(
                    engine_record,
                    others + extra,
                    style="plain",
                    placement="both",
                    expected_values=self.expected_values,
                    episode=episode,
                    solution="sibling",
                    **self.hint_option,
                )
                if pooled.get("solution_episode"):
                    context, others = pooled, others + extra
                    self.stats["solution_from_pool"] += 1
        for entry in (context.get("coverage") or {}).get("suppressed") or []:
            self.stats["suppressed:" + str(entry.get("reason"))] += 1
        if context.get("skip_reason"):
            self.stats["skipped:" + context["skip_reason"]] += 1
            return None
        withheld = []
        for _round in range(WITHHOLD_ROUNDS + 1):
            violations, checked = self.check(
                episode, engine_record, context, others, expected_values=self.expected_values
            )
            self.stats["lines_checked"] += checked
            found = (
                []
                if (violations or self.expected_values)
                else self.leaks(engine_record, context, episode_messages=episode.messages)
            )
            if not violations and not found:
                break
            bad = {v["item"] for v in violations if v.get("item")} | {
                f["item"] for f in found if f.get("item")
            }
            if (
                not self.withhold_untrue
                or _round == WITHHOLD_ROUNDS
                or not bad
                or (violations and len(bad) < len({v.get("item") for v in violations}))
            ):  # a violation without an item cannot be withheld
                if violations:
                    self.stats["excluded_truth"] += 1
                    raise ValueError("truth: " + "; ".join(sorted({v["kind"] for v in violations})))
                self.stats["excluded_leak"] += 1
                raise ValueError("leak: " + "; ".join(sorted({str(f.get("key")) for f in found})))
            reasons = {
                **{v["item"]: "untrue: " + v["kind"] for v in violations if v.get("item")},
                **{f["item"]: "leak: " + str(f.get("key")) for f in found if f.get("item")},
            }
            withheld += [{"id": i, "reason": reasons[i]} for i in sorted(bad)]
            engine_record = engine_record.model_copy(
                update={"items": tuple(i for i in engine_record.items if i.id not in bad)}
            )
            context = self.build_context(
                engine_record,
                others,
                style="plain",
                placement="both",
                expected_values=self.expected_values,
                episode=episode,
                solution="sibling" if arm == "B" else "none",
                **self.hint_option,
            )
            if context.get("skip_reason"):
                self.stats["skipped:" + context["skip_reason"]] += 1
                return None
        if withheld:
            self.stats["contexts_with_withheld"] += 1
            self.stats["withheld_items"] += len(withheld)
            context.setdefault("coverage", {})["withheld"] = withheld
        self.stats["contexts"] += 1
        self.stats["with_solution"] += bool(context.get("solution_episode"))
        context["episode_id"] = record["id"]
        return context
