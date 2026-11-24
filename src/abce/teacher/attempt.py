"""The engine views a teacher context is built from: the hidden-value engine (`make_engine`), where its
feedback first points (`point_turn`), and the attempt's conversation path (`assistant_path`).
"""

from __future__ import annotations

ENGINE_OPTIONS = {
    "line_budget": None,
    "char_budget": None,
    "not_done_source": "benchmark",
    "final_note": True,
}

ARMS = {
    "T-A2": {"expected_values": False, "pool": False, "fix_hints": [], "arm": "A"},
    "T-A1": {"expected_values": True, "pool": False, "fix_hints": ["call_fix"], "arm": "B"},
    # 14 Nov (added measurement arm, not pre-registered): T-A2's context plus QUIET; the smoke run showed both
    # pre-registered arms narrating the feedback in their reasoning ("as per the feedback from the earlier
    # attempt"), which a student would learn to claim without having it
    "T-A2q": {"expected_values": False, "pool": False, "fix_hints": [], "arm": "A", "base": "T-A2"},
    # 15 Nov (engine usefulness test, north star section 2 B; approved): the no-feedback control. Same points and
    # item digests as T-A2 (its engine context is built only for the digest check), but the teacher continues from
    # the student's own tokens with no feedback at all, so its fixes measure what resampling alone gives.
    "T-0": {
        "expected_values": False,
        "pool": False,
        "fix_hints": [],
        "arm": "A",
        "base": "T-A2",
        "no_feedback": True,
    },
    # 15 Nov (Ali's engine requirements; Ali: move ahead on my judgement): feedback versions N and P, rewritten on
    # the CPU from T-A2's own context (weekend/engine-usefulness/rewrite.py) and read here by T-A2's digest. N = the
    # same lines once in the first message under a plain heading, no notes in the conversation; P = N with the
    # "never observed" placeholders replaced by verified locations (no values) and the missed-write lines added.
    "T-N": {
        "expected_values": False,
        "pool": False,
        "fix_hints": [],
        "arm": "A",
        "base": "T-A2",
        "rewrite": "N",
    },
    "T-P": {
        "expected_values": False,
        "pool": False,
        "fix_hints": [],
        "arm": "A",
        "base": "T-A2",
        "rewrite": "P",
    },
    # 16 Nov (overnight, Ali: improve the work on my judgement; measurement only): version Q = P + one guidance
    # line for the "this task forbids" lines (feedback_versions.q_text), aimed at requirements 4 and 5
    "T-Q": {
        "expected_values": False,
        "pool": False,
        "fix_hints": [],
        "arm": "A",
        "base": "T-A2",
        "rewrite": "Q",
    },
    # 16 Nov (plan approved by Ali, Step 3 change 1): version V1 = the old feedback's layout (lines in the first
    # message and again as notes at the mistake turns, final note) with P's pointers and missed-write lines
    # (feedback_versions.version_v1); its rewrite is a whole context (first message, notes, final note)
    "T-V1": {
        "expected_values": False,
        "pool": False,
        "fix_hints": [],
        "arm": "A",
        "base": "T-A2",
        "rewrite": "V1",
    },
    # 16 Nov (Step 3 change 2): V2 = V1 + for each failed prohibition whose target is one record of the initial
    # world, where that record is and to check it first (feedback_versions.version_v2; forbid_locate.py)
    "T-V2": {
        "expected_values": False,
        "pool": False,
        "fix_hints": [],
        "arm": "A",
        "base": "T-A2",
        "rewrite": "V2",
    },
    # 16-17 Nov (Step 3 change 3, widened by Ali): V3 = base + the engine's call-fix lines + one what-to-check
    # sentence per not-found tool error (feedback_versions.add_call_fixes / add_error_checks); on V1 or V2
    "T-V3-1": {
        "expected_values": False,
        "pool": False,
        "fix_hints": [],
        "arm": "A",
        "base": "T-A2",
        "rewrite": "V3-1",
    },
    "T-V3-2": {
        "expected_values": False,
        "pool": False,
        "fix_hints": [],
        "arm": "A",
        "base": "T-A2",
        "rewrite": "V3-2",
    },
    # Step 3 change 4: V4 = calm, short wording (feedback_versions.version_calm) on V1, V2, V3-1 or V3-2
    "T-V4-1": {
        "expected_values": False,
        "pool": False,
        "fix_hints": [],
        "arm": "A",
        "base": "T-A2",
        "rewrite": "V4-1",
    },
    "T-V4-2": {
        "expected_values": False,
        "pool": False,
        "fix_hints": [],
        "arm": "A",
        "base": "T-A2",
        "rewrite": "V4-2",
    },
    "T-V4-31": {
        "expected_values": False,
        "pool": False,
        "fix_hints": [],
        "arm": "A",
        "base": "T-A2",
        "rewrite": "V4-31",
    },
    "T-V4-32": {
        "expected_values": False,
        "pool": False,
        "fix_hints": [],
        "arm": "A",
        "base": "T-A2",
        "rewrite": "V4-32",
    },
    # 17 Nov (V4 diagnosis): V5 = V4-1's lines with V1's heading, closing line and note prefix (version_frame)
    "T-V5": {
        "expected_values": False,
        "pool": False,
        "fix_hints": [],
        "arm": "A",
        "base": "T-A2",
        "rewrite": "V5",
    },
}

QUIET = (
    "When you work on this task now, reason only from the task request and the data your tools return. Do not "
    "mention these notes, any feedback, any check, or any earlier attempt in your reasoning, messages or replies."
)


def make_engine(arm, engine_class=None):
    """The teacher's engine for one arm. T-A1's solution comes from the attempt's own successful siblings (arm B reads
    the siblings given to the call), never from another model."""
    if engine_class is None:
        from abce.teacher.engine import EngineContext as engine_class
    spec = ARMS[arm]
    return engine_class(
        expected_values=spec["expected_values"],
        pool=spec["pool"],
        fix_hints=list(spec["fix_hints"]),
        options=dict(ENGINE_OPTIONS),
        withhold_untrue=True,
    )


def point_turn(context):
    """Earliest turn (1-based action index) the feedback speaks about, or None when it says nothing located."""
    turns = [int(t) for t in (context.get("notes") or {})]
    final = (context.get("final_note") or {}).get("turn")
    if final is not None:
        turns.append(int(final))
    return min(turns) if turns else None


def path_nodes(nodes: list[dict]) -> list[dict]:
    """Nodes on the path from the root to the deepest leaf (the conversation the harness ran)."""
    depth = {}
    for i, n in enumerate(nodes):
        depth[i] = 0 if n["parent"] is None else depth[n["parent"]] + 1
    leaf = max(depth, key=lambda i: (depth[i], i))
    out = []
    while leaf is not None:
        out.append(nodes[leaf])
        leaf = nodes[leaf]["parent"]
    return out[::-1]


def assistant_path(record):
    (trace,) = record["traces"]
    return [n["message"] for n in path_nodes(trace["nodes"])]
