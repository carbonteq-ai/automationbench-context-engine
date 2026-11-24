"""Wording rules shared by the teacher text: value pointers, calmer phrasing, unread-value notes, tidying and the
closing frame, plus the engine options the values-shown view uses (`SHOWN_ENGINE`).
"""

from __future__ import annotations

import json
import re
from collections import Counter

PLACEHOLDER = "a value this attempt never observed"

OLD_HEAD = "The following is feedback from your unsuccessful earlier attempt:"

OLD_TAIL = "Correctly solve the original question."

NEW_HEAD = "Results of an earlier attempt at this same task (that attempt failed):"

NEW_TAIL = "Do the task correctly now."

NO_PLACE = "a value not found in anything this attempt read"

# the values-shown engine the pointers were measured with (weekend/engine-usefulness/probe_shown.py)
SHOWN_ENGINE = {
    "expected_values": True,
    "fix_hints": [
        "value_location",
        "wrong_value",
        "missed_write",
        "unrequested_change",
        "call_fix",
        "why_clause",
        "solution_note",
    ],
}

FIELD = re.compile(
    r'^In the initial world, "(?P<v>.+?)" is the (?P<field>.+?) of (?P<where>.+?)'
    r"(?: \((?P<tools>.+?) returns it\))?\.$"
)

NAMED = re.compile(
    r'^In the initial world, there is an? (?P<what>.+?) named "(?P<v>.+?)"(?: \(id ".*?"\))?'
    r"(?: \((?P<tools>.+?) returns it\))?\.$"
)


def pointer(line):
    """(kind, words) for one value-location line, or None."""
    m = FIELD.match(line)
    if m:
        tools = f" ({m['tools']} returns it)" if m["tools"] else ""
        return "field", f"the {m['field']} of {m['where']}{tools}"
    m = NAMED.match(line)
    if m:
        tools = f" ({m['tools']} lists them)" if m["tools"] else ""
        return "name", f"the name of a {m['what']}{tools}"
    return None


def values_in(hidden, shown):
    """The values that fill the placeholders of `hidden`, read from the same line rendered with values shown."""
    parts = hidden.split(PLACEHOLDER)
    pattern = "".join(
        re.escape(p) + ('"(.+?)"' if i < len(parts) - 1 else "") for i, p in enumerate(parts)
    )
    m = re.match(pattern.rstrip("\\."), shown)
    return list(m.groups()) if m else None


def split_feedback(first_message):
    """(task prompt with its trailing blank lines, the feedback lines block)."""
    if first_message.count(OLD_HEAD) != 1 or not first_message.rstrip().endswith(OLD_TAIL):
        raise ValueError("not an engine feedback message")
    prompt, rest = first_message.split(OLD_HEAD)
    return prompt, rest.rstrip()[: -len(OLD_TAIL)].strip()


def version_p(context, shown):
    """(context, counts). The counts say how each placeholder was replaced."""
    from collections import Counter

    prompt, lines = split_feedback(context["first_message"])
    hidden_facts = {f["text"]: f["id"] for f in context.get("facts") or [] if f.get("text")}
    shown_facts = {f["id"]: f["text"] for f in (shown or {}).get("facts") or [] if f.get("text")}
    hints = ((shown or {}).get("fix_hints") or {}).get("lines", [])
    places = {}
    for x in hints:
        if x.get("kind") == "value_location" and "value" in x:
            p = pointer(x["line"])
            if p:
                places.setdefault(x["value"], p)
    counts, new_lines = Counter(), []
    for ln in lines.split("\n"):
        if PLACEHOLDER not in ln:
            new_lines.append(ln)
            continue
        fid = hidden_facts.get(ln) or hidden_facts.get(ln.rstrip("."))
        vals = values_in(ln, shown_facts[fid]) if fid and fid in shown_facts else None
        pieces = ln.split(PLACEHOLDER)
        rebuilt = pieces[0]
        for i, piece in enumerate(pieces[1:]):
            v = vals[i] if vals and i < len(vals) else None
            p = places.get(v) if v is not None else None
            counts["placeholders"] += 1
            if p:
                counts["pointer_" + p[0]] += 1
                rebuilt += p[1] + piece
            else:
                counts["no_place" if vals else "unaligned"] += 1
                rebuilt += NO_PLACE + piece
        new_lines.append(rebuilt)
    for x in hints:
        if x.get("kind") == "missed_write" and x.get("line") and x["line"] not in new_lines:
            new_lines.append(x["line"])
            counts["missed_write"] += 1
    text = f"{prompt}{NEW_HEAD}\n\n" + "\n".join(new_lines) + f"\n\n{NEW_TAIL}"
    for v in (
        places
    ):  # a hidden value never reaches the teacher (a value the feedback already showed may stay)
        if v not in context["first_message"] and v in text:
            raise ValueError(f"version P would show the hidden value {v!r}")
    return {**context, "first_message": text, "notes": {}, "final_note": None}, dict(counts)


def _places(shown):
    places = {}
    for x in ((shown or {}).get("fix_hints") or {}).get("lines", []):
        if x.get("kind") == "value_location" and "value" in x:
            p = pointer(x["line"])
            if p:
                places.setdefault(x["value"], p)
    return places


def _core(text):
    """A fact or line text without its line prefix ("Not done: ", "At turn N, ") and final full stop: the form in which
    the engine repeats it inside notes."""
    return re.sub(r"^(Not done: |At turn \d+, )", "", text).rstrip(".")


def _fact_pairs(context, shown, places, counts):
    """[(hidden core, rebuilt core)] for every fact with a placeholder, longest first, rebuilt by P's rule (the value
    read from the same fact rendered with values shown -> its pointer, else NO_PLACE). A hidden core shared by facts
    whose rebuilt cores differ (two facts, same wording, different hidden values) is ambiguous inside a note: it is left
    out, so the note says NO_PLACE there rather than possibly the other fact's pointer (counted)."""
    shown_facts = {f["id"]: f["text"] for f in (shown or {}).get("facts") or [] if f.get("text")}
    by_core = {}
    for f in context.get("facts") or []:
        text = f.get("text") or ""
        if PLACEHOLDER not in text:
            continue
        vals = values_in(text, shown_facts[f["id"]]) if f["id"] in shown_facts else None
        pieces = text.split(PLACEHOLDER)
        rebuilt = pieces[0]
        for i, piece in enumerate(pieces[1:]):
            v = vals[i] if vals and i < len(vals) else None
            p = places.get(v) if v is not None else None
            rebuilt += (p[1] if p else NO_PLACE) + piece
        by_core.setdefault(_core(text), set()).add(_core(rebuilt))
    pairs = []
    for old, news in by_core.items():
        if len(news) == 1:
            pairs.append((old, next(iter(news))))
        else:
            counts["note_ambiguous_core"] += 1
    return sorted(pairs, key=lambda x: (-len(x[0]), x[0]))


def _fill(text, pairs, counts, tag):
    """A note (or final-note) text with each fact's core replaced by its rebuilt core; a placeholder left after that
    (no fact matched) becomes NO_PLACE (counted)."""
    if PLACEHOLDER not in text:
        return text
    n = text.count(PLACEHOLDER)
    for old, new in pairs:
        text = text.replace(old, new)
    left = text.count(PLACEHOLDER)
    counts[f"{tag}_placeholders"] += n
    counts[f"{tag}_by_fact"] += n - left
    counts[f"{tag}_no_fact"] += left
    return text.replace(PLACEHOLDER, NO_PLACE)


def version_v1(context, shown):
    """(context, counts). The old feedback's layout with P's pointers and missed-write lines (see above)."""
    from collections import Counter

    p, counts = version_p(context, shown)
    counts = Counter(counts)
    prompt, old_block = split_feedback(context["first_message"])
    _, p_block = p["first_message"].split(NEW_HEAD, 1)
    p_lines = p_block.rstrip()[: -len(NEW_TAIL)].strip().split("\n")
    old_lines = old_block.split("\n")
    # P matches a line to its fact by the line's text, so two facts with the same hidden wording but different values
    # get one fact's value on both lines; V1 says NO_PLACE on such a line instead (never a pointer to the other record)
    shown_facts = {f["id"]: f["text"] for f in (shown or {}).get("facts") or [] if f.get("text")}
    variants = {}
    for f in context.get("facts") or []:
        t = f.get("text") or ""
        if PLACEHOLDER in t:
            variants.setdefault(t.rstrip("."), set()).add(shown_facts.get(f["id"]))
    for i, ln in enumerate(old_lines):
        if PLACEHOLDER in ln and (
            len(variants.get(ln.rstrip("."), ())) > 1 or old_lines.count(ln) > 1
        ):
            p_lines[i] = ln.replace(PLACEHOLDER, NO_PLACE)
            counts["line_ambiguous"] += 1
    first = f"{prompt}{OLD_HEAD}\n\n" + "\n".join(p_lines) + f"\n\n{OLD_TAIL}"
    places = _places(shown)
    pairs = _fact_pairs(context, shown, places, counts)
    notes = {k: _fill(v, pairs, counts, "note") for k, v in (context.get("notes") or {}).items()}
    final = context.get("final_note")
    if (
        final
    ):  # the final note is metadata (its text is also a note); its own fields get the same rule
        final = dict(final)
        if final.get("line"):
            final["line"] = _fill(final["line"], pairs, Counter(), "final")
        if "texts" in final:
            final["texts"] = [_fill(t, pairs, Counter(), "final") for t in final["texts"] or []]

    def visible(f, n, fn):
        return f + "\n" + "\n".join(n.values()) + "\n" + json.dumps(fn or {}, ensure_ascii=False)

    before, after = (
        visible(context["first_message"], context.get("notes") or {}, context.get("final_note")),
        visible(first, notes, final),
    )
    for v in (
        places
    ):  # a hidden value never reaches the teacher (a value the feedback already showed may stay)
        if v not in before and v in after:
            raise ValueError(f"version V1 would show the hidden value {v!r}")
    if PLACEHOLDER in after:
        raise ValueError("version V1 left a placeholder")
    return {**context, "first_message": first, "notes": notes, "final_note": final}, dict(counts)


FORBID_CHECK = "check that record before acting on it."


def _first_error(message):
    """The first of the error messages a tool-error line or note states ("... (and 2 other ... errors at turn 4)")."""
    return re.sub(
        r" \(and \d+ other [a-z0-9_]+ errors? at turns? [0-9, ]+\)\.?$", "", message.strip()
    )


# Change 3, widened (Ali, 16 Nov 22:2x): for a tool error that names an object it could not find, one plain sentence
# on what to check. Each rule: (failing tools, error pattern, sentence, lookup tool and what it does). The lookup tool
# is named only when it is in the task's own tool list and is not the failing tool; what it does is said in the words
# of its own description (environment v1.1). Syntax and argument errors get no sentence: their own text says what to
# change. No sentence holds a value.
ERROR_CHECKS = (
    (
        r"^slack_",
        r"^Channel '.*' not found",
        "Use an exact channel name or ID.",
        ("slack_list_channels", "lists all channels"),
    ),
    (
        r"^google_sheets_",
        r"^Worksheet '.*' not found in spreadsheet",
        "Use an exact worksheet title or ID.",
        ("google_sheets_get_spreadsheet_by_id", "returns a spreadsheet with its worksheets"),
    ),
    (
        r"^google_sheets_",
        r"^Spreadsheet '.*' not found",
        "Use an exact spreadsheet title or ID.",
        ("google_drive_find_multiple_files", "finds spreadsheets by name"),
    ),
    (
        r"^google_sheets_update_row$",
        r"^Row .* not found in worksheet",
        "The row argument takes the row's own row_id, as the rows read from the sheet show it.",
        None,
    ),
    (
        r"^jira_",
        r"^jira_project_not_found",
        "Use an exact project id, key or name.",
        ("jira_project", "finds a project"),
    ),
    (
        r"^zoom_find_meeting$",
        r"^No matching meeting found",
        "Check the meeting first.",
        ("zoom_list_meetings", "lists the meetings"),
    ),
    (
        r"^docusign_",
        r"^Template not found",
        "Use an exact template name or ID.",
        ("docusign_find_template", "finds a template by name or ID"),
    ),
    (
        r"^google_ads_",
        r"^Campaign not found in this account",
        "The campaign is in another account. Check the mainAccountId value.",
        None,
    ),
    (
        r"^google_ads_",
        r"^Campaign not found$",
        "Use an exact campaign name or ID.",
        ("google_ads_find_campaign_by_name", "finds a campaign by name"),
    ),
    (
        r"^slack_send_direct_message$",
        r"^User '.*' not found",
        "slack_send_direct_message takes a user ID, username or email.",
        None,
    ),
)


def error_check(tool, error, task_tools):
    """The sentence for one tool error, or None."""
    first = _first_error(error).rstrip(".")
    for tools, pattern, sentence, lookup in ERROR_CHECKS:
        if re.search(tools, tool) and re.search(pattern, first):
            if lookup is not None and lookup[0] in task_tools and lookup[0] != tool:
                return f"{sentence} {lookup[0]} {lookup[1]}."
            return sentence
    return None


CALM_HEAD = "Notes on an earlier attempt at this task:"

CALM_TAIL = "Now do the task."

NOTE_OLD, NOTE_NEW = "Note on your next action: ", "Note: "

NO_PLACE_OLD, NO_PLACE_NEW = "a value not found in anything this attempt read", "an unseen value"

EXISTS = re.compile(
    r"^(?P<lead>Not done: |\(\d+\) |\d+\. )?(?P<x>.+?) exists, but (?:none with|has no) (?P<y>.+?)(?P<end>[.;]?)$"
)

REQUIRED = re.compile(
    r"^(?P<lead>Not done: |\(\d+\) |\d+\. )?no (?P<x>.+?) exists; one is required(?P<end>[.;]?)$"
)


def calm_item(text):
    """One line or list item."""
    m = REQUIRED.match(text)
    if m:
        lead = "Still needed: " if m["lead"] == "Not done: " else (m["lead"] or "")
        return f"{lead}{m['x']}{m['end'] or '.'}"
    m = EXISTS.match(text)
    if m:
        lead = "Still needed: " if m["lead"] == "Not done: " else (m["lead"] or "")
        return f"{lead}{m['x']} with {m['y']} (still missing){m['end'] or '.'}"
    if text.startswith("Not done: "):
        return "Still needed: " + text[len("Not done: ") :]
    return text


def calm_text(text):
    text = text.replace(NO_PLACE_OLD, NO_PLACE_NEW).replace(" failed: ", " returned an error: ")
    # notes join several statements with a space only; each gets its own sentence
    text = re.sub(
        r"(?<=[^.\s;:]) (?=[a-z0-9_]+ returned an error: |Still needed: |No call wrote )",
        ". ",
        text,
    )
    text = re.sub(r"; this task forbids ", ". This task forbids ", text)
    text = re.sub(r"; their (\S+) had to stay ", r". Its \1 had to stay ", text)
    return text


def calm_note(note):
    body = note[len(NOTE_OLD) :] if note.startswith(NOTE_OLD) else note
    body = body.replace("Still not done at this point: ", "Still needed: ")
    parts = re.split(r"(\(\d+\) |\n\d+\. )", body)
    out = parts[0]
    for i in range(1, len(parts), 2):
        item = parts[i + 1]
        tail = item.rstrip()
        sep = tail[-1] if tail[-1:] in ";." else ""
        core = tail[:-1] if sep else tail
        r = REQUIRED.match(core)
        m = EXISTS.match(core)
        core = r["x"] if r else (f"{m['x']} with {m['y']} (still missing)" if m else core)
        out += parts[i] + core + ("." if sep else "") + item[len(tail) :]
    out = calm_text(out)
    seen, kept = set(), []
    for sentence in re.split(r"(?<=\.)\s+", out):
        if sentence not in seen:
            seen.add(sentence)
            kept.append(sentence)
    return NOTE_NEW + " ".join(kept)


# Short, single statements (16-17 Nov night, after the B5 audit of the first calm version; meaning kept, calm_check):
#   R1 2+ record sentences "...; check that record before acting on it." in one text: each record sentence kept, the
#      instruction said once at the end ("For each of these, check that record before acting on it.")
#   R2 the engine's call-fix line "The value V given at turn N is the D in the initial world, not a K; T accepted ..."
#      -> "The value V given at turn N is not a K. It is the D in the initial world. T accepted ..."
#   R3 'with cell "F" = not "A", with cell "F" = not "B"' -> 'with cell "F" = not any of "A", "B"'
#   R4 'with X containing "A" or with X containing "B"' -> 'with X containing "A" or "B"'
#   R5 a full stop after a tool's error text that runs into an appended sentence (punctuation only)
#   R6 '... worksheet "W" with cell ...' -> '... worksheet "W". Its cell ...', only when it adds no repeated sentence
FORBID_TAIL = "; " + FORBID_CHECK

FORBID_ONCE = "For each of these, " + FORBID_CHECK

CALL_FIX_LINE = re.compile(
    r'The value (?P<v>"[^"]*") given at turn (?P<t>\d+) is the (?P<desc>.+?) in the initial world, '
    r"not (?P<k>an? [a-z][a-z ]*?)(?P<rest>(?:; [A-Za-z0-9_]+ accepted .*?)?)\."
)

NOT_RUN = re.compile(
    r'(?P<lead>with|Its) cell (?P<f>"[^"]*") = not (?P<a>"[^"]*")(?P<more>(?:, with cell (?P=f) = not "[^"]*")+)'
)

OR_RUN = re.compile(
    r'(?P<pre>with (?P<what>[a-z ]+?) containing )(?P<a>"[^"]*")(?P<more>(?: or with (?P=what) containing "[^"]*")+)'
)

APPENDED_START = re.compile(
    r'(?<=[^.\s]) (?=The value "|The error at turn \d|Use an exact |The row argument |The campaign is |'
    r"Check the meeting |slack_send_direct_message takes )"
)

CELLS = re.compile(r'(worksheet "[^"]*") with cell (?=")')


def _short(text):
    text = CALL_FIX_LINE.sub(
        lambda m: (
            f"The value {m['v']} given at turn {m['t']} is not {m['k']}. It is the {m['desc']} "
            f"in the initial world." + (f" {m['rest'][2:]}." if m["rest"] else "")
        ),
        text,
    )
    text = APPENDED_START.sub(". ", text)

    def repeats(s):
        parts = [
            p[len(NOTE_NEW) :] if p.startswith(NOTE_NEW) else p
            for p in re.split(r"(?<=[.;])\s+|\n+", s)
            if p.strip()
        ]
        return len(parts) - len(set(parts))

    split = CELLS.sub(r"\1. Its cell ", text)
    if repeats(split) <= repeats(text):
        text = split
    text = NOT_RUN.sub(
        lambda m: (
            f"{m['lead']} cell {m['f']} = not any of "
            + ", ".join([m["a"]] + re.findall(r'= not ("[^"]*")', m["more"]))
        ),
        text,
    )
    text = OR_RUN.sub(
        lambda m: m["pre"] + " or ".join([m["a"]] + re.findall(r'containing ("[^"]*")', m["more"])),
        text,
    )
    if text.count(FORBID_TAIL) >= 2:
        text = text.replace(FORBID_TAIL, ".") + " " + FORBID_ONCE
    return text


def version_calm(ctx):
    first = ctx["first_message"]
    prompt, rest = first.split(OLD_HEAD, 1)
    lines = rest.rstrip()[: -len(OLD_TAIL)].strip().split("\n")
    new_lines = []
    for ln in lines:  # the engine sometimes states a line twice; once is enough
        calm = _short(calm_text(calm_item(ln)))
        if calm not in new_lines:
            new_lines.append(calm)
    out_first = f"{prompt}{CALM_HEAD}\n\n" + "\n".join(new_lines) + f"\n\n{CALM_TAIL}"
    notes = {k: _short(calm_note(v)) for k, v in (ctx.get("notes") or {}).items()}
    return {**ctx, "first_message": out_first, "notes": notes}


NOT_READ = "a value you have not read"

UNREAD_PAST = re.compile(
    r"^\s*(?:Note: )?(?:At turns? [0-9, and]+, )?[a-z0-9]+_[a-z0-9_]+ "
    r"(?:produced|added|updated|created|changed|sent|posted|wrote|moved|deleted|archived)\b"
)

_U = NO_PLACE_NEW

UNREAD_RULES = [  # (pattern on the masked item, kind); group f names the field, the match is the removed clause
    (
        re.compile(
            r",? with (?P<f>[a-z][a-z ]{0,30}?) (?P<op>containing )?(?P<vals>(?:\x00|" + _U + r")"
            r"(?:(?:, and |, | and )(?:\x00|" + _U + r"))+)"
        ),
        "list",
    ),
    (re.compile(r",? with (?P<f>[a-z][a-z ]{0,30}?) matching the pattern " + _U), "match"),
    (re.compile(r",? with additional checks (?P<f>[a-z][a-z ]{0,30}?) " + _U), "is"),
    (re.compile(r" on record " + _U), " on a record you have not read"),
    (re.compile(r"(?<=message) " + _U + r"(?= is required)"), " you have not read"),
    (re.compile(r" (?P<f>phone) " + _U), "is"),
    (re.compile(r" for contact " + _U), " for a contact you have not read"),
    (re.compile(r"(?<=\x00) (?P<f>phone|email|title|name|date|amount|status) " + _U), "is"),
    (
        re.compile(r"(?:,? with |; |, | and )(?P<f>[a-z][a-z ]{0,30}?) not containing " + _U),
        "leave_out",
    ),
    (re.compile(r",? with (?P<f>[a-z][a-z ]{0,30}?) containing " + _U), "contain"),
    (
        re.compile(r"(?<! not) containing " + _U),
        "contain",
    ),  # never after "not": that is a must-leave-out
    (re.compile(r" to " + _U), "to"),
    (re.compile(r"(?:; |, | and )(?P<f>[A-Za-z][A-Za-z0-9 /_-]{0,30}?) = " + _U), "is"),
    (re.compile(r"(?:; |, | and )(?P<f>[a-z][a-z ]{0,20}?) " + _U), "is"),
    (
        re.compile(
            r" with cells (?:containing )?(?P<f>[A-Za-z][A-Za-z0-9 /_-]{0,30}?) = "
            + _U
            + r"(?P<next>; )?"
        ),
        "cell",
    ),
    (re.compile(r",? with (?P<f>[a-z][a-z ]{0,20}?) " + _U), "is"),
    (
        re.compile(
            r" in (?P<f>list|spreadsheet|calendar|channel|project|board|folder|account|worksheet) "
            + _U
        ),
        "in",
    ),
    (re.compile(r" label " + _U), " with a label you have not read"),
    (re.compile(r" (?P<f>[a-z][a-z ]{0,20}?) " + _U + r"(?P<next>; )"), "is_first"),
]


def _unread_item(item):
    """One masked item (a line, a note sentence or a list item) -> the item without stand-ins."""
    if _U not in item:
        return item
    past = bool(UNREAD_PAST.match(item))
    needs = []
    for _ in range(12):
        if _U not in item:
            break
        for pat, kind in UNREAD_RULES:
            m = pat.search(item)
            if not m or (kind == "list" and _U not in m["vals"]):
                continue
            f = re.sub(r"^with ", "", m.groupdict().get("f") or "")
            new = ""
            if kind == "list":
                vals = re.split(r", and |, | and ", m["vals"])
                shown = [v for v in vals if v == "\x00"]
                if shown:
                    new = f" with {f} {m['op'] or ''}" + (
                        shown[0] if len(shown) == 1 else ", ".join(shown[:-1]) + " and " + shown[-1]
                    )
                n = len(vals) - len(shown)
                needs.append(
                    f"whose {f} {'also ' if shown else ''}{'held' if past else 'needs'} "
                    f"{'a value' if n == 1 else 'values'} you have not read"
                )
            elif kind == "contain":
                verb = "held" if past else "need" if f == "cells" else "needs"  # "whose cells need"
                needs.append(f"whose {f or 'text'} {verb} {NOT_READ}")
            elif kind == "leave_out":  # "with body not containing X": the body must leave X out
                needs.append(f"whose {f} {'had' if past else 'has'} to leave out {NOT_READ}")
            elif kind == "match":
                needs.append(
                    f"whose {f} {'had' if past else 'has'} to match a pattern you have not read"
                )
            elif kind in ("is", "to", "cell", "is_first"):
                field = (
                    "recipient"
                    if kind == "to"
                    else " ".join(f.split()) + (" cell" if kind == "cell" else "")
                )
                needs.append(f"whose {field} {'was' if past else 'is'} one you have not read")
                new = (
                    " with cells "
                    if kind == "cell" and m["next"]
                    else (" " if kind == "is_first" else "")
                )
            elif kind == "in":
                new = f" in a {f} unknown to you"  # these lines often say "created no ...": no second negation
            else:
                new = kind  # a fixed in-place phrase
            item = item[: m.start()] + new + item[m.end() :]
            break
        else:
            break
    item = item.replace(
        _U, NOT_READ
    )  # identity slots no rule names a field for ("a docusign envelope for X")
    if needs:
        tail = re.search(r"(?: for \x00)?(?: \(still missing\))?[.;]?\s*$", item)
        item = (
            item[: tail.start()] + ", " + " and ".join(dict.fromkeys(needs)) + item[tail.start() :]
        )
    return item


def say_unread(text):
    quoted = re.findall(r'"[^"]*"', text)
    pieces = re.split(r"((?<=\.) |\n|\(\d+\) |(?<=\n)\d+\. )", re.sub(r'"[^"]*"', "\x00", text))
    out = "".join(_unread_item(p) for p in pieces).split("\x00")
    return out[0] + "".join(q + piece for q, piece in zip(quoted, out[1:]))


def version_unread(ctx):
    """V4 = version_calm then say_unread on the first message and every note."""
    return {
        **ctx,
        "first_message": say_unread(ctx["first_message"]),
        "notes": {k: say_unread(v) for k, v in (ctx.get("notes") or {}).items()},
    }


JARGON = {"from": "sender", "to": "recipient"}

QUALIFIER = re.compile(r' for "[^"]*"(?=(?: \(still missing\))?\.$)')


def _outside_quotes(pattern, repl, text):
    """re.sub applied only where the match lies outside quoted values (found on a same-length mask)."""
    masked = re.sub(r'"[^"]*"', lambda m: '"' + "\x00" * (len(m[0]) - 2) + '"', text)
    out, last = [], 0
    for m in re.finditer(pattern, masked):
        out.append(text[last : m.start()])
        out.append(repl(m) if callable(repl) else repl)
        last = m.end()
    return "".join(out) + text[last:]


def _statements(text):
    """(separator, statement) pairs: first-message lines and a note's sentences, split outside quoted values."""
    masked = re.sub(r'"[^"]*"', lambda m: '"' + "\x00" * (len(m[0]) - 2) + '"', text)
    pairs, last, sep = [], 0, ""
    for m in re.finditer(
        r"\n|(?<=\.) (?=[A-Z(]|[a-z0-9]+_[a-z0-9_]+ )", masked
    ):  # notes start sentences with tool names
        pairs.append((sep, text[last : m.start()]))
        sep, last = m[0], m.end()
    pairs.append((sep, text[last:]))
    return pairs


def _plain_words(text):
    text = _outside_quotes(r",\. ", ", ", text)
    return _outside_quotes(
        r"\bthe (from|to) of (?=gmail message \")", lambda m: f"the {JARGON[m[1]]} of ", text
    )


DEPENDENT = re.compile(
    r"^(This task |These |Its |It |For each of these|The value |The error |The row |The campaign |"
    r"Use an exact |Check the |[a-z0-9]+_[a-z0-9_]+ (?:lists|returns|finds|takes) )"
)


def _facts(text):
    """Statements grouped into facts: a statement plus the statements after it that lean on it ("This task forbids it",
    "Its cell ...", "It is the ..."); a line break always starts a new fact. Returns (separator, [statements], [seps])."""
    facts = []
    for sep, s in _statements(text):
        if facts and sep != "\n" and DEPENDENT.match(s):
            facts[-1][1].append(s)
            facts[-1][2].append(sep)
        else:
            facts.append((sep, [s], [""]))
    return facts


def _fact_text(statements, seps):
    return "".join(sp + s for sp, s in zip(seps, statements))


def tidy(text):
    """Merge whole facts said more than once (also when they differ only in a trailing 'for "X"' qualifier): the first
    is kept without its qualifiers, later ones are dropped with all their dependent statements."""
    facts = _facts(_plain_words(text))
    plain = [_fact_text([QUALIFIER.sub("", s) for s in st], sp) for _, st, sp in facts]
    key = [
        pl[len(NOTE_NEW) :] if pl.startswith(NOTE_NEW) else pl for pl in plain
    ]  # a note's first fact starts "Note: "
    count = Counter(key)
    out, said, newline = [], set(), False
    for (sep, st, sp), pl, k in zip(facts, plain, key):
        body = _fact_text(st, sp)
        if count[k] > 1 and body.strip():
            if k in said:
                newline = newline or sep == "\n"  # a dropped line start passes its line break on
                continue
            said.add(k)
            body = pl
        out.append(("\n" if newline and out else sep if out else "") + body)
        newline = False
    return "".join(out)


def version_tidy(ctx):
    return {
        **ctx,
        "first_message": tidy(ctx["first_message"]),
        "notes": {k: tidy(v) for k, v in (ctx.get("notes") or {}).items()},
    }


def version_frame(ctx):
    first = ctx["first_message"]
    if first.count(CALM_HEAD) != 1 or not first.rstrip().endswith(CALM_TAIL):
        raise ValueError("not a calm (V4) first message")
    body = first.rstrip()
    first = body[: -len(CALM_TAIL)].replace(CALM_HEAD, OLD_HEAD) + OLD_TAIL + first[len(body) :]
    notes = {
        k: (NOTE_OLD + v[len(NOTE_NEW) :] if v.startswith(NOTE_NEW) else v)
        for k, v in (ctx.get("notes") or {}).items()
    }
    return {**ctx, "first_message": first, "notes": notes}


def w4_guard(hidden, shown):
    """True when every placeholder of the engine's hidden context is aligned to its value (the same fact rendered with
    values shown) and the shown context has no value-location line for that value: the engine found it in no record of
    the initial world, so "not written in any record" is the engine's own finding."""
    shown_facts = {f["id"]: f["text"] for f in (shown or {}).get("facts") or [] if f.get("text")}
    located = {
        x.get("value")
        for x in ((shown or {}).get("fix_hints") or {}).get("lines", [])
        if x.get("kind") == "value_location"
    }
    seen = 0
    for f in hidden.get("facts") or []:
        text = f.get("text") or ""
        if PLACEHOLDER not in text:
            continue
        vals = values_in(text, shown_facts[f["id"]]) if f.get("id") in shown_facts else None
        if not vals or len(vals) != text.count(PLACEHOLDER):
            return False
        if any(v in located for v in vals):
            return False
        seen += 1
    return seen > 0
