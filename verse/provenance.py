"""File-write detection and content provenance for recorded agent traces.

Typed edit tools. A tool call may produce file content and/or consume (match against) existing
file content. An edit that consumes a string S depends on the latest earlier event that produced
text containing S. If that producer is dropped from a replay prefix, S is absent on a clean
replay and the edit silently does nothing. `build_provenance` records these content-dependency
edges and `causal_closure` adds the producers back to a keep-set. To support a new typed harness,
add one line per file-mutating tool to CONTENT_SPEC; the typed arguments already carry the
produced and consumed text, so no command parsing is needed.

Shell tools. Shell-only harnesses (e.g. mini-SWE-agent, Terminus) have a single tool whose
argument is a raw command string, so a write must be recognized from shell syntax (redirections,
in-place editors, heredocs, cp/mv, ...). `shell_writes_file` and `shell_write_tier` do this for
any harness whose tool arguments carry a command string. The written content is not recovered,
so shell events get no content-dependency edges; they are only marked as writes.
"""
from __future__ import annotations

import re

PROVENANCE_VERSION = "0.3"

# (tool_name, command) -> (produces_field, consumes_field). A command of None matches any
# command (for harnesses where the tool name is the operation). This is the only
# harness-specific knowledge: one line per file-mutating tool.
CONTENT_SPEC = {
    ("str_replace_editor", "create"):      ("file_text", None),
    ("str_replace_editor", "str_replace"): ("new_str", "old_str"),
    ("str_replace_editor", "insert"):      ("new_str", None),
}

# Shell tools whose argument is a raw command string, read from the "command" field (or
# "cmd" / "script", see _shell_command_of). Add a name here for a new shell harness.
SHELL_TOOLS = ("execute_bash", "bash", "shell", "run", "exec")

# Patterns for "this shell command writes a file". Conservative: each pattern is a write that
# changes repo state on replay; read-only commands (cat, ls, grep, pytest, python -c without a
# write) do not match. Redirects to /dev/* and fd duplications (`2>/dev/null`, `>&2`) are
# excluded, and `install` matches only as a command head (coreutils install, not `pip install`),
# so these common commands do not crowd the real edits out of a bounded leave-one-out budget.
_SHELL_WRITE_PATTERNS = [
    r"(?<![0-9])>>?\s*(?!/dev/|&)[^\s|&;>]+",  # > file / >> file (not 2>/dev/null or >&2)
    r"\btee\b\s+(-a\s+)?[^\s|&;]+",            # tee file  /  tee -a file
    r"\bsed\b[^\n]*\s-i\b",                    # sed -i ...            (in-place edit)
    r"\bperl\b[^\n]*\s-i\b",                   # perl -i -pe ...       (in-place edit)
    r"<<-?\s*['\"]?[A-Za-z_][A-Za-z0-9_]*",   # here-doc (cat <<EOF > f, python - <<PY ...)
    r"\bcp\b\s+[^\s|&;]+\s+[^\s|&;]+",          # cp src dst
    r"\bmv\b\s+[^\s|&;]+\s+[^\s|&;]+",          # mv src dst
    r"\b(touch|truncate|patch|dd)\b\s+[^\s|&;]",  # touch/truncate/patch/dd
    r"(?:^|&&|;|\|)\s*install\b\s+[^\s|&;]",    # coreutils install (NOT `pip install`)
    r"\bapply_patch\b",                         # apply_patch heredoc (codex/terminus style)
]
_SHELL_WRITE_RE = re.compile("|".join(_SHELL_WRITE_PATTERNS))

# Write-target extraction for shell_write_tier. A strong write is an in-repo content edit, the
# kind of step a patch is made of. Tiers order leave-one-out candidates so a bounded budget
# tests real edits before incidental writes (pip caches, copied configs, ...). Matching itself
# stays with _SHELL_WRITE_RE.
_HEREDOC_TARGET_RE = re.compile(r">\s*([^\s|&;>]+)\s*<<")          # cat > path << EOF
_REDIRECT_TARGET_RE = re.compile(r"(?<![0-9])>>?\s*(?!/dev/|&)([^\s|&;>]+)")
_SED_I_TARGET_RE = re.compile(r"\bsed\b[^\n]*\s-i[^\s]*\s+(?:-e\s+)?(?:'[^']*'|\"[^\"]*\"|\S+)\s+([^\s|&;]+)")


def shell_write_targets(command: str) -> list:
    """Return the file paths a shell command writes, best effort (heredoc, redirect, sed -i).

    Returns an empty list when no target can be extracted (pip install, dd, apply_patch, ...).
    """
    if not command or not isinstance(command, str):
        return []
    targets = []
    for rx in (_HEREDOC_TARGET_RE, _SED_I_TARGET_RE, _REDIRECT_TARGET_RE):
        for m in rx.finditer(command):
            t = m.group(1).strip("'\"")
            if t and t not in targets and not t.startswith(("/dev/", "&")):
                targets.append(t)
    return targets


def shell_write_tier(command: str) -> int:
    """Classify a shell command: 2 = strong write, 1 = weak write, 0 = not a write.

    Strong: a write with an extractable target outside /tmp (an in-repo edit). Weak: matches
    the write patterns but has no such target (pip, dd, patch, apply_patch, ...).
    """
    if not shell_writes_file(command):
        return 0
    targets = [t for t in shell_write_targets(command) if not t.startswith("/tmp")]
    return 2 if targets else 1


def shell_writes_file(command: str) -> bool:
    """True if a raw shell command writes, creates or edits a file.

    Conservative: read-only commands do not match. Works for any harness whose tool argument
    is a shell command string.
    """
    if not command or not isinstance(command, str):
        return False
    return bool(_SHELL_WRITE_RE.search(command))


def _shell_command_of(content: dict):
    """Return the raw command string if this tool_call is a shell tool, else None."""
    if (content.get("tool_name") or "") not in SHELL_TOOLS:
        return None
    args = content.get("arguments") or {}
    cmd = args.get("command")
    if cmd is None:           # some shell tools key the string differently
        cmd = args.get("cmd") or args.get("script")
    return cmd if isinstance(cmd, str) else None


def _spec_for(tool, command, spec):
    return spec.get((tool, command)) or spec.get((tool, None))


def effect_of(event_dict, spec=CONTENT_SPEC):
    """Return {path, produces, consumes} if a tool_call event touches file content, else None.

    produces/consumes are strings (possibly empty) or None when the spec has no such field.
    """
    if event_dict.get("kind") != "tool_call":
        return None
    c = event_dict.get("content", {}) or {}
    s = _spec_for(c.get("tool_name"), (c.get("arguments") or {}).get("command"), spec)
    if not s:
        return None
    args = c.get("arguments") or {}
    path = args.get("path")
    if not path:
        return None
    prod_f, cons_f = s
    return {"path": path,
            "produces": args.get(prod_f) if prod_f else None,
            "consumes": args.get(cons_f) if cons_f else None}


def is_write_event(event_dict, spec=CONTENT_SPEC):
    """True if a tool_call event produces file content.

    Typed edit tools (str_replace_editor) are read through the produces field in CONTENT_SPEC.
    Shell tools (execute_bash, exec, ...) have no typed field, so the write is detected from the
    command string (redirection, in-place edit, heredoc, cp/mv/touch, ...).
    """
    if event_dict.get("kind") != "tool_call":
        return False
    eff = effect_of(event_dict, spec)
    if eff and eff.get("produces"):
        return True
    cmd = _shell_command_of(event_dict.get("content", {}) or {})
    return shell_writes_file(cmd)


def build_provenance(events, spec=CONTENT_SPEC):
    """Record content-dependency edges on the events, in place, and return summary stats.

    For each event that consumes a string S on file F, link it to the latest earlier event whose
    produced text on F contains S (substring match, as the edit matches on replay). The producer
    event_id is appended to depends_on and the edge is recorded in
    content.extra['provenance']['content_deps']. When no producer is found, S is assumed to be
    in the base repo and no edge is added. Accepts Event dataclass objects or plain dicts.
    """
    last_by_path = {}   # path -> list of (event_id, produced_text)
    stats = {"version": PROVENANCE_VERSION, "n_writes": 0, "n_content_deps": 0, "n_base_repo": 0}

    for e in events:
        ed = _as_dict(e)
        eff = effect_of(ed, spec)
        if eff is None:
            continue
        path, produces, consumes = eff["path"], eff["produces"], eff["consumes"]
        if consumes:
            producer = None
            for (peid, ptext) in reversed(last_by_path.get(path, [])):
                if ptext and consumes in ptext:
                    producer = peid
                    break
            if producer is not None:
                _append_dep(e, producer)
                _add_content_dep(e, producer, path)
                stats["n_content_deps"] += 1
            else:
                stats["n_base_repo"] += 1
        if produces is not None:
            stats["n_writes"] += 1
            # create replaces the whole file; str_replace/insert append to its history
            cmd = (ed.get("content", {}).get("arguments") or {}).get("command")
            if cmd == "create":
                last_by_path[path] = [(ed["event_id"], produces)]
            else:
                last_by_path.setdefault(path, []).append((ed["event_id"], produces))
    return stats


def content_dep_producers(event_dict):
    """Producer event_ids recorded on an event (for causal closure)."""
    prov = ((event_dict.get("content", {}) or {}).get("extra", {}) or {}).get("provenance", {}) or {}
    return [d["producer_event_id"] for d in prov.get("content_deps", [])]


# Helpers that accept an Event dataclass or a plain dict.
def _as_dict(e):
    if isinstance(e, dict):
        return e
    c = e.content
    return {"event_id": e.event_id,
            "kind": e.kind.value if hasattr(e.kind, "value") else e.kind,
            "content": {"tool_name": c.tool_name, "arguments": c.arguments, "extra": c.extra}}


def _provenance_extra(e):
    if isinstance(e, dict):
        return e.setdefault("content", {}).setdefault("extra", {}).setdefault("provenance", {})
    if e.content.extra is None:
        e.content.extra = {}
    return e.content.extra.setdefault("provenance", {})


def _add_content_dep(e, producer_eid, path):
    prov = _provenance_extra(e)
    prov.setdefault("content_deps", []).append({"producer_event_id": producer_eid, "path": path})


def _append_dep(e, peid):
    deps = e.setdefault("depends_on", []) if isinstance(e, dict) else e.depends_on
    if peid not in deps:
        deps.append(peid)


def causal_closure(keep_msgs, trace):
    """Expand a keep-set of raw message indices so a forced replay prefix is self-sufficient.

    Two edge types are followed, both from an action to the antecedent it needs:
      1. Result to call: a kept tool result pulls in the tool call that produced it. Replay
         forces actions and drops observations, so a kept result without its call would replay
         no action, and the edit would be missing from the patch.
      2. Content dependency: a kept edit that consumes file text S pulls in the earlier edit
         that produced S, so a str_replace does not silently do nothing on a clean repo.
    The call-to-result direction is never followed, so the set does not grow without need.
    Producer event ids are mapped back to raw message indices. Returns (sorted_msgs, info).
    """
    events = trace["events"]
    eid2msg = {e["event_id"]: e["raw_msg"] for e in events}
    by_eid = {e["event_id"]: e for e in events}
    msg2eids = {}
    for e in events:
        msg2eids.setdefault(e["raw_msg"], []).append(e["event_id"])

    keep = set(keep_msgs)
    added = []           # (antecedent_eid, antecedent_msg, edge_kind)
    coarse_msgs = set()  # pulled msgs that also carry tool_calls beyond the antecedent

    def _pull(peid, edge):
        pm = eid2msg.get(peid)
        if pm is None or pm <= 1 or pm in keep:
            return False
        keep.add(pm)
        added.append((peid, pm, edge))
        sibling_calls = [s for s in msg2eids.get(pm, [])
                         if by_eid.get(s, {}).get("kind") == "tool_call"]
        if len(sibling_calls) > 1:
            coarse_msgs.add(pm)
        return True

    changed = True
    while changed:
        changed = False
        for m in list(keep):
            for eid in msg2eids.get(m, []):
                ev = by_eid.get(eid, {})
                if ev.get("kind") == "tool_result":
                    for dep in ev.get("depends_on", []) or []:
                        if by_eid.get(dep, {}).get("kind") == "tool_call":
                            changed |= _pull(dep, "result_to_call")
                for peid in ev.get("content_deps", []) or []:
                    changed |= _pull(peid, "content_dep")
    info = {"closure_added_msgs": sorted({pm for _, pm, _ in added}),
            "closure_added_edges": added,
            "coarse_producer_msgs": sorted(coarse_msgs)}
    return sorted(keep), info
