"""The optimizer's agent loop (the intervener), using Bedrock tool calling.

One loop serves every configuration; the configuration decides which tools are offered:

    always:              list_dir / read_file / grep (read-only file tools, modeled on the
                         Read/Glob/Grep tools of Meta-Harness's proposer) and bash (the
                         analysis sandbox)
    verification tools:  fix_probe / ablate / substitute / replay, when ctx.probes is set
                         (results are recorded as ExperimentRecords and attached to Claims)
    optional:            write_notes, and caller-provided tools (e.g. run_episode)
    last:                submit_proposal (typed edits and predicted fixes)

The file tools are read-only over the run directory (trajectory dumps, reports, a snapshot
of the executor harness), and the sandbox cannot run tasks. Whether the optimizer can run
experiments is therefore set by which tools it has, not by the prompt.

There are no Write/Edit tools: edits go through submit_proposal so the edit-space component
can enforce its constraints (e.g. the single-file rule of typed_hooks). Meta-Harness's
proposer instead writes files directly.
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import time

from verse.claims import Claim, ExperimentRecord

from verse.evolution.protocols import Proposal

_MAX_TOOL_CHARS = 6000
# Episodes with verification tools need roughly 8-12 extra turns for experiments; all
# configurations use the same cap.
_MAX_TURNS = 60
# Endgame: no reminders before the last _ENDGAME_WARN turns; from then on a factual
# [budget] line is added to each tool result. The final turn offers only submit_proposal
# and forces the call, so a long investigation is not lost to a missing submit.
_ENDGAME_WARN = 5


def _as_int(v, default: int) -> int:
    """Parse an integer argument leniently. Models sometimes emit malformed integers
    (e.g. start='300, \\n'); take the first integer in the string, else the default."""
    try:
        return int(v)
    except (TypeError, ValueError):
        m = re.search(r"-?\d+", str(v))
        return int(m.group()) if m else default


def _as_task_ids(v, known: set | None = None) -> list:
    """Normalize a predicted_fixes/at_risk list to bare task ids.

    Models often write entries like 'repo__pkg-123 (rationale...)', which would never
    match a task id. Take the first whitespace-separated token. When `known` is given
    and that token is not a known id, take the first known id that appears anywhere in
    the entry, or drop the entry if there is none."""
    out = []
    for item in v or []:
        s = str(item).strip()
        if not s:
            continue
        tok = s.split()[0].strip("(),;:")
        if known and tok not in known:
            m = next((k for k in known if k in s), None)
            if m is None:
                continue                  # no known id anywhere in the entry
            tok = m
        if tok and tok not in out:
            out.append(tok)
    return out


def _as_edits(v) -> list:
    """Normalize the edits argument. Models sometimes send a string or a JSON-encoded
    list; parse it and keep only dict entries. A degenerate result surfaces downstream
    as 'proposal contains no edits' instead of a crash."""
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return []
    if not isinstance(v, list):
        return []
    return [e for e in v if isinstance(e, dict)]


def _append_user_text(msgs: list, text: str) -> None:
    """Attach text to the trailing user turn if there is one (keeps tool_use/tool_result
    pairing valid for the API), else append a new user message."""
    if msgs and msgs[-1].get("role") == "user" and isinstance(msgs[-1].get("content"), list):
        msgs[-1]["content"].append({"type": "text", "text": text})
    else:
        msgs.append({"role": "user", "content": text})


# File tools
def _tool_list_dir(root: str, rel: str = ".") -> str:
    p = os.path.normpath(os.path.join(root, rel))
    if not p.startswith(os.path.abspath(root)):
        return "ERROR: path escapes the run directory"
    if not os.path.isdir(p):
        return f"ERROR: not a directory: {rel}"
    rows = []
    for name in sorted(os.listdir(p)):
        fp = os.path.join(p, name)
        tag = "dir " if os.path.isdir(fp) else f"{os.path.getsize(fp):>8}B"
        rows.append(f"{tag}  {os.path.join(rel, name)}")
    return "\n".join(rows[:400]) or "(empty)"


def _tool_read_file(root: str, rel: str, start: int = 1, limit: int = 200) -> str:
    p = os.path.normpath(os.path.join(root, rel))
    if not p.startswith(os.path.abspath(root)):
        return "ERROR: path escapes the run directory"
    if not os.path.isfile(p):
        return f"ERROR: no such file: {rel}"
    try:
        lines = open(p, errors="replace").read().splitlines()
    except Exception as e:
        return f"ERROR: {e}"
    start = max(1, _as_int(start, 1))
    chunk = lines[start - 1 : start - 1 + max(1, _as_int(limit, 200))]
    # Long lines are cut for display with an explicit marker; an unmarked cut looks
    # like file corruption to the model.
    body = "\n".join(
        f"{i:5d}| {ln[:500]}"
        + (f" <<DISPLAY-CAP: line is {len(ln)} chars in the file; the file itself is intact>>"
           if len(ln) > 500 else "")
        for i, ln in enumerate(chunk, start))
    more = f"\n... ({len(lines)} lines total)" if start - 1 + len(chunk) < len(lines) else ""
    return (body or "(empty file)") + more


def _tool_grep(root: str, pattern: str, glob: str = "**/*", max_hits: int = 80) -> str:
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"ERROR: bad regex: {e}"
    hits = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            fp = os.path.join(dirpath, name)
            rel = os.path.relpath(fp, root)
            if not fnmatch.fnmatch(rel, glob):
                continue
            try:
                for i, ln in enumerate(open(fp, errors="replace"), 1):
                    if rx.search(ln):
                        s = ln.strip()
                        hits.append(f"{rel}:{i}: {s[:200]}"
                                    + (f" <<DISPLAY-CAP: line is {len(s)} chars>>" if len(s) > 200 else ""))
                        if len(hits) >= max_hits:
                            return "\n".join(hits) + f"\n... (capped at {max_hits})"
            except Exception:
                continue
    return "\n".join(hits) or "(no matches)"


# Optimizer bash sandbox
# An analysis shell, as in the baselines' optimizers (AHE's run_shell_command,
# Meta-Harness's proposer Bash), which use it to validate artifacts and analyze
# trajectories rather than to run benchmark tasks. The run directory is read-only,
# /scratch is writable, and python is available. There is no network, so the optimizer
# cannot look up upstream fixes for the tasks, and no task containers: real execution
# comes only from the verification tools. Offered in every configuration.
_TEACHER_BASH_CAP = 60           # commands per episode
_TEACHER_BASH_TIMEOUT = 120      # seconds per command


def _tool_teacher_bash(run_dir: str, scratch: str, cmd: str) -> str:
    if not (cmd or "").strip():
        return "ERROR: empty command"
    docker = os.environ.get("T2E_SWE_DOCKER", "docker")
    image = os.environ.get("T2E_TEACHER_BASH_IMAGE", "python:3.11-slim")
    os.makedirs(scratch, exist_ok=True)
    import shlex as _sh
    import subprocess as _sp
    full = (f"{docker} run --rm --network none --cpus=2 -m 2g "
            f"-v {_sh.quote(os.path.abspath(run_dir))}:/run_dir:ro "
            f"-v {_sh.quote(os.path.abspath(scratch))}:/scratch "
            f"-w /run_dir {image} bash -c {_sh.quote(cmd)}")
    try:
        p = _sp.Popen(full, shell=True, stdout=_sp.PIPE, stderr=_sp.PIPE,
                      text=True, start_new_session=True)
        out, err = p.communicate(timeout=_TEACHER_BASH_TIMEOUT)
        body = (out or "") + (("\n" + err) if err else "")
        return body.strip() or f"(exit {p.returncode}, no output)"
    except _sp.TimeoutExpired:
        try:
            pgid = os.getpgid(p.pid)
            _sp.run(f"kill -9 -{pgid} 2>/dev/null", shell=True, timeout=10)
        except Exception:
            pass
        return f"ERROR: command timed out after {_TEACHER_BASH_TIMEOUT}s"
    except Exception as e:
        return f"ERROR: {e}"


# Tool schemas
_FILE_TOOLS = [
    {"name": "list_dir",
     "description": "List a directory inside the run dir (trajectories, reports, harness workspace).",
     "input_schema": {"type": "object", "properties": {
         "path": {"type": "string", "description": "relative path, default '.'"}},
         "required": []}},
    {"name": "read_file",
     "description": "Read a file inside the run dir with line numbers. Use start/limit to page.",
     "input_schema": {"type": "object", "properties": {
         "path": {"type": "string"}, "start": {"type": "integer"}, "limit": {"type": "integer"}},
         "required": ["path"]}},
    {"name": "grep",
     "description": "Regex-search across files in the run dir (e.g. find which tasks share an error).",
     "input_schema": {"type": "object", "properties": {
         "pattern": {"type": "string"}, "glob": {"type": "string", "description": "e.g. trajectories/*.md"}},
         "required": ["pattern"]}},
]

_PROBE_TOOLS = [
    {"name": "fix_probe",
     "description": ("VERIFIED EXPERIMENT — USE THIS BEFORE SUBMITTING: apply your DRAFT harness "
                     "edits to a scratch workspace and re-run up to 3 tasks you predict they fix. "
                     "Returns real flip results. A verified fix is worth far more than an untested "
                     "one; a 0-flip result means revise the edits or the target list. Costs one "
                     "budget unit per task run."),
     "input_schema": {"type": "object", "properties": {
         "edits": {"type": "array", "description": "same format as submit_proposal edits",
                   "items": {"type": "object", "properties": {
                       "path": {"type": "string"}, "content": {"type": "string"},
                       "delete": {"type": "boolean"}}, "required": ["path"]}},
         "target_task_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 3,
                             "description": "failed tasks these edits should fix (<=3)"}},
         "required": ["edits", "target_task_ids"]}},
    {"name": "ablate",
     "description": ("VERIFIED EXPERIMENT: replay a task WITHOUT step k. On a FAILED task a flip "
                     "to pass proves step k the culprit (rare — most failures are missing-fix). "
                     "On a PASSING task a flip to fail PROVES that behavior load-bearing for the "
                     "win — use this to confirm which successful behaviors to distill into skills. "
                     "Step indices are shown in trajectory files."),
     "input_schema": {"type": "object", "properties": {
         "task_id": {"type": "string"}, "step": {"type": "integer"}}, "required": ["task_id", "step"]}},
    {"name": "substitute",
     "description": ("VERIFIED EXPERIMENT: replay a failed task with step k's command REPLACED by "
                     "a better one you propose. A flip to pass proves a better action existed there "
                     "and validates your exact replacement — direct evidence for a workflow/skill "
                     "edit teaching that action."),
     "input_schema": {"type": "object", "properties": {
         "task_id": {"type": "string"}, "step": {"type": "integer"},
         "new_command": {"type": "string"}}, "required": ["task_id", "step", "new_command"]}},
    {"name": "replay",
     "description": ("VERIFIED EXPERIMENT (validity ticket): re-execute a task's recorded commands "
                     "in a fresh environment. Confirms the world is deterministic before you trust "
                     "step-level results; a non-reproduction voids all step claims on that trace."),
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "string"}},
                      "required": ["task_id"]}},
]

# Contract-only descriptions for the `verse_*` configurations: mechanism, arguments, cost and
# return value, without usage advice. Names and schemas match _PROBE_TOOLS, so usage notes
# the optimizer writes refer to the same tools.
_PROBE_TOOLS_CONTRACT = [
    {"name": "fix_probe",
     "description": ("EXPERIMENT: apply DRAFT harness edits to a scratch copy of the "
                     "workspace and re-run up to 3 named TRAIN tasks under them. Returns "
                     "per-task pass/fail; a fail->pass flip is re-run once and only "
                     "counts if it reproduces. Costs one budget unit per task run. "
                     "Results are bound to the exact edits tested (their hash)."),
     "input_schema": _PROBE_TOOLS[0]["input_schema"]},
    {"name": "ablate",
     "description": ("EXPERIMENT: replay a task's recorded commands WITHOUT step k, in a "
                     "fresh environment, then grade the outcome. Works on failed and "
                     "passing trajectories. Step indices are shown in trajectory files. "
                     "Costs one budget unit."),
     "input_schema": _PROBE_TOOLS[1]["input_schema"]},
    {"name": "substitute",
     "description": ("EXPERIMENT: replay a FAILED task's recorded commands with step k "
                     "replaced by a command you supply, then grade the outcome. Costs "
                     "one budget unit."),
     "input_schema": _PROBE_TOOLS[2]["input_schema"]},
    {"name": "replay",
     "description": ("EXPERIMENT: re-execute a task's recorded commands unchanged in a "
                     "fresh environment and grade the outcome (reproducibility check). "
                     "Costs one budget unit."),
     "input_schema": _PROBE_TOOLS[3]["input_schema"]},
]

_BASH_TOOL_TEACHER = {
    "name": "bash",
    "description": ("Run a shell command in a sandboxed ANALYSIS container (python3 + "
                    "coreutils; no network). The run directory is mounted READ-ONLY at "
                    "/run_dir (your cwd); /scratch is writable for scripts and notes. "
                    "Use it to compute over trajectories (cluster errors, count patterns), "
                    "prototype/lint code you plan to propose (python /scratch/draft.py), "
                    "and validate file formats. It CANNOT run benchmark tasks — use "
                    "fix_probe (when available) for real counterfactual execution."),
    "input_schema": {"type": "object", "properties": {
        "command": {"type": "string", "description": "the shell command"}},
        "required": ["command"]}}

_NOTES_TOOL = {
    "name": "write_notes",
    "description": ("Persist strategy notes ACROSS ROUNDS (max 4KB, replaces previous notes). "
                    "They are shown back to you at the start of every future round — use them "
                    "so your successor does not re-derive your findings from scratch: which "
                    "failure mechanisms are live vs extinct, what was tried, what to attack "
                    "next, where the bodies are buried in trajectories/. AHE-style long-term "
                    "memory for the harness engineer itself."),
    "input_schema": {"type": "object", "properties": {
        "content": {"type": "string", "description": "full replacement notes (markdown)"}},
        "required": ["content"]}}

# Verification-tool prompt text
# Two pieces, both added to the prompt of the `verified_*` configurations (fixed optimizer):
#   _PROBE_CONTRACT  the verdict vocabulary, with usage advice
#   PROBE_WISDOM     the hand-written usage strategy
#
# `verse_*` configurations get neither in the prompt. They see the advice-free vocabulary
# (_PROBE_CONTRACT_NEUTRAL) and the contract-only tool descriptions (_PROBE_TOOLS_CONTRACT);
# how to use the tools is left for the optimizer to learn and record in its own harness.
# With seed_wisdom, their optimizer harness starts with the PROBE_WISDOM text as editable
# content (see meta_teacher._seed_equipment).
_PROBE_CONTRACT = (
    "\n\n## How to use experiment verdicts\n"
    "[VERIFIED] = mechanism confirmed by a real experiment — trust and prioritize it.\n"
    "[REFUTED] = that SPECIFIC claim was disproven — re-diagnose or revise rather than "
    "dropping the task from consideration.\n"
    "[HYPOTHESIS] = untested — still actionable on its own merits; verify with a probe "
    "when it is load-bearing for your edit, otherwise use normal judgment.\n")
# Advice-free version of the vocabulary (`verse_*` configurations): what each label means,
# not what to do about it
_PROBE_CONTRACT_NEUTRAL = (
    "\n\n## Experiment verdict vocabulary\n"
    "Experiment tools label their results:\n"
    "[VERIFIED] = the claim was confirmed by real execution.\n"
    "[REFUTED] = that specific claim was contradicted by real execution.\n"
    "[HYPOTHESIS] = a claim no experiment has tested.\n"
    "[INCONCLUSIVE] = the experiment ran but did not decide the claim.\n")
PROBE_WISDOM = (
    "\n## Experiment strategy (spend your budget where the answer is uncertain)\n"
    "1. fix_probe is your highest-value tool: BEFORE submitting, draft your edits and "
    "fix_probe them on 2-3 tasks you predict they fix. Historical predicted-fix hit "
    "rates without testing are 0-16% — a verified draft is the single biggest upgrade "
    "you can make to a proposal. Iterate: 0 flips -> revise -> probe again, but "
    "AT MOST twice; then submit your best change-set regardless. A refuted probe "
    "downgrades specific predicted_fixes, it does not veto the proposal — probes "
    "test 1-3 noisy tasks, and probe-refuted proposals have measured best-of-arm "
    "on the full validation sweep. NEVER end an episode without submitting: an "
    "unsubmitted episode scores zero and wastes the whole round. "
    "Evidence attaches to the EXACT change-set probed: submit the same edits your "
    "last successful probe tested (byte-identical), or the verification does not "
    "carry over to your submission.\n"
    "2. The report's Attribute layer (when present) already carries EXECUTED minimal "
    "cores with an omission/commission classification and a failure-mode ledger — "
    "route on it: OMISSION modes need the harness to teach the missing behavior "
    "(draft the edit, validate with fix_probe); COMMISSION modes need the causal "
    "step banned or replaced (validate the replacement with substitute). Attack the "
    "largest ALIVE/GROWING ledger modes first; do not re-attack EXTINCT ones.\n"
    "3. ablate PASSING trajectories to prove which winning behaviors are load-bearing, "
    "then distill exactly those into skills (successes live in trajectories/ too).\n"
    "4. Do NOT spend budget re-confirming what is already near-certain (replaying a "
    "deterministic failure, ablating single steps of failed traces: these almost never "
    "flip). Verification tells you WHERE to be confident, not to shrink your ambition: "
    "propose the full set of edits the evidence supports.")

def _wisdom_for(mounted) -> str:
    """PROBE_WISDOM restricted to the offered subset of verification tools.

    With all four tools (or no subset given) the constant is returned unchanged.
    Otherwise strategy items that need a missing tool are dropped and the rest are
    renumbered. Items are single lines keyed by their leading number."""
    names = set(mounted or ())
    if not names or names >= {"replay", "ablate", "substitute", "fix_probe"}:
        return PROBE_WISDOM
    required = {"1": {"fix_probe"}, "2": {"fix_probe", "substitute"},
                "3": {"ablate"}, "4": set()}
    out, n = [], 0
    for line in PROBE_WISDOM.split("\n"):
        if len(line) > 2 and line[1] == "." and line[0].isdigit():
            if required.get(line[0], set()) <= names:
                n += 1
                out.append(f"{n}." + line[2:])
        else:
            out.append(line)
    return "\n".join(out)


_SUBMIT_TOOL = {
    "name": "submit_proposal",
    "description": ("Submit your final harness change-set. Every claim in root_causes should be "
                    "backed by evidence you actually gathered (experiments when available)."),
    "input_schema": {"type": "object", "properties": {
        "description": {"type": "string"},
        "rationale": {"type": "string", "description": "root-cause reasoning behind the edits"},
        "edits": {"type": "array", "items": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"},
            "delete": {"type": "boolean"}}, "required": ["path"]}},
        "predicted_fixes": {"type": "array", "items": {"type": "string"}},
        "at_risk": {"type": "array", "items": {"type": "string"}},
        "base_round": {"type": "integer", "description":
            "OPTIONAL branch point: apply your edits on the harness state AFTER this round "
            "instead of the current head. Use when history shows a later round was HARMFUL "
            "and you want to build from the last good state (0 = the original harness). "
            "Only kept rounds (or 0) are valid bases."}},
        "required": ["description", "edits", "predicted_fixes"]}}


# The loop
_MAX_NOTES_CHARS = 4096


def run_intervener(prompt: str, run_dir: str, ctx, *, model: str, regions: str,
                   max_turns: int = _MAX_TURNS, audit_path: str | None = None,
                   notes_path: str | None = None, validate_payload=None,
                   extra_tools: list | None = None,
                   bash_cap: int = _TEACHER_BASH_CAP,
                   allow_empty_edits: bool = False,
                   teacher_hooks=None, hooks_pipeline: bool = True,
                   max_out_tokens: int = 32768,
                   probe_wisdom: bool = True) -> Proposal:
    """Run one optimizer episode (explore, optionally experiment, submit) and return a
    Proposal.

    The verification tools are offered iff ctx.probes is set; their results are recorded
    as ExperimentRecords and attached to the proposal's claims. audit_path receives a
    JSONL log of every tool call. notes_path enables write_notes: memory across rounds
    that the driver adds to every round's prompt.

    validate_payload(payload_dict) -> error_str or '': when given, a submit_proposal
    whose change-set fails validation is rejected inside the episode. The error comes
    back as the tool result and the episode continues, so the optimizer can fix it while
    it still has its context. The forced final turn accepts any payload (no turns are
    left to retry); callers check the result afterwards (the driver's smoke-repair step,
    the meta-episode's own checks).

    extra_tools: [{spec, fn}] with fn(args) -> str, caller-provided tools offered next to
    the file tools (e.g. HarnessX's digest_trajectories worker, or run_episode in the
    self_teacher configurations).

    bash_cap: number of analysis-sandbox commands allowed in this episode (meta-episodes
    use a smaller cap than candidate episodes).

    allow_empty_edits: accept a submit without edits if its description starts with
    'NO_CHANGES' (used by the meta-episode).

    teacher_hooks: (HookedRuntime, env), the optimizer's self-written code, loaded on its
    own loop at the same hooks the executor loop exposes (system_prompt, before_llm,
    after_llm, before_tool, after_tool, on_turn_end), plus the tools it defines, which
    act through env. Hooks are fail-open, as on the executor side. hooks_pipeline=False
    loads only the tools; the meta-episode uses this so broken self-written hooks cannot
    block the episode that would repair them.

    probe_wisdom: add the hand-written usage strategy (PROBE_WISDOM) and the advice-carrying
    tool descriptions. True for the `verified_*` configurations; False for the VERSE
    configurations, which get contract-only descriptions (with seed_wisdom, their copy of
    the strategy lives in teacher_ws/prompt.md)."""
    from verse.runtime.executor import _invoke_bedrock_raw, _TEMP_DEPRECATED

    known_tids = set(getattr(ctx, "insts_by_tid", None) or {})
    tools = list(_FILE_TOOLS)
    tools.append(_BASH_TOOL_TEACHER)
    # Per-episode scratch: candidates run concurrently and must not read each other's
    # drafts. The key is the audit filename stem (intervener_audit_<lens>), else the pid.
    ep_key = (os.path.basename(audit_path).replace("intervener_audit_", "")
              .replace(".jsonl", "") if audit_path else f"ep{os.getpid()}")
    bash_scratch = os.path.join(run_dir, "teacher_scratch", ep_key)
    bash_used = 0
    extra_fns = {}
    for t in extra_tools or []:
        spec = t.get("spec") or {}
        if spec.get("name") and callable(t.get("fn")):
            tools.append(spec)
            extra_fns[spec["name"]] = t["fn"]
    th_rt, th_env = teacher_hooks if teacher_hooks else (None, None)
    th_tool_names = set()
    if th_rt is not None:
        # loop settings, validated as on the executor side: max_turns can only shrink,
        # and obs_cap sets the tool-result display cap
        lc = th_rt.loop_config(max_turns)
        if "max_turns" in lc:
            max_turns = min(max_turns, lc["max_turns"])
        tool_chars = int(lc.get("obs_cap", _MAX_TOOL_CHARS))
        for spec in th_rt.extra_tool_specs():
            if spec["name"] not in extra_fns:       # caller-provided tools win on clash
                tools.append(spec)
                th_tool_names.add(spec["name"])
        if hooks_pipeline:
            prompt = th_rt.system_prompt(prompt)
    else:
        tool_chars = _MAX_TOOL_CHARS
    if notes_path is not None:
        tools.append(_NOTES_TOOL)
    scheduler = ctx.extra.get("probe_scheduler") if ctx.probes is not None else None
    sched_records: list = []
    if scheduler is not None:
        # Scheduling ablation: the verification tools and their budget stay, but a
        # policy (random or fixed rule) chooses the experiments and the optimizer only
        # reads the results. The tools are not offered to the optimizer; at submission
        # the policy also runs fix_probe on the submitted draft (see the submit branch).
        sched_records = list(scheduler.pre_episode(ctx))
        prompt += _PROBE_CONTRACT if probe_wisdom else _PROBE_CONTRACT_NEUTRAL
        prompt += ("\n\n## Scheduled experiment results\n"
                   "An automatic scheduler ran these real-execution experiments; "
                   "you cannot run more yourself. Your submitted edits will "
                   "additionally be probed against real episodes after you submit.\n"
                   + ("\n".join(f"[{r.verdict.upper()}] {r.setup} -> {r.outcome}"
                                 for r in sched_records) or "(none ran this round)"))
    elif ctx.probes is not None and ctx.extra.get("probe_tools", True):
        mounted = getattr(ctx.probes, "tools", None)
        if probe_wisdom:
            # `verified_*` configurations (fixed optimizer): advice-carrying descriptions
            # and the hand-written usage strategy, filtered to the configured tool subset
            tools += [t for t in _PROBE_TOOLS
                      if mounted is None or t["name"] in mounted]
            prompt += _PROBE_CONTRACT + _wisdom_for(mounted)
        else:
            # `verse_*` configurations: contract-only descriptions and the neutral
            # vocabulary; any usage strategy lives in the optimizer harness
            tools += [t for t in _PROBE_TOOLS_CONTRACT
                      if mounted is None or t["name"] in mounted]
            prompt += _PROBE_CONTRACT_NEUTRAL
    tools.append(_SUBMIT_TOOL)

    msgs = [{"role": "user", "content": prompt}]
    experiments: list = list(sched_records)   # ExperimentRecords collected this episode
    audit = open(audit_path, "a") if audit_path else None

    def _log(kind: str, payload: dict):
        if audit:
            audit.write(json.dumps({"ts": time.time(), "kind": kind, **payload}) + "\n")
            audit.flush()

    try:
        nudges = 0
        for turn in range(max_turns):
            turns_left = max_turns - turn            # counting the current one
            final_turn = turns_left == 1
            # optimizer-side before_llm (self_teacher only): the self-written message
            # transform; fail-open, and the last message must stay a user message
            body_msgs = (th_rt.before_llm(msgs, turn)
                         if th_rt is not None and hooks_pipeline else msgs)
            body = {"max_tokens": max_out_tokens,
                    "tools": ([_SUBMIT_TOOL] if final_turn else tools),
                    "messages": body_msgs}
            if final_turn:
                # last turn: only submit_proposal is offered and the call is forced
                body["tool_choice"] = {"type": "tool", "name": "submit_proposal"}
            if not any(t in model for t in _TEMP_DEPRECATED):
                body["temperature"] = 0.0
            try:
                resp = _invoke_bedrock_raw(model, regions, body)
            except Exception:
                if not final_turn:
                    raise
                # some backends reject a forced tool choice; retry the final turn with the
                # instruction in the message instead (still only submit_proposal)
                body.pop("tool_choice", None)
                _append_user_text(msgs, "FINAL TURN: call submit_proposal NOW with your "
                                        "best change-set from the evidence you have.")
                body["messages"] = msgs
                resp = _invoke_bedrock_raw(model, regions, body)
            content = resp.get("content", [])
            # Output-limit truncation: when the reply hits max_tokens inside a tool_use,
            # the API keeps only the fully parsed top-level input keys, so `edits` (the
            # largest, emitted last) is lost. Detect this and report the real cause;
            # otherwise the model retries the same oversized submit.
            truncated = resp.get("stop_reason") == "max_tokens"
            if truncated and any(b.get("type") == "tool_use" for b in content):
                _log("output_truncated", {"turn": turn, "tools":
                     [b.get("name") for b in content if b.get("type") == "tool_use"]})
            if th_rt is not None and hooks_pipeline:
                # optimizer-side after_llm: self-written repair of the raw model output
                content = th_rt.after_llm(content, turn)
            msgs.append({"role": "assistant", "content": content})
            tool_uses = [b for b in content if b.get("type") == "tool_use"]

            if not tool_uses:
                nudges += 1
                if nudges > 2 or final_turn:
                    raise RuntimeError("intervener ended without submit_proposal")
                nudge = "Use the tools to investigate, then call submit_proposal."
                if turns_left - 1 <= _ENDGAME_WARN:
                    nudge += (f"\n[budget] {turns_left - 1} turns remain; an episode that "
                              "never submits scores zero.")
                msgs.append({"role": "user", "content": nudge})
                continue

            results = []
            for tu in tool_uses:
                name, tid, args = tu.get("name"), tu.get("id"), tu.get("input") or {}
                _log("tool_call", {"turn": turn, "tool": name, "args": args})
                if name == "submit_proposal":
                    edits = _as_edits(args.get("edits"))
                    if truncated and not final_turn:
                        # The reply was cut by the output limit: `edits` is either
                        # missing or partial. Neither is safe to commit, so reject the
                        # submit and name the cause and the fix.
                        verr = ("your submit_proposal call EXCEEDED THE OUTPUT TOKEN "
                                "LIMIT and was cut off mid-generation — the `edits` "
                                "array "
                                + ("arrived INCOMPLETE" if edits else "was LOST IN "
                                   "TRANSIT (not missing from your intent)")
                                + ". Do NOT retry the same oversized submit. "
                                "Resubmit ONE smaller change-set: fewer files (start "
                                "with the 1-2 highest-value ones) and trimmed file "
                                "contents, with a brief description/rationale. A "
                                "focused change-set that arrives intact beats a "
                                "comprehensive one that cannot fit.")
                        results.append({"type": "tool_result", "tool_use_id": tid,
                                        "content": "PROPOSAL REJECTED — fix and "
                                                   "resubmit:\n" + verr[:1500]})
                        _log("submit_rejected", {"turn": turn,
                                                 "error": "output-cap truncation"})
                        continue
                    if truncated and final_turn:
                        # Forced final turn: this submit skips validation, so a truncated
                        # one would become an empty, invalid proposal. Retry once with
                        # the cause named.
                        _log("final_truncation_retry", {"turn": turn})
                        msgs.append({"role": "user", "content": [
                            {"type": "tool_result", "tool_use_id": tid,
                             "content": "REJECTED: your reply hit the output-token "
                                        "limit and the `edits` array was lost. FINAL "
                                        "CHANCE: resubmit ONE small change-set NOW — "
                                        "at most 2 files, trimmed contents, brief "
                                        "description."}]})
                        body_r = {"max_tokens": max_out_tokens,
                                  "tools": [_SUBMIT_TOOL], "messages": msgs,
                                  "tool_choice": {"type": "tool",
                                                  "name": "submit_proposal"}}
                        if not any(t in model for t in _TEMP_DEPRECATED):
                            body_r["temperature"] = 0.0
                        try:
                            resp_r = _invoke_bedrock_raw(model, regions, body_r)
                        except Exception:
                            body_r.pop("tool_choice", None)
                            resp_r = _invoke_bedrock_raw(model, regions, body_r)
                        tu_r = next((b for b in resp_r.get("content", [])
                                     if b.get("type") == "tool_use"
                                     and b.get("name") == "submit_proposal"), None)
                        if tu_r is not None:
                            args = tu_r.get("input") or {}
                            edits = _as_edits(args.get("edits"))
                            _log("tool_call", {"turn": turn, "tool": "submit_proposal",
                                               "args": args, "rescued": True})
                    if validate_payload is not None and not final_turn:
                        verr = ""
                        if not edits and not allow_empty_edits:
                            verr = ("your proposal contains no edits (the `edits` array "
                                    "is empty or malformed — each entry needs a `path` "
                                    "and full `content`). If you DID include edits, your "
                                    "reply may have hit the output-token cap — split the "
                                    "change-set into smaller submits (<=2 files each)")
                        elif not edits and not str(args.get("description", "")).strip() \
                                .upper().startswith("NO_CHANGES"):
                            # allow_empty_edits (the meta-episode): models sometimes
                            # describe changes in prose and leave the edits array empty,
                            # which would discard the drafted work. A zero-change submit
                            # must say so explicitly.
                            verr = ("your submit contains NO edits — nothing will change. "
                                    "File changes must ride in the `edits` array as FULL "
                                    "file content (sandbox drafts are NOT submitted "
                                    "automatically; a description alone does nothing). "
                                    "If you meant to change files, resubmit with them in "
                                    "`edits`. If you truly intend ZERO changes this round, "
                                    "resubmit with a description starting with "
                                    "'NO_CHANGES'.")
                        else:
                            try:
                                verr = validate_payload(
                                    {"description": args.get("description", ""),
                                     "edits": edits,
                                     "predicted_fixes": _as_task_ids(
                                         args.get("predicted_fixes"), known_tids),
                                     "at_risk": _as_task_ids(
                                         args.get("at_risk"), known_tids)}) or ""
                            except Exception as e:
                                verr = ""          # a validator crash must not end the episode
                        if verr:
                            results.append({"type": "tool_result", "tool_use_id": tid,
                                            "content": "PROPOSAL REJECTED — fix and "
                                                       "resubmit:\n" + verr[:1500]})
                            _log("submit_rejected", {"turn": turn, "error": verr[:300]})
                            continue
                    if scheduler is not None and edits:
                        # Scheduled fix_probe on the submitted draft, which exists only
                        # now. The policy chooses the targets; the records carry this
                        # change-set's edits_sha, so they count as evidence for the
                        # final submission.
                        for rec in scheduler.on_submit(
                                ctx, edits,
                                _as_task_ids(args.get("predicted_fixes"), known_tids)):
                            experiments.append(rec)
                            _log("scheduled_probe", {"turn": turn, "kind": rec.kind,
                                                     "verdict": rec.verdict})
                    prop = Proposal(
                        description=args.get("description", ""),
                        rationale=args.get("rationale", ""),
                        files=[e.get("path", "") for e in edits],
                        predicted_fixes=_as_task_ids(args.get("predicted_fixes"),
                                                     known_tids),
                        at_risk=_as_task_ids(args.get("at_risk"), known_tids),
                        raw=json.dumps(args), meta={"turns": turn + 1,
                                                    "n_experiments": len(experiments)})
                    if args.get("base_round") is not None:
                        prop.meta["base_round"] = _as_int(args.get("base_round"), -1)
                    # Evidence attribution: a fix_probe verdict applies to one change-set
                    # (its edits_sha). Records for earlier, discarded drafts are kept
                    # apart, so a refuted draft does not decide the status of the final
                    # claim.
                    from verse.evolution.probes import edits_sha as _esha
                    sha_final = _esha(edits)
                    final_ev, draft_ev = [], []
                    for rec in experiments:
                        rsha = (rec.detail or {}).get("edits_sha")
                        (draft_ev if (rec.kind == "fix_probe" and rsha != sha_final)
                         else final_ev).append(rec)
                    prop.claims = [Claim(statement=prop.rationale[:500] or prop.description,
                                         evidence=final_ev,
                                         tags={"edits_sha": sha_final})]
                    # One claim per predicted fix, in every configuration, so claims have
                    # the same structure everywhere. A claim's evidence is the final
                    # records that name its task; without verification tools it stays
                    # empty (HYPOTHESIS).
                    for t in prop.predicted_fixes:
                        ev_t = [r for r in final_ev if t in (r.task_id or "")]
                        prop.claims.append(Claim(
                            statement=f"the submitted change-set should fix {t}",
                            task_id=t, evidence=ev_t,
                            tags={"predicted_fix": True, "edits_sha": sha_final}))
                    if draft_ev:
                        prop.claims.append(Claim(
                            statement=("Draft exploration history (change-sets probed and "
                                       "DISCARDED before the final submission — do not "
                                       "re-propose these exact edits):"),
                            evidence=draft_ev, tags={"exploration": True}))
                    prop.meta["edits"] = edits
                    _log("submit", {"turn": turn, "files": prop.files,
                                    "predicted_fixes": prop.predicted_fixes})
                    return prop
                # optimizer-side before_tool (self_teacher only): rewrite, block or answer
                # the optimizer's own tool calls; never applied to submit_proposal
                blocked_out = None
                if th_rt is not None and hooks_pipeline:
                    args, block, synth = th_rt.before_tool(name or "", args, turn)
                    if block:
                        blocked_out = synth if synth else f"[teacher harness] blocked: {block}"
                if blocked_out is not None:
                    out = blocked_out
                elif name in th_tool_names and th_rt is not None:
                    # self-written tool: implemented in teacher_code, acting through env
                    out = th_rt.run_tool(name, args, th_env, turn)
                elif name == "bash":
                    bash_used += 1
                    if bash_used > bash_cap:
                        out = (f"ERROR: bash budget ({bash_cap} commands/episode) "
                               "exhausted — proceed with the evidence you have")
                    else:
                        out = _tool_teacher_bash(run_dir, bash_scratch,
                                                 args.get("command", ""))
                elif name == "write_notes" and notes_path is not None:
                    body_txt = str(args.get("content") or "")[:_MAX_NOTES_CHARS]
                    try:
                        with open(notes_path, "w") as nf:
                            nf.write(body_txt)
                        out = (f"Notes saved ({len(body_txt)} chars) — they will open "
                               "every future round's briefing.")
                    except Exception as e:
                        out = f"ERROR: could not save notes: {e}"
                elif name == "list_dir":
                    out = _tool_list_dir(run_dir, args.get("path", "."))
                elif name == "read_file":
                    out = _tool_read_file(run_dir, args.get("path", ""),
                                          args.get("start", 1), args.get("limit", 200))
                elif name == "grep":
                    out = _tool_grep(run_dir, args.get("pattern", ""), args.get("glob", "**/*"))
                elif name in extra_fns:
                    try:
                        out = str(extra_fns[name](args))
                    except Exception as e:
                        out = f"ERROR: {name} failed: {e!r}"
                elif name in ("replay", "ablate", "substitute", "fix_probe") \
                        and ctx.probes is not None \
                        and ctx.extra.get("probe_tools", True) \
                        and scheduler is None \
                        and name in getattr(ctx.probes, "tools", (name,)):
                    if name == "ablate":
                        rec = ctx.probes.ablate(ctx, args.get("task_id", ""),
                                                _as_int(args.get("step"), -1))
                    elif name == "substitute":
                        rec = ctx.probes.substitute(ctx, args.get("task_id", ""),
                                                    _as_int(args.get("step"), -1),
                                                    args.get("new_command", ""))
                    elif name == "fix_probe":
                        rec = ctx.probes.fix_probe(ctx, _as_edits(args.get("edits")),
                                                   _as_task_ids(args.get("target_task_ids"), known_tids))
                    else:
                        rec = ctx.probes.replay(ctx, args.get("task_id", ""))
                    experiments.append(rec)
                    out = (f"[{rec.verdict.upper()}] {rec.setup} -> {rec.outcome}"
                           + (f" (probes left: {ctx.probes.remaining()})" if ctx.probes else ""))
                else:
                    out = f"ERROR: unknown tool {name}"
                # optimizer-side after_tool (self_teacher only): rewrite raw tool output
                # before the optimizer reads it (annotate, deduplicate, extract)
                if th_rt is not None and hooks_pipeline and blocked_out is None:
                    out = th_rt.after_tool(name or "", args, out, turn)
                # cap with an explicit marker: an unmarked cut mid-sentence reads as
                # data corruption
                if len(out) > tool_chars:
                    out = (out[:tool_chars]
                           + f"\n<<DISPLAY-CAP: result is {len(out)} chars; showing first "
                             f"{tool_chars}. Use start/limit or a narrower query for the rest;"
                             f" the underlying data is intact>>")
                results.append({"type": "tool_result", "tool_use_id": tid,
                                "content": out})
                _log("tool_result", {"turn": turn, "tool": name, "out_head": out[:300]})
            # endgame: a factual budget line, added only in the last _ENDGAME_WARN turns
            if 0 < turns_left - 1 <= _ENDGAME_WARN:
                results.append({"type": "text", "text":
                                f"[budget] {turns_left - 1} turns remain; an episode that "
                                "never calls submit_proposal scores zero."})
            # optimizer-side on_turn_end (self_teacher only): a self-written note for the
            # next turn
            if th_rt is not None and hooks_pipeline:
                note = th_rt.on_turn_end(turn)
                if note:
                    results.append({"type": "text", "text": f"[your harness] {note}"})
            msgs.append({"role": "user", "content": results})
        raise RuntimeError(f"intervener hit {max_turns}-turn cap without submitting")
    finally:
        if audit:
            audit.close()
