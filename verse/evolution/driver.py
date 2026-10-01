"""driver.py — the harness-evolution round loop (runs on a CPU box).

One driver serves every method. A method yaml in verse/configs/ names its components and is
deep-merged over verse/configs/_base.yaml, which holds the shared protocol. For example,
self_teacher/verse_mh.yaml:

    name: verse_mh
    evidence:   {kind: pull_explore}
    probes:     {kind: t2t_probes, budget: 8, budget_s: 3600}   # omitted by the baselines
    attribute:  {max_fail: 6, max_success: 2, wall_s: 2400}     # omitted by the baselines
    edit_space: {kind: code_hooks}
    meta_teacher: {self_edit: true, seed_wisdom: true, ...}     # optimizer self-evolution

and _base.yaml supplies the rest:

    intervener: {model: qwen38-flash-next, regions: us-west-2, max_turns: 60}
    executor:   {model: qwen38-27b, regions: us-west-2, max_turns: 100}
    gate: {kind: keep_all}
    rounds: 6
    sweep:  {concurrency: 24, flip_confirm: true}
    search: {n_candidates: 3, screen_tasks: 12}
    data:   {train_parquet: ..., val_parquet: ..., train_tasks: 110, val_tasks: 50, seed: 42}

Per-round flow:
    1. Train sweep, run only after a kept edit or two consecutive failed rounds (otherwise
       the last trajectories are reused).
    2. evidence.gather builds the report. n_candidates optimizer episodes each produce a
       Proposal, and a screen on a shared train subset picks one.
    3. Smoke check with a repair loop (up to 3 attempts: apply, assemble, run one task, and
       send any error back to the optimizer).
    4. Val sweep under the new harness.
    5. Flip confirmation on val: each flipped task is rerun once; unreproduced flips are
       discarded.
    6. gate.judge keeps the edit or reverts it.
    7. With meta_teacher: the round report and the optimizer self-evolution step.
After the last round, the top val rounds are re-evaluated and the best one is selected.
All state is journaled to out_dir/rounds.jsonl (resumable); trajectories, reports and audit
logs go to out_dir/round_NN/. predicted_fixes are checked against the next kept train sweep.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from verse.claims import dump_claims, render_claims
from verse.evolution import build_component
from verse.evolution.protocols import EvolutionContext, Proposal
from verse.evolution.intervener import run_intervener
from verse.evolution.evidence import materialize_run_dir
from verse.runtime import swe_judge as _judge
from verse.runtime.executor import run_task as _swe_run_task


def run_task(inst, system_prompt, url, model, **kw):
    """Dispatch by task family. SWE instances go to executor.run_task (graded on the git diff);
    instances with task_family == 'tb' go to tb_env.run_tb_task (graded by the task's own
    run-tests.sh). Both share one signature and return value, so callers (sweep,
    confirm_flips, smoke check, fix_probe) need not know the family."""
    if (inst.get("task_family") or "").lower() == "tb":
        from verse.runtime.tb_env import run_tb_task
        kw.pop("scratch_repo", None)                  # SWE-only argument
        return run_tb_task(inst, system_prompt, url, model, **kw)
    return _swe_run_task(inst, system_prompt, url, model, **kw)


def _load_hooks_src(ws_dir: str) -> str | None:
    """The workspace's current harness_code/hooks.py source (None when absent or blank, as in
    markdown-only harnesses). Read fresh at each use: the workspace HEAD moves with
    apply/revert."""
    from verse.evolution.hooks_runtime import load_hooks_source
    return load_hooks_source(ws_dir)


# Data loading
def load_instances(parquet: str, n: int, seed: int) -> list:
    import pandas as pd
    df = pd.read_parquet(parquet)
    if n and n < len(df):
        df = df.sample(n=n, random_state=seed)
    insts = []
    for _, row in df.iterrows():
        ei = row["extra_info"]
        ei = ei if isinstance(ei, dict) else json.loads(ei)
        family = str(row.get("data_source", "") or "").lower()
        if family == "tb" or "tb_instance" in ei:
            ti = ei.get("tb_instance", ei)
            inst = dict(json.loads(ti) if isinstance(ti, str) else ti)
            inst["task_family"] = "tb"
        else:
            inst = _judge._as_instance(ei)
        if not (inst.get("problem_statement") or "").strip():
            try:
                user_msgs = [m["content"] for m in row["prompt"] if m.get("role") == "user"]
                inst["problem_statement"] = user_msgs[-1] if user_msgs else ""
            except Exception:
                pass
        if not (inst.get("problem_statement") or "").strip():
            raise RuntimeError(f"{inst.get('instance_id')}: no problem statement")
        insts.append(inst)
    return insts


# Sweeps
def sweep(insts: list, system_prompt: str, executor: dict, *, concurrency: int,
          traj_dir: str | None = None, label: str = "", successes: dict | None = None,
          nan_retries: int = 1, hooks_src: str | None = None,
          hooks_stats: dict | None = None) -> tuple:
    """Run every instance under the harness; returns (outcomes {tid: phi}, fail_transcripts).
    When `successes` is passed, passing transcripts are collected into it (success-side
    evidence: contrast pairs for every method, minimization and perturbation targets when
    the verification tools are configured).
    Tasks with a NaN outcome (infrastructure failure) are retried at low concurrency, with
    exponential backoff between retry passes. Train sweeps use nan_retries=1; val and test
    sweeps pass more, because a NaN left there scores as 0 downstream.
    hooks_src loads the executable harness code (harness_code/hooks.py) into every episode
    (None for markdown-only harnesses). hooks_stats (a mutable dict) accumulates per-episode
    hook statistics and the sweep's token usage, so the round journal can report the hook
    exception rate: hooks fail open, and their errors must still be visible."""
    outcomes, fails = {}, {}
    tok_total = {"calls": 0, "input": 0, "output": 0}   # token usage summed over the sweep
    if traj_dir:
        os.makedirs(traj_dir, exist_ok=True)

    def _collect_hooks(transcript):
        if hooks_stats is None or not hooks_src or transcript is None:
            return
        from verse.evolution.hooks_runtime import extract_hooks_summary
        hs = extract_hooks_summary(transcript)
        load_err = next((m.get("content") for m in transcript
                         if isinstance(m, dict) and isinstance(m.get("content"), str)
                         and m["content"].startswith("HARNESS_CODE_LOAD_ERROR")), None)
        hooks_stats["episodes"] = hooks_stats.get("episodes", 0) + 1
        if load_err:
            hooks_stats["load_errors"] = hooks_stats.get("load_errors", 0) + 1
            hooks_stats.setdefault("first_load_error", load_err[:400])
        if hs:
            if hs.get("n_errors"):
                hooks_stats["episodes_with_errors"] = \
                    hooks_stats.get("episodes_with_errors", 0) + 1
                for e in hs.get("errors") or []:
                    key = f"{e.get('hook')}: {e.get('error')}"
                    top = hooks_stats.setdefault("top_errors", {})
                    if len(top) < 8 or key in top:
                        top[key] = top.get(key, 0) + 1
            for k, v in (hs.get("changed") or {}).items():
                ch = hooks_stats.setdefault("changed", {})
                ch[k] = ch.get(k, 0) + v
            if hs.get("llm_calls"):
                hooks_stats["llm_calls"] = (hooks_stats.get("llm_calls", 0)
                                            + hs["llm_calls"])

    def _one(inst):
        tid = inst.get("instance_id", "?")
        # one episode runs on one thread, so the executor's thread-local token
        # counters measure exactly this episode
        from verse.runtime.executor import _tok_reset, _tok_read
        _tok_reset()
        try:
            phi, transcript = run_task(
                inst, system_prompt, "bedrock", executor["model"],
                max_turns=int(executor.get("max_turns", 100)),
                bedrock_region=executor.get("regions", "us-west-2,us-east-1,us-east-2"),
                hooks_src=hooks_src)
        except Exception as e:
            print(f"[evo] {label} {tid}: EPISODE ERROR {type(e).__name__}: {e}", flush=True)
            return tid, float("nan"), None, _tok_read()
        if traj_dir:
            try:
                with open(os.path.join(traj_dir, f"{tid}.json"), "w") as f:
                    json.dump({"instance_id": tid, "phi": phi, "transcript": transcript,
                               "tokens": _tok_read(), "ts": time.time()}, f)
            except Exception:
                pass
        return tid, phi, transcript, _tok_read()

    def _run(batch, workers):
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for fut in as_completed([pool.submit(_one, i) for i in batch]):
                tid, phi, tr, tok = fut.result()
                if phi == phi:
                    outcomes[tid] = phi
                    if phi < 0.5 and tr is not None:
                        fails[tid] = tr
                    elif phi >= 0.5 and tr is not None and successes is not None:
                        successes[tid] = tr
                _collect_hooks(tr)
                if tok and tok.get("calls"):
                    for k in ("calls", "input", "output"):
                        tok_total[k] += tok.get(k, 0)
                print(f"[evo] {label} {tid}: phi={phi}", flush=True)

    _run(insts, concurrency)
    backoff = 30.0
    for attempt in range(max(0, int(nan_retries))):
        retry = [i for i in insts if i.get("instance_id", "?") not in outcomes]
        if not retry:
            break
        if attempt:                       # first retry is immediate (transient throttling);
            time.sleep(backoff)           # later retries back off
            backoff = min(backoff * 2, 600.0)
        print(f"[evo] {label}: retry {attempt + 1}/{nan_retries} for "
              f"{len(retry)} infra tasks", flush=True)
        _run(retry, 2)
    still_nan = sorted(i.get("instance_id", "?") for i in insts
                       if i.get("instance_id", "?") not in outcomes)
    if still_nan:
        print(f"[evo] {label}: {len(still_nan)} tasks still NaN after retries: "
              f"{','.join(still_nan[:10])}", flush=True)
    n_pass = sum(1 for v in outcomes.values() if v >= 0.5)
    print(f"[evo] {label} sweep done: {n_pass}/{len(outcomes)} "
          f"tokens[in={tok_total['input']:,} out={tok_total['output']:,} "
          f"calls={tok_total['calls']}]", flush=True)
    if hooks_stats is not None:
        tk = hooks_stats.setdefault("tokens", {"calls": 0, "input": 0, "output": 0})
        for k in ("calls", "input", "output"):
            tk[k] += tok_total[k]
    return outcomes, fails


def confirm_flips(before: dict, after: dict, fails_after: dict, insts_by_tid: dict,
                  system_prompt: str, executor: dict, concurrency: int = 6,
                  hooks_src: str | None = None) -> list:
    """Rerun each flipped task once under the new harness. A flip counts only when the
    rerun agrees with it. A disagreeing rerun marks the task flaky: the same harness gave
    two outcomes, so the flip cannot be attributed to the round's edit. Flaky flips (and
    flips whose rerun hit an infrastructure failure) are reset to the before outcome and
    labeled, so the gate and the optimizer's round report treat them as noise. There is no
    tiebreak rerun: a majority vote over a flaky task adds cost without adding signal.
    Mutates after/fails_after; returns the confirmation log."""
    flips = [t for t in sorted(set(before) & set(after))
             if (before[t] < 0.5) != (after[t] < 0.5)]
    log = []
    if not flips:
        return log

    def _run_once(tid):
        try:
            phi2, tr2 = run_task(insts_by_tid[tid], system_prompt, "bedrock", executor["model"],
                                 max_turns=int(executor.get("max_turns", 100)),
                                 bedrock_region=executor.get("regions",
                                                             "us-west-2,us-east-1,us-east-2"),
                                 hooks_src=hooks_src)
            return phi2, tr2
        except Exception:
            return float("nan"), None

    def _one(tid):
        phi2, tr2 = _run_once(tid)
        infra = phi2 != phi2                    # NaN rerun: infrastructure, not evidence
        agree = (not infra) and ((phi2 >= 0.5) == (after[tid] >= 0.5))
        return tid, [phi2], agree, infra, tr2

    with ThreadPoolExecutor(max_workers=max(1, min(len(flips), concurrency))) as pool:
        for tid, runs, ok, infra, tr2 in pool.map(_one, flips):
            claim = ("fixed" if after[tid] >= 0.5 else "regressed") if ok else \
                    ("infra" if infra else "flaky")
            log.append({"tid": tid, "claim": claim,
                        "rerun_phis": [p if p == p else None for p in runs],
                        "confirmed": bool(ok)})
            if not ok:
                after[tid] = before[tid]
                if before[tid] < 0.5 and tr2 is not None:
                    fails_after[tid] = tr2
                elif before[tid] >= 0.5:
                    fails_after.pop(tid, None)
            print(f"[evo] confirm {tid}: {claim} reruns={runs} "
                  f"{'CONFIRMED' if ok else 'DISCARDED'}", flush=True)
    return log


# Smoke check and repair
def smoke_with_repair(proposal_payload: dict, edit_space, ws_dir: str, round_idx: int,
                      smoke_inst: dict, executor: dict, ctx, intervener_cfg: dict,
                      max_attempts: int = 3, claims: list | None = None) -> tuple:
    """Apply the proposal and validate it in stages. Each error is sent back to the
    optimizer to repair the same proposal (up to max_attempts tries):
      0. anti-cheat check (_cheat_audit: no task ids or gold-patch n-grams; off by default)
      1. claims consistency (edits that rest on refuted diagnoses are sent back; only the
         verification tools produce refutations)
      2. apply (edit-space constraints; for code_hooks this includes the static safety
         check of hooks.py), then assemble (the harness must not be empty)
      3. hooks load and test run (only when hooks.py exists: load it in the restricted
         namespace, instantiate it, and call every hook once on inert inputs; a hooks.py
         that cannot load is caught here instead of silently becoming a no-op in every
         episode)
      4. a one-task smoke episode, with the hooks loaded
    The smoke episode checks only validity and that the harness runs; it gives no effect
    signal, so this loop cannot be used to optimize against the gate. Returns
    (sha | None, smoke_log, final_payload); final_payload is the payload after any repairs,
    i.e. what was actually committed."""
    log = []
    payload = proposal_payload
    for attempt in range(1, max_attempts + 1):
        cheat = _cheat_audit(payload, ctx)
        if cheat:
            log.append({"attempt": attempt, "stage": "audit", "error": cheat[:400]})
            repaired = _repair(payload, cheat, ctx, intervener_cfg)
            if repaired is None:
                return None, log, payload
            payload = repaired
            continue
        conflict = _claims_conflict(payload, claims)
        if conflict:
            log.append({"attempt": attempt, "stage": "claims", "error": conflict[:400]})
            repaired = _repair(payload, conflict, ctx, intervener_cfg)
            if repaired is None:
                return None, log, payload
            payload = repaired
            continue
        base_sha = edit_space.head(ws_dir)
        try:
            sha = edit_space.apply(payload, ws_dir, round_idx)
        except Exception as e:
            log.append({"attempt": attempt, "stage": "apply", "error": str(e)[:300]})
            repaired = _repair(payload, f"apply failed: {e}", ctx, intervener_cfg)
            if repaired is None:
                return None, log, payload
            payload = repaired
            continue
        sp = edit_space.assemble(ws_dir)
        if not sp.strip():
            edit_space.reset_to(ws_dir, base_sha)
            log.append({"attempt": attempt, "stage": "assemble", "error": "empty harness"})
            repaired = _repair(payload, "assembled harness is empty", ctx, intervener_cfg)
            if repaired is None:
                return None, log, payload
            payload = repaired
            continue
        hooks_src = _load_hooks_src(ws_dir)
        if hooks_src:
            err = _hooks_dry_run(hooks_src)
            if err:
                edit_space.reset_to(ws_dir, base_sha)
                log.append({"attempt": attempt, "stage": "hooks_load", "error": err[:400]})
                repaired = _repair(payload, f"harness_code/hooks.py failed to load/dry-run "
                                            f"(fix the code, keep the mechanism): {err}",
                                   ctx, intervener_cfg)
                if repaired is None:
                    return None, log, payload
                payload = repaired
                continue
        try:
            phi, _ = run_task(smoke_inst, sp, "bedrock", executor["model"], max_turns=8,
                              bedrock_region=executor.get("regions",
                                                          "us-west-2,us-east-1,us-east-2"),
                              hooks_src=hooks_src)
            crashed = phi != phi
        except Exception as e:
            crashed = True
        if not crashed:
            log.append({"attempt": attempt, "stage": "smoke", "ok": True})
            return sha, log, payload
        edit_space.reset_to(ws_dir, base_sha)
        log.append({"attempt": attempt, "stage": "smoke", "error": "episode crashed/infra"})
        repaired = _repair(payload, "harness crashed the smoke episode", ctx, intervener_cfg)
        if repaired is None:
            return None, log, payload
        payload = repaired
    return None, log, payload


def _hooks_dry_run(hooks_src: str) -> str:
    """Smoke stage 3: load hooks.py in the restricted namespace and call every hook once on
    inert inputs. Returns an error message for the optimizer, or '' when clean. This
    catches constructor crashes and immediate hook bugs before any episode runs; exceptions
    raised during real episodes still fail open and are recorded in the hook error log."""
    from verse.evolution.hooks_runtime import HookedRuntime, HookLoadError
    try:
        rt = HookedRuntime(hooks_src)
    except HookLoadError as e:
        return str(e)
    rt.loop_config()                    # invalid loop settings are recorded in rt.errors
    rt.system_prompt("dry-run system prompt")
    rt.before_llm([{"role": "user", "content": "dry-run"}], 0)
    rt.after_llm([{"type": "text", "text": "dry-run reply"}], 0)
    rt.before_tool("bash", {"command": "true"}, 0)
    rt.after_tool("bash", {"command": "true"}, "dry-run output", 0)
    rt.extra_tool_specs()
    rt.on_turn_end(0)
    if rt.errors:
        return ("hooks raised on trivial inputs (each would silently degrade to identity "
                "in real episodes): "
                + "; ".join(f"{e['hook']}: {e['error']}" for e in rt.errors[:5]))
    return ""


_TASKID_RE = re.compile(r"[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-\d+")


def _cheat_audit(payload: dict, ctx) -> str:
    """Smoke stage 0: anti-cheat check over the change-set. Two checks:
      1. no train/val task ids inside any edit (a harness keyed to specific task ids
         memorizes instead of generalizing; task ids differ at test time);
      2. no 13-gram overlap between edit content and the gold patch of any loaded train or
         val task (executable harness code could otherwise hand answers to the executor,
         which runs without network access).
    Returns a message for the optimizer, or ''.

    Off by default: the train, val and test splits are disjoint (val is repo-disjoint, test
    is a separate pinned set), so test scores do not depend on this check.
    Train-derived examples in a harness are ordinary in-context learning, which none of the
    four baselines forbid. Enable with T2E_CHEAT_AUDIT=1, e.g. for an experiment whose
    splits overlap."""
    if os.environ.get("T2E_CHEAT_AUDIT", "0") != "1":
        return ""
    known = set((ctx.insts_by_tid or {}).keys())
    texts = [(e.get("path", ""), e.get("content", "") or "")
             for e in (payload.get("edits") or []) if not e.get("delete")]
    hits = []
    for path, content in texts:
        ids = {m for m in _TASKID_RE.findall(content) if m in known}
        if ids:
            hits.append(f"- {path}: contains task id(s) {sorted(ids)[:4]} — harness edits "
                        "must be GENERAL (task ids rotate; memorized answers do not transfer)")
    def _grams(s, n=13):
        toks = s.split()
        return {" ".join(toks[i:i + n]) for i in range(max(0, len(toks) - n + 1))}
    gold_grams = set()
    for tid, inst in (ctx.insts_by_tid or {}).items():
        g = inst.get("patch") or inst.get("gold_patch") or ""
        if g:
            gold_grams |= _grams(g)
    if gold_grams:
        for path, content in texts:
            if _grams(content) & gold_grams:
                hits.append(f"- {path}: shares a 13-gram with a task's gold patch — do not "
                            "embed task solutions in the harness")
    if not hits:
        return ""
    return ("ANTI-CHEAT AUDIT FAILED — your change-set was not applied:\n"
            + "\n".join(hits))


def _claims_conflict(payload: dict, claims: list) -> str:
    """Check the change-set against this round's experiment evidence. Two checks, both
    possible only when the verification tools are configured (without them no claim is
    ever refuted, so nothing is blocked; there is no separate code path):
      1. a predicted fix for a task whose diagnosed root cause was refuted by an experiment
         (e.g. the replay did not reproduce the failure) rests on a disproven diagnosis;
      2. a fix_probe run on this exact change-set (matched by edits_sha) fixed none of its
         tested targets, yet those targets are still in predicted_fixes. A revised
         change-set has a new sha and is not blocked by refutations of an earlier one.
    Returns a repair message, or ""."""
    from verse.evolution.probes import edits_sha
    refuted = {}
    fp_refuted = []                # (tested_tids, record) for fix_probe refutes on this sha
    sha_now = edits_sha(payload.get("edits") or [])
    for c in claims or []:
        for e in getattr(c, "evidence", []):
            if getattr(e, "verdict", "") != "refutes":
                continue
            if getattr(e, "kind", "") == "fix_probe":
                d = getattr(e, "detail", {}) or {}
                if d.get("edits_sha") == sha_now:
                    fp_refuted.append((d.get("tested") or [], e))
            elif getattr(c, "task_id", ""):
                refuted[c.task_id] = (c, e)
    lines = []
    hits = [t for t in payload.get("predicted_fixes", []) if t in refuted]
    for t in hits:
        c, e = refuted[t]
        lines.append(f"- {t}: diagnosis was «{c.statement[:160]}» but experiment[{e.kind}] "
                     f"showed: {e.outcome[:160]}")
    for tested, e in fp_refuted:
        still = [t for t in payload.get("predicted_fixes", []) if t in tested]
        if still:
            lines.append(f"- fix_probe ran YOUR CURRENT edits and they fixed none of "
                         f"{', '.join(still)}: {e.outcome[:160]}")
    if not lines:
        return ""
    return ("CONSISTENCY CHECK FAILED: your change-set conflicts with real experiment "
            "results from this round:\n" + "\n".join(lines) +
            "\nFor each: either revise the edits, ground them in evidence that was NOT "
            "refuted, or drop the task from predicted_fixes. Keep every edit that rests "
            "on verified or untested evidence.")


def _repair(payload: dict, error: str, ctx, intervener_cfg: dict) -> dict | None:
    """One repair turn: send the rejected payload and the error back to the optimizer model."""
    from verse.runtime.executor import _invoke_bedrock, _json_candidates, _TEMP_DEPRECATED
    model = intervener_cfg["model"]
    body = {"max_tokens": 8192,
            "messages": [{"role": "user", "content":
                          "Your harness change-set failed a validity check and was NOT applied.\n"
                          f"Error: {error}\n\nOriginal change-set JSON:\n{json.dumps(payload)[:12000]}\n\n"
                          "Reply with ONLY the corrected change-set JSON (same schema: "
                          '{"description", "edits": [{"path", "content"}], "predicted_fixes", "at_risk"}).'}]}
    if not any(t in model for t in _TEMP_DEPRECATED):
        body["temperature"] = 0.0
    try:
        reply = _invoke_bedrock(model, intervener_cfg.get("regions",
                                                          "us-west-2,us-east-1,us-east-2"), body)
        for cand in _json_candidates(reply):
            try:
                d = json.loads(cand)
                if isinstance(d, dict) and d.get("edits"):
                    return d
            except Exception:
                continue
    except Exception as e:
        print(f"[evo] repair call failed: {e}", flush=True)
    return None


# Prompt assembly
# The task brief has two versions that differ only in the branching paragraph:
#   _TASK_BRIEF (configurations without meta_teacher) includes a hand-written strategy for
#     when to build forward and when to branch back.
#   _TASK_BRIEF_SELF (configurations with meta_teacher) states only how branching works.
#     When to branch is left for the self-evolving optimizer to learn and record in its
#     own harness.
_BRIEF_HEAD = """You are the harness engineer for a frozen coding agent (the "executor"). The
executor failed a number of software tasks under its current harness. Your job this round:
diagnose WHY it fails and improve the harness so it succeeds more.

## The harness you may edit (submit via submit_proposal)
Its current files are under workspace/ in the run directory. Components:
  system_prompt.md | workflow.md | tool_notes/*.md | skills/*.md | middleware/*.md |
  subagents/*.md | memory.md{code_component}
Your edits must be GENERAL: task ids rotate; memorized answers will not transfer. Every
round's edit is applied and measured on held-back validation tasks; the best round wins.
"""
_BRIEF_TAIL = """{code_section}
## Your notes from previous rounds
{notes}

## Your previous proposals and outcomes
{history}

## Evidence
{evidence}
"""
_BRANCH_WISDOM = """Branching: submit_proposal accepts base_round=N — your edits then apply on round N's state
instead of the current head (0 = original harness). Before you invest this round, look at
the val trajectory in history: if the recent kept rounds form a net LOSS (e.g. two HARMFUL
in a row, or val well below its earlier peak), the current head is damaged ground — building
on it means your fix competes with someone else's regression. In that situation the usually
better move is base_round=<the last round where val peaked> and apply this round's idea
there. Branching costs nothing; un-editing a bad change by hand costs your whole round.
Rule of thumb: repair forward when the damage IS your target; branch back when the damage
is unrelated to what you plan to fix.
"""
_BRANCH_CONTRACT = """Branching: submit_proposal accepts base_round=N — your edits then apply on round N's state
instead of the current head (0 = original harness). Only kept rounds (or 0) are valid bases.
"""
_TASK_BRIEF = _BRIEF_HEAD + _BRANCH_WISDOM + _BRIEF_TAIL
_TASK_BRIEF_SELF = _BRIEF_HEAD + _BRANCH_CONTRACT + _BRIEF_TAIL

# Guidance on choosing markdown or code for each fix, adapted from the escalation rule in
# AHE's evolve prompt: prompt-level fixes cannot repair mechanical failure modes.
_CODE_SECTION = """
## Choosing the layer for each fix (markdown vs code)
For EVERY failure class you attack, decide the layer deliberately — executable code is a
first-class option, not a last resort:
- Instructional gaps (the model didn't know WHAT to do) -> markdown (skills/workflow).
- Mechanical failure modes (malformed output, noisy/oversized observations, missing error
  structure, repeated manual workflows) -> harness_code/hooks.py. An instruction "please
  format tool names correctly" is advisory; an after_llm/after_tool transform is 100%
  mechanical. If the same failure class survived a markdown fix in an earlier round, that
  is the SIGNAL to re-fix it in code. The single highest-yield code fix for weak executors
  is usually after_llm PARSE-RESCUE: find trajectories where the model emitted tool-call
  markup as plain TEXT (XML tags, DSML, json blobs) and the turn died — a hook that parses
  that markup into a real tool_use block converts dead turns into actions.
Before writing hooks.py, read executor_source/ in the run dir — the actual loop your
hooks mount into (executor.py) and the hook runtime contract (hooks_runtime.py). Code
written against the real call sites beats code written against the API doc alone.
{hooks_api}
"""

_VERIFIED_CODE_RULE = """
## Code-edit discipline (you have fix_probe — use it)
Code edits are high-variance: one wrong after_tool can truncate every observation; one
right sanitize mechanically kills a whole failure class. NEVER submit a hooks.py change
without a fix_probe run on it: the probe mounts your draft code on real episodes and
returns per-hook call counts and exception tracebacks alongside the flip results. A
0-flip, 0-exception probe means your code ran clean but changed nothing that matters —
revise the mechanism, not just the syntax."""

# Code section for configurations with meta_teacher: the hook API contract without the
# layer-selection guidance (parse-rescue recipe, markdown-versus-code rule). Choosing the
# layer is left to the self-evolving optimizer.
_CODE_SECTION_SELF = """
## The executable layer (harness_code/hooks.py)
Besides the markdown components you may write executable Python that runs inside the
executor's agent loop. Submit it as a normal edit with path `harness_code/hooks.py`.
Reference: executor_source/ in the run dir contains the actual loop it mounts into
(executor.py) and the hook runtime contract (hooks_runtime.py).
{hooks_api}
"""


def _history_text(history: list, max_entries: int = 8) -> str:
    if not history:
        return "(first round)"
    lines = []
    stale = []                    # failure mechanisms already attacked without effect
    for h in history[-max_entries:]:
        v = h.get("verdict") or {}
        size = (f" [{len(h.get('files') or [])} files/{h['edit_bytes']:,}B]"
                if h.get("edit_bytes") is not None else "")
        lines.append(f"- round {h.get('round')}: {h.get('description', '')[:120]}{size} -> "
                     f"{v.get('label')} kept={h.get('kept')} ({v.get('reason', '')}); "
                     f"predicted {h.get('predicted_fixes', [])[:6]}")
        pa = h.get("prediction_attribution") or {}
        if pa:
            from collections import Counter as _C
            cnt = _C(pa.values())
            regs = [t for t, o in pa.items() if o == "regressed"]
            lines.append(f"    prediction outcome: {dict(cnt)}"
                         + (f" — REGRESSED: {', '.join(regs[:3])}" if regs else ""))
        hs = h.get("hooks_stats") or {}
        if hs.get("error_rate"):
            top = "; ".join(list((hs.get("top_errors") or {}).keys())[:2])
            lines.append(f"    HARNESS CODE: exceptions in {hs['error_rate']:.0%} of val "
                         f"episodes{' — ' + top if top else ''}"
                         + (" [HOOKS_BROKEN — round excluded from gains]"
                            if h.get("hooks_broken") else ""))
        if hs.get("first_load_error"):
            lines.append(f"    HARNESS CODE: {hs['first_load_error'][:200]}")
        if v.get("label") in ("INERT", "INEFFECTIVE") or (not h.get("kept")):
            stale.append(h.get("description", "")[:80])
    lines.append("Rounds are precious. Build on what worked (EFFECTIVE).")
    if stale:
        lines.append("These ideas were already tried and did NOT improve validation — attack a "
                     "DIFFERENT failure mechanism instead of rephrasing them:")
        lines += [f"  x {s}" for s in stale[-4:]]
    return "\n".join(lines)


# Candidate generation and screening
# Best-of-N candidate generation. Two sources of diversity:
#   - Candidate guidance (_LENSES, used when probes are configured): each of the N
#     candidates focuses on a different failure class (omission, commission, mechanics).
#   - Sampling (_SAMPLE_LENSES, the baselines): as in AHE (N worktrees on the same prompt)
#     and Self-Harness (K proposals from the same evidence), all N candidates get the same
#     generic suffix and diversity comes from sampling alone. Names stay distinct for the
#     journal.
# Configurations with meta_teacher take their guidance from search.md instead (see
# run_evolution).
_LENSES = [
    ("omission", "\n\n## Your lens this round\nAttack OMISSION failures: cases where the "
     "executor never produced the needed behavior (missing exploration, missing "
     "verification, missing fix). Teach the missing behavior at the right layer."),
    ("commission", "\n\n## Your lens this round\nAttack COMMISSION failures: cases where a "
     "specific wrong action sank the episode (bad edits, destructive commands, wrong "
     "submit timing). Ban or replace the causal action at the right layer."),
    ("mechanics", "\n\n## Your lens this round\nAttack MECHANICAL failure modes: tool-call "
     "formatting, observation noise/truncation, context bloat, error messages the model "
     "misreads. Prefer the executable code layer (harness_code/hooks.py) when available — "
     "these modes are exactly what instructions cannot reliably fix."),
]
_SAMPLE_SUFFIX = ("\n\n## Candidate generation\nPropose the highest-impact harness "
                  "change-set you can justify from the evidence this round.")
_SAMPLE_LENSES = [("sample_a", _SAMPLE_SUFFIX), ("sample_b", _SAMPLE_SUFFIX),
                  ("sample_c", _SAMPLE_SUFFIX)]


def _proposal_payload(proposal) -> dict:
    return {"description": proposal.description,
            "edits": proposal.meta.get("edits", []),
            "predicted_fixes": proposal.predicted_fixes, "at_risk": proposal.at_risk}


def _probe_confirmed_flips(prop) -> int:
    """Screening tie-breaker: the number of train tasks that fix_probe confirmed this
    candidate's final change-set flips (matched by edits_sha; the probe itself already
    confirms flips best-of-2). Without probes every candidate scores 0 and the order falls
    back to candidate order. Records of draft change-sets that were probed and then
    discarded are excluded, since they do not describe the submitted edits."""
    from verse.evolution.probes import edits_sha
    sha = edits_sha((prop.meta or {}).get("edits") or [])
    flips = set()
    for c in prop.claims or []:
        if (getattr(c, "tags", None) or {}).get("exploration"):
            continue
        for e in getattr(c, "evidence", []) or []:
            if getattr(e, "kind", "") != "fix_probe":
                continue
            d = getattr(e, "detail", None) or {}
            if d.get("edits_sha") == sha:
                flips.update(d.get("flips") or [])
    return len(flips)


def _screen_candidates(cands: list, edit_space, ws_dir: str, rdir: str, rnd: int,
                       train_out: dict, insts_by_tid: dict, executor: dict,
                       screen_n: int, seen_shas: dict) -> tuple:
    """Best-of-N selection: run every candidate's change-set on one shared train subset and
    return (winner_index, screen_log). The subset is the union of the candidates' predicted
    fixes, padded with other failing tasks, plus two passing tasks as a regression check;
    all candidates run on the same tasks. Screening is a driver mechanism, like the gate,
    so every method gets the same search width. Cost is about N * screen_n episodes (about
    0.7 val sweeps for N=3, screen_n=12).
    Ranking: net screen score, then probe-confirmed flips, then candidate order. The screen
    subset is small and a weak executor rarely flips tasks, so ties are common; breaking
    them by candidate order alone would reduce best-of-N to best-of-1."""
    from verse.evolution.probes import edits_sha
    if len(cands) == 1:
        return 0, [{"note": "single candidate — no screening"}]
    fails_now = sorted(t for t, v in train_out.items() if v < 0.5)
    passes_now = sorted(t for t, v in train_out.items() if v >= 0.5)
    tids = []
    for _lens, prop in cands:
        for t in prop.predicted_fixes:
            if t in insts_by_tid and train_out.get(t, 1.0) < 0.5 and t not in tids:
                tids.append(t)
    for t in fails_now:
        if len(tids) >= max(2, screen_n - 2):
            break
        if t not in tids:
            tids.append(t)
    tids = tids[:screen_n - 2] + passes_now[:2]        # 2 passing tasks: regression check
    screen_insts = [insts_by_tid[t] for t in tids if t in insts_by_tid]
    log, scores = [], []
    for idx, (lens, prop) in enumerate(cands):
        payload = _proposal_payload(prop)
        sha = edits_sha(payload.get("edits") or [])
        prior = seen_shas.get(sha)
        if prior and prior.get("dead"):
            log.append({"lens": lens, "sha": sha, "score": None,
                        "error": f"duplicate of round {prior['round']} "
                                 f"({prior['label']}) — disqualified"})
            scores.append((float("-inf"), 0, -idx))
            continue
        scratch = os.path.join(rdir, f"screen_{lens}")
        shutil.rmtree(scratch, ignore_errors=True)
        shutil.copytree(ws_dir, scratch, ignore=shutil.ignore_patterns(".git"))
        from verse.workspace import HarnessWorkspace7
        try:
            HarnessWorkspace7(scratch).init_blank()          # scratch needs its own git
            edit_space.apply(payload, scratch, rnd)
            sp = edit_space.assemble(scratch)
            hooks = _load_hooks_src(scratch)
        except Exception as e:
            log.append({"lens": lens, "sha": sha, "score": None,
                        "error": f"apply failed: {str(e)[:200]}"})
            scores.append((float("-inf"), 0, -idx))
            shutil.rmtree(scratch, ignore_errors=True)
            continue
        out, _fails = sweep(screen_insts, sp, executor, concurrency=min(6, len(screen_insts)),
                            label=f"r{rnd}-screen-{lens}", hooks_src=hooks)
        shutil.rmtree(scratch, ignore_errors=True)
        fixed = sum(1 for t, p in out.items() if p >= 0.5 and train_out.get(t, 1.0) < 0.5)
        regressed = sum(1 for t, p in out.items() if p < 0.5 and train_out.get(t, 0.0) >= 0.5)
        net = fixed - regressed
        confirmed = _probe_confirmed_flips(prop)
        log.append({"lens": lens, "sha": sha, "screen_tids": tids,
                    "fixed": fixed, "regressed": regressed, "score": net,
                    "probe_confirmed_flips": confirmed})
        # ties on net: most probe-confirmed flips wins, then the earlier candidate
        scores.append((net, confirmed, -idx))
    best = max(range(len(scores)), key=lambda i: scores[i])
    if scores[best][0] == float("-inf"):
        best = 0                                        # all failed; smoke repair handles it
    return best, log


def _frontier_section(history: list) -> str:
    """Per-task val frontier: which val tasks were fixed or lost in which kept round (as in
    Meta-Harness's frontier_val). Makes regressions visible to the optimizer."""
    events = {}
    for h in history:
        if not h.get("kept"):
            continue
        v = h.get("verdict") or {}
        for t in v.get("fixed", []):
            events.setdefault(t, []).append((h["round"], "FIXED"))
        for t in v.get("regressed", []):
            events.setdefault(t, []).append((h["round"], "LOST"))
    if not events:
        return ""
    lines = ["\n\n## Val task frontier (state changes across kept rounds)",
             "Tasks marked LOST after FIXED are regressions a later edit caused — defend "
             "won ground; do not re-break it."]
    for t in sorted(events)[:30]:
        lines.append(f"- {t}: " + " -> ".join(f"r{r} {e}" for r, e in events[t]))
    return "\n".join(lines)


def _efficiency_section(fail_trs: dict, succ_trs: dict) -> str:
    """Turns-per-episode summary (as in Meta-Harness's rollout_metrics): mean turns for
    passing and failing tasks, and the slowest passes. A fix that also cuts turns is worth
    more under the episode wall-clock cap."""
    def _turns(tr):
        return sum(1 for m in tr or [] if isinstance(m, dict) and m.get("role") == "assistant")
    lines = []
    if succ_trs:
        pt = sorted(((_turns(tr), t) for t, tr in succ_trs.items()), reverse=True)
        vals = [n for n, _ in pt]
        lines.append(f"- passing: mean {sum(vals) / len(vals):.0f} turns; slowest "
                     + ", ".join(f"{t} ({n})" for n, t in pt[:5]))
    if fail_trs:
        fv = [_turns(tr) for tr in fail_trs.values()]
        if fv:
            lines.append(f"- failing: mean {sum(fv) / len(fv):.0f} turns before cap/give-up")
    if not lines:
        return ""
    return ("\n\n## Efficiency (turns per episode, this train sweep)\n" + "\n".join(lines)
            + "\nEpisode wall-clock is capped: fixes that also cut wasted turns compound.")


def _seen_shas(history: list) -> dict:
    """{edits_sha: {round, label, dead}} over journaled rounds. dead=True marks change-sets
    that were reverted or judged INERT/INEFFECTIVE. The novelty check (from HarnessX) uses
    it to reject byte-identical re-proposals mechanically."""
    seen = {}
    for h in history:
        sha = h.get("edits_sha")
        if not sha:
            continue
        label = ((h.get("verdict") or {}).get("label") or "").upper()
        dead = (not h.get("kept")) or label in ("INERT", "INEFFECTIVE")
        seen[sha] = {"round": h.get("round"), "label": label or "UNGATED", "dead": dead}
    return seen


# The round loop
def run_evolution(cfg: dict) -> None:
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    journal = os.path.join(out_dir, "rounds.jsonl")

    evidence_src = build_component("evidence", cfg["evidence"])
    probes = build_component("probes", cfg.get("probes"))
    # probe-selection ablation: a fixed policy (random or rule-based) spends the probe
    # budget instead of the optimizer; the optimizer episode only reads the results
    probe_scheduler = build_component("probe_scheduler", cfg.get("probe_scheduler"))
    if probe_scheduler is not None and probes is None:
        raise ValueError("probe_scheduler requires a configured 'probes' block")
    edit_space = build_component("edit_space", cfg["edit_space"])
    gate = build_component("gate", cfg["gate"])
    executor = dict(cfg["executor"])
    intervener_cfg = dict(cfg["intervener"])
    executor["intervener_model"] = intervener_cfg["model"]
    sweep_cfg = cfg.get("sweep") or {}
    conc = int(sweep_cfg.get("concurrency", 12))
    # optimizer self-evolution: the optimizer also edits its own harness. None for every
    # other method, which makes all meta_teacher steps below no-ops.
    meta_teacher = None
    if cfg.get("meta_teacher") is not None:
        from verse.evolution.meta_teacher import MetaTeacher
        meta_teacher = MetaTeacher(cfg.get("meta_teacher") or {}, out_dir,
                                   intervener_cfg, executor)

    data = cfg["data"]
    train_insts = load_instances(data["train_parquet"], int(data.get("train_tasks", 100)),
                                 int(data.get("seed", 42)))
    val_insts = load_instances(data["val_parquet"], int(data.get("val_tasks", 50)),
                               int(data.get("seed", 42)))
    insts_by_tid = {i.get("instance_id", "?"): i for i in train_insts + val_insts}
    smoke_inst = train_insts[0]

    ws_dir = os.path.join(out_dir, "workspace")
    # base_prompt: an explicit string is used as is; "TB" selects the Terminal-Bench seed
    # prompt; the default is the SWE (/testbed) prompt. On TB tasks the SWE prompt can make
    # some executors issue no bash commands at all, which leaves no evidence.
    bp = cfg.get("base_prompt")
    if bp == "TB":
        bp = __import__("verse.runtime.tb_env", fromlist=["TB_BASE_PROMPT"]).TB_BASE_PROMPT
    elif not bp:
        bp = __import__("verse.runtime.executor", fromlist=["BASE_PROMPT"]).BASE_PROMPT
    edit_space.init(ws_dir, base_prompt=bp)

    # Resume: replay the journal.
    history, round0 = [], None
    if os.path.exists(journal):
        for ln in open(journal):
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            if rec.get("round") == 0:
                round0 = rec
            elif isinstance(rec.get("round"), int):
                history.append(rec)          # skips the "final" record of a finished run
    start_round = (history[-1]["round"] + 1) if history else 1
    # The previous process may have died mid-round, after committing a proposal the gate
    # never judged. Reset to the journal's last ws_head so ungated edits never enter the
    # lineage. Error and degraded round records carry no ws_head, so walk back to the
    # newest record that has one (otherwise the reset would go to the round-0 seed and
    # drop every kept edit).
    last_head = next((h.get("ws_head") for h in reversed(history) if h.get("ws_head")),
                     None) or (round0.get("ws_head") if round0 else None)
    if last_head and edit_space.head(ws_dir) != last_head:
        print(f"[evo] resume: workspace at {edit_space.head(ws_dir)[:8]} != journaled "
              f"{last_head[:8]} — resetting orphan un-gated edits", flush=True)
        edit_space.reset_to(ws_dir, last_head)

    # Round 0: train and val sweeps of the initial harness.
    if round0 is None:
        sp0 = edit_space.assemble(ws_dir)
        hooks0 = _load_hooks_src(ws_dir)
        tr_out, tr_fails = sweep(train_insts, sp0, executor, concurrency=conc,
                                 traj_dir=os.path.join(out_dir, "round_00", "traj_train"),
                                 label="r0-train", hooks_src=hooks0)
        va_out, _ = sweep(val_insts, sp0, executor, concurrency=conc, label="r0-val",
                          nan_retries=4, hooks_src=hooks0)
        round0 = {"round": 0, "train_outcomes": tr_out, "val_outcomes": va_out,
                  "ws_head": edit_space.head(ws_dir), "ts": time.time()}
        open(journal, "a").write(json.dumps(round0) + "\n")
    train_out = {k: float(v) for k, v in round0["train_outcomes"].items()}
    val_out = {k: float(v) for k, v in round0["val_outcomes"].items()}
    # On resume, the round-0 outcomes describe the initial harness, and every kept round
    # since changed them. Replay the kept verdicts' flips onto val_out (phi is binary, so
    # fixed/regressed reconstruct the val outcomes exactly) and rebuild train_out from the
    # newest train sweep's trajectory files. Otherwise a restarted driver would gate round N
    # against round 0 and count every earlier fix again.
    for h in history:
        if not h.get("kept"):
            continue
        v = h.get("verdict") or {}
        for t in v.get("fixed", []):
            val_out[t] = 1.0
        for t in v.get("regressed", []):
            val_out[t] = 0.0
    train_out = _reload_train_out(out_dir, history) or train_out
    # transcripts for the current harness state: reload from the newest train sweep
    train_fails = _reload_fails(out_dir, history, train_out)
    train_succ = _reload_successes(out_dir, history, train_out)
    # If error rounds followed a kept round, that kept edit never got its train sweep.
    # Force one when the newest train sweep on disk predates the last kept round.
    kept_rounds = [h["round"] for h in history if h.get("kept")]
    _nd = _newest_kept_traj_dir(out_dir, history)
    _nd_round = int(os.path.basename(os.path.dirname(_nd)).split("_")[1]) if _nd else 0
    force_train_sweep = bool(kept_rounds) and kept_rounds[-1] >= _nd_round

    for rnd in range(start_round, int(cfg.get("rounds", 6)) + 1):
        rdir = os.path.join(out_dir, f"round_{rnd:02d}")
        os.makedirs(rdir, exist_ok=True)
        rec = {"round": rnd, "ts": time.time()}

        # 1. Train sweep, only after a kept edit or after two consecutive error/INVALID
        # rounds. Failed rounds do not refresh trajectories, so the optimizer would re-read
        # identical evidence and tend to repeat the same failure; a fresh sweep breaks that
        # loop.
        two_dead = (len(history) >= 2
                    and all(not h.get("kept") for h in history[-2:])
                    and any(h.get("error") or (h.get("verdict") or {}).get("label")
                            == "INVALID" for h in history[-2:]))
        if (history and history[-1].get("kept")) or force_train_sweep or two_dead:
            force_train_sweep = False
            if two_dead and not (history[-1].get("kept")):
                print(f"[evo] round {rnd}: refreshing train sweep after consecutive "
                      "dead rounds (stale-evidence loop breaker)", flush=True)
            sp = edit_space.assemble(ws_dir)
            train_succ = {}
            prev_train_out = dict(train_out)      # pre-edit outcomes for attribution
            train_out, train_fails = sweep(
                train_insts, sp, executor, concurrency=conc,
                traj_dir=os.path.join(rdir, "traj_train"), label=f"r{rnd}-train",
                successes=train_succ, hooks_src=_load_hooks_src(ws_dir))
            # Check the kept proposal's predicted fixes against this sweep, only when the
            # newest round was kept (a refresh after failed rounds has no edit to check).
            # Each prediction gets an outcome label, as in HarnessX's
            # journal.compute_attribution (flipped, still_T, still_F, regressed, absent), so a
            # predicted fix that caused a regression or has no outcome stays visible.
            if history and history[-1].get("kept"):
                prev = history[-1]
                prev_out = prev_train_out or {}
                attribution = {}
                for t in prev.get("predicted_fixes", []):
                    if t not in train_out or t not in prev_out:
                        attribution[t] = "absent"
                    elif train_out[t] >= 0.5:
                        attribution[t] = "flipped" if prev_out[t] < 0.5 else "still_T"
                    else:
                        attribution[t] = "regressed" if prev_out[t] >= 0.5 else "still_F"
                rec["predicted_fixes_realized"] = [
                    t for t, o in attribution.items() if o == "flipped"]
                rec["prediction_attribution"] = attribution
        rec["train_score"] = f"{sum(1 for v in train_out.values() if v >= 0.5)}/{len(train_out)}"

        # 2. Evidence and optimizer episodes.
        ctx = EvolutionContext(
            round_idx=rnd, ws_dir=ws_dir, traj_dir=os.path.join(rdir, "traj_train"),
            train_outcomes=train_out, insts_by_tid=insts_by_tid,
            train_tids=frozenset(i.get("instance_id", "?") for i in train_insts),
            fail_transcripts=train_fails, success_transcripts=train_succ,
            history=history, probes=probes,
            scratch_dir=rdir, executor=executor)
        # probe_tools: false keeps the probes configured (the attribution layer uses them)
        # but withholds the four probe tools from the optimizer episode
        ctx.extra["probe_tools"] = cfg.get("probe_tools", True) is not False
        if probe_scheduler is not None:
            ctx.extra["probe_scheduler"] = probe_scheduler
        if probes is not None:
            probes.reset_round()
        # copy the executor harness into the run dir for the optimizer to read
        ws_mirror = os.path.join(rdir, "workspace")
        shutil.rmtree(ws_mirror, ignore_errors=True)
        shutil.copytree(ws_dir, ws_mirror, ignore=shutil.ignore_patterns(".git"))
        # code_hooks: also copy the executor source (executor.py, hooks_runtime.py), so the
        # optimizer writes hooks.py against the real call sites. AHE and HarnessX likewise
        # give their optimizers framework source. It is read with the existing file tools.
        if (cfg.get("edit_space") or {}).get("kind") == "code_hooks":
            src_dir = os.path.join(rdir, "executor_source")
            os.makedirs(src_dir, exist_ok=True)
            import verse.runtime.executor as _exmod
            import verse.evolution.hooks_runtime as _hrmod
            for mod, name in ((_exmod, "executor.py"), (_hrmod, "hooks_runtime.py")):
                try:
                    shutil.copyfile(mod.__file__, os.path.join(src_dir, name))
                except Exception:
                    pass
        with open(os.path.join(rdir, "history.json"), "w") as f:
            json.dump([{k: v for k, v in h.items() if k != "val_outcomes_full"}
                       for h in history[-8:]], f, indent=1)

        n_cand_cfg = int((cfg.get("search") or {}).get("n_candidates", 1))
        if probes is not None and n_cand_cfg > 1:
            # the probe budget is per optimizer episode: N parallel candidates share one
            # probes object, so scale the round pool to keep each episode's budget unchanged
            if not hasattr(probes, "_base_budget"):
                probes._base_budget = probes.budget
            probes.budget = probes._base_budget * n_cand_cfg
        try:
            report, ev_claims = evidence_src.gather(ctx)
            # Attribution layer: enabled by an `attribute:` config section, and only when
            # probes are configured. It runs outside the probe budget, so every round gets
            # the same diagnosis coverage regardless of how the optimizer spends its probes.
            if probes is not None and cfg.get("attribute") is not None:
                from verse.evolution.attribute import run_attribute_layer, update_ledger
                attr_cfg = cfg.get("attribute") or {}
                # novelty first: failure signatures already in the ledger were minimized in
                # an earlier round, so new failure modes get the minimization slots
                seen_sigs = set()
                try:
                    with open(os.path.join(out_dir, "ledger.json")) as lf:
                        seen_sigs = set(json.load(lf).keys())
                except Exception:
                    pass
                try:
                    attr_report, attr_claims, fingerprints = run_attribute_layer(
                        ctx, max_fail=int(attr_cfg.get("max_fail", 4)),
                        max_success=int(attr_cfg.get("max_success", 1)),
                        wall_s=float(attr_cfg.get("wall_s", 1500)),
                        seen_sigs=seen_sigs)
                except Exception as e:
                    print(f"[evo] attribute layer failed (non-fatal): {e}", flush=True)
                    attr_report, attr_claims, fingerprints = "", [], []
                # Training audit: register new failure fingerprints and record which known
                # ones recur in this sweep. It feeds the optimizer's report only, never the
                # gate or the final selection.
                try:
                    ledger_sec = update_ledger(os.path.join(out_dir, "ledger.json"), rnd,
                                               train_fails, fingerprints)
                except Exception as e:
                    print(f"[evo] ledger update failed (non-fatal): {e}", flush=True)
                    ledger_sec = ""
                # attribute.ledger: false withholds the training-audit section from the
                # report; ledger.json keeps updating, so the novelty-first choice is unaffected
                if attr_cfg.get("ledger", True) is False:
                    ledger_sec = ""
                report = report + attr_report + ledger_sec
                ev_claims = list(ev_claims) + attr_claims
            if probes is not None:
                probes.new_phase()      # fresh wall-clock budget for the optimizer's probes
            report = report + _frontier_section(history) \
                + _efficiency_section(train_fails, train_succ)
            # code_hooks: add the layer-choice guidance and the hooks API contract;
            # configurations with probes also get _VERIFIED_CODE_RULE. Configurations with
            # meta_teacher get the contract-only versions (_CODE_SECTION_SELF,
            # _TASK_BRIEF_SELF), since this guidance is what the optimizer develops itself.
            code_arm = (cfg.get("edit_space") or {}).get("kind") == "code_hooks"
            code_section = ""
            code_component = ""
            if code_arm:
                from verse.evolution.hooks_runtime import (
                    HOOKS_API_DOC, HOOKS_API_DOC_CONTRACT)
                code_component = " | harness_code/hooks.py (EXECUTABLE)"
                if meta_teacher is not None:
                    code_section = _CODE_SECTION_SELF.format(
                        hooks_api=HOOKS_API_DOC_CONTRACT)
                else:
                    code_section = _CODE_SECTION.format(hooks_api=HOOKS_API_DOC)
                if probes is not None and meta_teacher is None:
                    # configurations with meta_teacher get this text in their editable
                    # optimizer harness instead (seed_wisdom); adding it here too would
                    # duplicate it and make it uneditable
                    code_section += _VERIFIED_CODE_RULE
            notes_path = os.path.join(out_dir, "teacher_notes.md")
            notes = ""
            if os.path.exists(notes_path):
                notes = open(notes_path, errors="replace").read()[:4096]
            brief_tmpl = _TASK_BRIEF_SELF if meta_teacher is not None else _TASK_BRIEF
            prompt = brief_tmpl.format(
                history=_history_text(history), evidence=report,
                notes=notes or "(none yet — use write_notes to leave some)",
                code_component=code_component, code_section=code_section)
            if meta_teacher is not None:
                # the optimizer harness (prompt, skills, notes) is appended to the brief;
                # run_episode and the optimizer's own tools are added per episode below.
                # The run budget scales with n_candidates like probes.budget above, so
                # each episode gets the same allowance as with probes.
                prompt += meta_teacher.begin_round(rnd, ctx, rdir,
                                                   n_candidates=max(1, n_cand_cfg))
            # Best-of-N candidate generation: N parallel optimizer episodes, each with its
            # own candidate guidance, screened on a shared train subset; the winner goes on
            # to the smoke check and the val sweep. With n_candidates=1 a single episode runs
            # without a guidance suffix.
            n_cand = n_cand_cfg
            screen_n = int((cfg.get("search") or {}).get("screen_tasks", 12))
            # Source of the candidate guidance. Every method runs the same N episodes and
            # shared screening; only the guidance text differs:
            #   probes, no meta_teacher: the hand-written _LENSES
            #   baselines: the same generic suffix for every candidate (_SAMPLE_LENSES)
            #   meta_teacher: the optimizer's own search.md (empty at the start, which
            #     means plain sampling; with seed_wisdom it starts as a copy of _LENSES
            #     and may be rewritten in later rounds)
            if meta_teacher is not None:
                lenses = meta_teacher.search_lenses(max(1, n_cand))
            else:
                lens_pool = _LENSES if probes is not None else _SAMPLE_LENSES
                lenses = lens_pool[:max(1, n_cand)]

            def _validate(payload):
                """In-episode proposal validation: the edit-space and anti-cheat checks of
                the smoke stage, run while the optimizer still has its investigation context
                (repair calls without that context rarely converge)."""
                edits = payload.get("edits") or []
                try:
                    if len(edits) > getattr(edit_space, "max_files", 100):
                        raise ValueError(f"proposal touches {len(edits)} files > limit "
                                         f"{edit_space.max_files}")
                    edit_space._check(edits)
                    ws = HarnessWorkspace7(ws_dir)
                    for e in edits:
                        ws.resolve(e.get("path", ""))
                except Exception as e:
                    return str(e)
                return _cheat_audit(payload, ctx)

            from verse.workspace import HarnessWorkspace7

            def _episode(lens_pair):
                lens, suffix = lens_pair
                if probes is not None:
                    probes.new_phase()
                # evidence-source tools (e.g. the digest sub-agent of HarnessX's
                # pull_digest): fresh instances per episode, because each carries its own
                # call counter and the episodes run concurrently
                ev_tools = (evidence_src.intervener_tools(ctx)
                            if hasattr(evidence_src, "intervener_tools") else None)
                th = None
                if meta_teacher is not None:
                    # run_episode (simple verification) plus the optimizer's own harness
                    # code and tools: a fresh runtime per episode, with the run_episode
                    # budget shared across the round
                    ev_tools = (ev_tools or []) + meta_teacher.episode_tools(ctx, rdir)
                    th = meta_teacher.episode_hooks(ctx, rdir, key=lens)
                return lens, run_intervener(
                    prompt + (suffix if n_cand > 1 else ""), rdir, ctx,
                    model=intervener_cfg["model"],
                    regions=intervener_cfg.get("regions", "us-west-2,us-east-1,us-east-2"),
                    max_turns=int(intervener_cfg.get("max_turns", 60)),
                    audit_path=os.path.join(rdir, f"intervener_audit_{lens}.jsonl"),
                    notes_path=notes_path, validate_payload=_validate,
                    extra_tools=ev_tools, teacher_hooks=th,
                    # meta_teacher: the hand-written probe strategy lives in the editable
                    # optimizer harness (seed_wisdom), not in the system prompt
                    probe_wisdom=(meta_teacher is None))
            candidates, cand_errors = [], []
            if n_cand == 1:
                candidates.append(_episode(("solo", "")))
            else:
                with ThreadPoolExecutor(max_workers=n_cand) as pool:
                    futs = {pool.submit(_episode, lp): lp[0] for lp in lenses}
                    for fut in as_completed(futs):
                        try:
                            candidates.append(fut.result())
                        except Exception as e:
                            cand_errors.append({"lens": futs[fut],
                                                "error": f"{type(e).__name__}: {e}"})
            if not candidates:
                raise RuntimeError("every candidate episode failed: "
                                   + json.dumps(cand_errors)[:500])
            candidates.sort(key=lambda lp: [l for l, _ in lenses].index(lp[0])
                            if lp[0] in [l for l, _ in lenses] else 99)
            best_i, screen_log = _screen_candidates(
                candidates, edit_space, ws_dir, rdir, rnd, train_out, insts_by_tid,
                executor, screen_n, _seen_shas(history))
            lens_won, proposal = candidates[best_i]
            rec["search"] = {"n_candidates": len(candidates), "winner_lens": lens_won,
                             "screen": screen_log, "episode_errors": cand_errors}
            if ev_claims:
                proposal.claims = ev_claims + proposal.claims
        except Exception as e:
            import traceback
            rec.update({"error": f"{type(e).__name__}: {e}", "kept": False,
                        "traceback": traceback.format_exc()[-2000:]})
            open(journal, "a").write(json.dumps(rec) + "\n")
            history.append(rec)
            print(f"[evo] round {rnd}: intervener failed ({e})", flush=True)
            # meta_teacher: the round report and self-evolution step still run
            # (end_round fails open)
            if meta_teacher is not None:
                meta_teacher.end_round(rnd, rec, rdir, ctx)
            continue
        dump_claims(proposal.claims, os.path.join(rdir, "claims.json"))
        from verse.evolution.probes import edits_sha as _esha
        rec.update({"description": proposal.description,
                    "predicted_fixes": proposal.predicted_fixes,
                    "at_risk": proposal.at_risk, "files": proposal.files,
                    "edits_sha": _esha(proposal.meta.get("edits", [])),
                    "claims_summary": render_claims(proposal.claims)[:2000]})

        # 3. Smoke check and repair. The novelty check runs first: a change-set
        # byte-identical to one already judged dead is sent back for repair.
        # Branching (Meta-Harness frontier semantics): a proposal may set base_round, so its
        # edits apply on that kept round's state instead of HEAD. The optimizer can then
        # leave a harmful but kept stretch of the lineage without spending a round undoing it.
        head_before_round = edit_space.head(ws_dir)   # restore target for every revert
        base_r = proposal.meta.get("base_round")
        if base_r is not None:
            base_sha_map = {0: round0.get("ws_head")}
            base_sha_map.update({h["round"]: h["ws_head"] for h in history
                                 if h.get("kept") and h.get("ws_head")})
            if base_r in base_sha_map and base_sha_map[base_r]:
                edit_space.reset_to(ws_dir, base_sha_map[base_r])
                rec["base_round"] = base_r
                print(f"[evo] round {rnd}: branching from round {base_r} "
                      f"({base_sha_map[base_r][:8]})", flush=True)
            else:
                print(f"[evo] round {rnd}: base_round={base_r} not a kept round — "
                      "building on HEAD", flush=True)
        payload = _proposal_payload(proposal)
        seen = _seen_shas(history)
        prior = seen.get(_esha(payload.get("edits") or []))
        if prior and prior.get("dead"):
            payload = _repair(
                payload,
                f"NOVELTY GATE: this exact change-set was already tried in round "
                f"{prior['round']} and judged {prior['label']} — it will not be re-run. "
                "Propose a materially different change (different mechanism or layer).",
                ctx, intervener_cfg) or payload
            rec["edits_sha"] = _esha(payload.get("edits") or [])
        sha, smoke_log, payload = smoke_with_repair(payload, edit_space, ws_dir, rnd,
                                                    smoke_inst, executor, ctx,
                                                    intervener_cfg, claims=proposal.claims)
        rec["smoke_log"] = smoke_log
        # Journal what was actually committed: a repair may have rewritten the change-set.
        # predicted_fixes are normalized again because repair payloads come straight from
        # the LLM and can reintroduce the 'id (rationale...)' form that run_intervener strips.
        from verse.evolution.intervener import _as_task_ids
        rec["files"] = [e.get("path", "") for e in payload.get("edits") or []]
        rec["predicted_fixes"] = _as_task_ids(payload.get("predicted_fixes"),
                                              set(insts_by_tid))
        rec["edits_sha"] = _esha(payload.get("edits") or [])
        # Change-set size in bytes, journaled and shown to the optimizer in the round
        # history, so it can relate edit size to realized val gains across rounds.
        rec["edit_bytes"] = sum(len((e.get("content") or "").encode())
                                for e in payload.get("edits") or []
                                if not e.get("delete"))
        if sha is None:
            # restore the pre-round head: after a base_round branch the workspace sits at
            # the branch base, and leaving it there would move the lineage
            edit_space.reset_to(ws_dir, head_before_round)
            rec.update({"kept": False, "verdict": {"label": "INVALID",
                                                   "reason": "failed smoke after repairs"},
                        "ws_head": edit_space.head(ws_dir)})
            open(journal, "a").write(json.dumps(rec) + "\n")
            history.append(rec)
            print(f"[evo] round {rnd}: INVALID (smoke)", flush=True)
            # meta_teacher: the round report and self-evolution step still run. An INVALID
            # round can mean the optimizer's own submission process broke, which is what
            # the self-evolution step needs to see.
            if meta_teacher is not None:
                meta_teacher.end_round(rnd, rec, rdir, ctx)
            continue

        # 4-5. Val sweep and flip confirmation.
        sp_new = edit_space.assemble(ws_dir)
        hooks_new = _load_hooks_src(ws_dir)
        hooks_stats = {} if hooks_new else None
        val_after, val_fails_after = sweep(val_insts, sp_new, executor, concurrency=conc,
                                           label=f"r{rnd}-val", nan_retries=4,
                                           hooks_src=hooks_new, hooks_stats=hooks_stats)
        if hooks_stats:
            # Journal hook health. Above a 50% exception rate the sweep measured a largely
            # degraded harness: the round is flagged hooks_broken, excluded from the final
            # selection, and the next round's history tells the optimizer why.
            n_ep = max(1, hooks_stats.get("episodes", 0))
            err_rate = (hooks_stats.get("episodes_with_errors", 0)
                        + hooks_stats.get("load_errors", 0)) / n_ep
            hooks_stats["error_rate"] = round(err_rate, 3)
            rec["hooks_stats"] = hooks_stats
            if err_rate > 0.5:
                rec["hooks_broken"] = True
                print(f"[evo] round {rnd}: HOOKS_BROKEN — code layer failed in "
                      f"{err_rate:.0%} of val episodes", flush=True)
        # Degraded-sweep check on the before/after intersection, not on the raw count: the
        # gate only compares tasks present in both sweeps, so tasks that fail on
        # infrastructure every round (e.g. an image that cannot be pulled) do not distort it.
        # A collapsed intersection (e.g. an outage mid-sweep) is caught here, because a
        # partial flip set biases the verdict toward INERT.
        coverage = len(set(val_out) & set(val_after)) / max(1, len(val_insts))
        rec["val_coverage"] = round(coverage, 3)
        if coverage < 0.8:
            edit_space.reset_to(ws_dir, head_before_round)
            rec.update({"error": f"val sweep degraded: before∩after covers "
                                 f"{coverage:.0%} of {len(val_insts)} tasks (<80%) "
                                 "— infra, not gated", "kept": False})
            open(journal, "a").write(json.dumps(rec) + "\n")
            history.append(rec)
            print(f"[evo] round {rnd}: DEGRADED val sweep (coverage {coverage:.0%}), "
                  "not gated", flush=True)
            # meta_teacher: the round report and self-evolution step still run (see the
            # INVALID path)
            if meta_teacher is not None:
                meta_teacher.end_round(rnd, rec, rdir, ctx)
            continue
        if sweep_cfg.get("flip_confirm", True):
            rec["flip_confirm"] = confirm_flips(val_out, val_after, val_fails_after,
                                                insts_by_tid, sp_new, executor,
                                                hooks_src=hooks_new)

        # 6. Gate.
        verdict = gate.judge(ctx, proposal, val_out, val_after)
        rec.update({"verdict": {"label": verdict.label, "reason": verdict.reason,
                                **verdict.detail}, "kept": verdict.keep,
                    "val_score_before": f"{sum(1 for v in val_out.values() if v >= 0.5)}/{len(val_out)}",
                    "val_score_after": f"{sum(1 for v in val_after.values() if v >= 0.5)}/{len(val_after)}"})
        if verdict.keep:
            val_out = val_after
        else:
            # reset (not revert_last): after a base_round branch the last commit sits on
            # the branch base, and popping it would move the lineage there
            edit_space.reset_to(ws_dir, head_before_round)
        rec["ws_head"] = edit_space.head(ws_dir)
        open(journal, "a").write(json.dumps(rec) + "\n")
        history.append(rec)
        print(f"[evo] round {rnd}: {verdict.label} kept={verdict.keep} "
              f"val {rec['val_score_before']} -> {rec['val_score_after']}", flush=True)

        # 7. meta_teacher only: the round report and the self-evolution step, in which the
        # optimizer edits its own harness. It fails open, and its edits are not gated.
        if meta_teacher is not None:
            meta_teacher.end_round(rnd, rec, rdir, ctx)

    # Final selection: the reported harness is the state of the best round on val. The
    # argmax of single-sweep val scores tends to pick a noise peak, so the top candidates
    # (search.final_confirm_top, default 2) each get a fresh val sweep, and selection uses
    # the mean of the journaled and confirmation scores.
    candidates = [(sum(1 for v in round0["val_outcomes"].values() if float(v) >= 0.5), 0,
                   round0.get("ws_head"))]
    for h in history:
        va = h.get("val_score_after")
        if h.get("kept") and va and h.get("ws_head") and not h.get("hooks_broken"):
            # hooks_broken rounds are excluded: their val score measured a degraded
            # harness, not the proposed one
            candidates.append((int(va.split("/")[0]), h["round"], h["ws_head"]))
    confirm_log = {}
    # Round 0 competes like any other round, with no reserved slot. Ties on journal score
    # prefer the later round (later rounds have seen more evidence).
    n_confirm = int((cfg.get("search") or {}).get("final_confirm_top", 2))
    if n_confirm > 1 and len(candidates) > 1:
        top = sorted(candidates, key=lambda c: (c[0], c[1]), reverse=True)[:n_confirm]
        rescored = []
        for score, rnd_i, sha_i in top:
            try:
                edit_space.reset_to(ws_dir, sha_i)
                sp_i = edit_space.assemble(ws_dir)
                out_i, _ = sweep(val_insts, sp_i, executor, concurrency=conc,
                                 label=f"final-confirm-r{rnd_i}", nan_retries=4,
                                 hooks_src=_load_hooks_src(ws_dir))
                conf = sum(1 for v in out_i.values() if v >= 0.5)
                # combined = mean of the journaled and confirmation scores (two estimates of
                # the same quantity); the confirmation score is rescaled for missing tasks
                scale = len(val_insts) / max(1, len(out_i))
                combined = (score + conf * scale) / 2
                confirm_log[str(rnd_i)] = {"journaled": score, "confirm": conf,
                                           "n_judged": len(out_i),
                                           "combined": round(combined, 2)}
                rescored.append((combined, rnd_i, sha_i))
            except Exception as e:
                print(f"[evo] final confirm r{rnd_i} failed ({e}) — keeping journal score",
                      flush=True)
                rescored.append((float(score), rnd_i, sha_i))
        best_score, best_round, best_sha = max(rescored)
        best_score = int(round(best_score))
    else:
        best_score, best_round, best_sha = max(candidates)  # ties -> latest round wins
    # leave the workspace at the picked state (the test evaluation checks out best_sha)
    try:
        edit_space.reset_to(ws_dir, best_sha)
    except Exception:
        pass
    final = {"round": "final", "best_round": best_round, "best_val": best_score,
             "best_sha": best_sha, "confirm": confirm_log, "ts": time.time()}
    open(journal, "a").write(json.dumps(final) + "\n")
    print(f"[evo] DONE -> {journal} | frontier pick: round {best_round} "
          f"(val {best_score}/{len(round0['val_outcomes'])}, sha {str(best_sha)[:8]})", flush=True)


def _newest_kept_traj_dir(out_dir: str, history: list) -> str | None:
    """The newest non-empty traj_train directory over all rounds. A train sweep runs only at
    round 0 or right after a kept round, so the newest one reflects the harness after the
    last kept edit. Note that this directory lives at round K+1, not at the kept round K."""
    last = history[-1]["round"] if history else 0
    dirs = [os.path.join(out_dir, f"round_{r:02d}", "traj_train")
            for r in range(0, last + 1)]
    for d in reversed(dirs):
        if os.path.isdir(d) and os.listdir(d):
            return d
    return None


def _reload_train_out(out_dir: str, history: list) -> dict:
    """Rebuild {tid: phi} from the newest train sweep's trajectory files (resume support).
    NaN phis are skipped, as in sweep(); otherwise train_score's denominator would drift
    between rounds, and a task that passed before and is NaN now would be labeled
    'regressed' (NaN >= 0.5 is False)."""
    d = _newest_kept_traj_dir(out_dir, history)
    out = {}
    if d:
        for name in os.listdir(d):
            try:
                rec = json.load(open(os.path.join(d, name)))
                phi = float(rec["phi"])
                if phi == phi:                        # NaN = infrastructure, not an outcome
                    out[rec.get("instance_id", name[:-5])] = phi
            except Exception:
                continue
    return out


def _reload_fails(out_dir: str, history: list, train_out: dict) -> dict:
    """Rebuild {tid: transcript} for currently failing tasks from the newest train sweep
    (resume support: the driver process may have restarted since that sweep ran)."""
    fails = {}
    d = _newest_kept_traj_dir(out_dir, history)
    if d:
        for name in os.listdir(d):
            tid = name[:-5]
            if tid in fails or train_out.get(tid, 1.0) >= 0.5:
                continue
            try:
                fails[tid] = json.load(open(os.path.join(d, name)))["transcript"]
            except Exception:
                continue
    return fails


def _reload_successes(out_dir: str, history: list, train_out: dict) -> dict:
    """Mirror of _reload_fails for passing tasks (success-side evidence)."""
    succ = {}
    d = _newest_kept_traj_dir(out_dir, history)
    if d:
        for name in os.listdir(d):
            tid = name[:-5]
            if tid in succ or train_out.get(tid, 0.0) < 0.5:
                continue
            try:
                succ[tid] = json.load(open(os.path.join(d, name)))["transcript"]
            except Exception:
                continue
    return succ


# CLI
def main():
    ap = argparse.ArgumentParser(description="config-driven harness evolution")
    ap.add_argument("--config", required=True, help="method yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[],
                    help="dotted override, e.g. --set rounds=2 --set sweep.concurrency=6")
    args = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(open(args.config))
    # _base.yaml sits at the configs/ root; method yamls live one level down
    # (configs/baselines/, configs/ours/, ...) — check both locations.
    cfg_dir = os.path.dirname(os.path.abspath(args.config))
    base_path = os.path.join(cfg_dir, "_base.yaml")
    if not os.path.exists(base_path):
        base_path = os.path.join(os.path.dirname(cfg_dir), "_base.yaml")
    if os.path.exists(base_path):
        def _merge(base, over):
            out = dict(base)
            for k, v in over.items():
                out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
            return out
        cfg = _merge(yaml.safe_load(open(base_path)), cfg)
    for ov in args.set:
        key, _, val = ov.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        try:
            val = json.loads(val)
        except Exception:
            pass
        node[parts[-1]] = val
    cfg["out_dir"] = args.out
    run_evolution(cfg)


if __name__ == "__main__":
    main()
