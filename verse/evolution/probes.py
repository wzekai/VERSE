"""Probes: replay, perturbation and verification experiments as a pluggable component.

This component switches verification on. When a config enables it
(`probes: {kind: t2t_probes}`), evidence sources (and the pull_explore optimizer, through
tools) can run real experiments and attach the resulting ExperimentRecords to their claims.
Without it, every claim stays a hypothesis with no evidence.

Each probe answers a question whose answer is not known in advance:
    replay       Replay tool. Re-runs the recorded commands in a fresh environment.
                 Reproduction only confirms that the trace is valid (verdict inconclusive);
                 non-reproduction refutes every step-level claim on the trace. It is the
                 only trace probe that can refute, for failed and passing tasks alike.
    ablate       Perturbation tool, leave-one-out in both directions: drop step k and
                 replay. On a failed task a flip shows step k caused the failure (rare,
                 since most failures are omissions); on a passing task a flip to fail
                 shows step k is necessary for the success.
    substitute   Perturbation tool. Replaces step k's command and replays; a flip shows a
                 better action exists at k and validates that exact replacement.
    fix_probe    Verification tool. Applies the optimizer's draft harness edits to a
                 scratch workspace and re-runs the tasks it claims to fix. A zero-flip
                 result is a refutation that the smoke gate enforces against the same edits.

Probes reuse the existing execution code: step_certificates._replay_phi
(tb_env.tb_replay_phi on TB), driver.run_task and HarnessWorkspace7.

Budget: a per-round experiment budget (a count plus a wall-clock deadline). Over budget, a
probe returns an 'inconclusive' record that says so.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from verse.claims import ExperimentRecord

from verse.evolution.registry import register_component


def edits_sha(edits: list) -> str:
    """Stable fingerprint of a draft change-set. fix_probe stamps its records with it, so
    the smoke gate enforces refutations only against the same edits; a revised change-set
    is a new hypothesis and is not blocked by earlier refutations."""
    return hashlib.sha1(json.dumps(edits or [], sort_keys=True).encode()).hexdigest()[:16]


PROBE_TOOL_NAMES = ("replay", "ablate", "substitute", "fix_probe")


@register_component("probes", "t2t_probes")
class T2TProbeKit:
    """The replay, perturbation and verification tools.

    budget: max experiments per round. budget_s: wall-clock seconds per phase (see
    new_phase). tools: which probe tools the optimizer may call (for ablations); None
    enables all four. The class always implements all four, so evidence-side verification and
    the gate keep their access regardless of what the optimizer sees.
    Thread-safe: evidence-side verification runs probes concurrently (independent
    containers, no LLM calls), so the count budget is consumed under a lock."""

    def __init__(self, budget: int = 8, budget_s: float = 3600.0, probe_n: int = 2,
                 tools: list | None = None):
        self.budget = int(budget)
        self.budget_s = float(budget_s)
        self.probe_n = int(probe_n)
        if tools is not None:
            unknown = [t for t in tools if t not in PROBE_TOOL_NAMES]
            if unknown:
                raise ValueError(f"unknown probe tools {unknown}; "
                                 f"valid: {list(PROBE_TOOL_NAMES)}")
        self.tools = tuple(tools) if tools is not None else PROBE_TOOL_NAMES
        self._used = 0
        self._deadline = None
        self._lock = threading.Lock()

    # driver calls this at the start of each round
    def reset_round(self) -> None:
        self._used = 0
        self._deadline = time.monotonic() + self.budget_s

    def new_phase(self) -> None:
        """Start a fresh wall-clock window for the next consumer of the same count budget.
        One ablate on a long trace takes two full container replays (30-60 minutes under
        throttling), so the evidence phase alone could use up the round deadline and leave
        the optimizer's probe tools unusable. The count still caps total spend; the
        wall-clock limit is per phase."""
        self._deadline = time.monotonic() + self.budget_s

    def remaining(self) -> int:
        return max(0, self.budget - self._used)

    @property
    def used(self) -> int:
        return self._used

    def take_unit(self) -> int | None:
        """Reserve one execution unit for a consumer other than the probes. With optimizer
        self-evolution, these tools are enabled alongside the run_episode tool (simple
        verification), and both draw from this one per-round pool: a fix_probe target, a
        replay and a run_episode call each cost one unit. Adding run_episode therefore does
        not raise the execution budget. Returns a unique slot id, or None when the count or
        wall-clock budget is spent."""
        with self._lock:
            if self.budget - self._used <= 0 or (
                    self._deadline is not None and time.monotonic() > self._deadline):
                return None
            self._used += 1
            return self._used

    def _budget_record(self, kind: str, task_id: str) -> ExperimentRecord:
        # name the limit that was hit: "budget exhausted" while count budget remains reads
        # as a contradiction and can make the optimizer stop probing altogether
        if self.remaining() <= 0:
            why = f"probe count budget exhausted ({self.budget}/round)"
        else:
            why = (f"probe wall-clock for this phase expired ({int(self.budget_s)}s); "
                   "count budget remains but no more experiments fit this round")
        return ExperimentRecord(kind=kind, setup="(not run)", outcome=why,
                                verdict="inconclusive", task_id=task_id)

    # Experiments
    def replay(self, ctx, task_id: str) -> ExperimentRecord:
        """Replay tool: do the recorded commands reproduce the recorded outcome in a fresh
        environment? Works on failed and passing tasks. Reproduction is the expected outcome
        and only confirms the trace is valid (verdict inconclusive); non-reproduction refutes
        all step-level claims on the trace."""
        if self.remaining() <= 0 or (self._deadline is not None and time.monotonic() > self._deadline):
            return self._budget_record("replay", task_id)
        from verse.runtime import swe_judge as _judge
        from verse.runtime.step_certificates import StepCertificates, _replay_phi
        inst, steps, recorded_pass = self._task_state(ctx, task_id)
        if inst is None:                 # invalid call — costs no budget
            return ExperimentRecord("replay", "(not run)", f"unknown task {task_id}",
                                    "inconclusive", task_id)
        with self._lock:
            if self.remaining() <= 0:
                return self._budget_record("replay", task_id)
            self._used += 1
        cmds = [s["command"] for s in steps if s["kind"] == "bash"]
        if self._is_tb(inst):
            # TB is judged on end state: replay the recorded commands in a fresh compose
            # environment and run the task's own tests (no patch step, unlike SWE)
            from verse.runtime.tb_env import tb_replay_phi
            phi = tb_replay_phi(inst, cmds, self._deadline,
                                int(os.environ.get("T2E_SWE_RUN_TIMEOUT", "1800")))
        else:
            phi = _replay_phi(inst, _judge._image_name(inst), cmds, self._deadline,
                              int(os.environ.get("T2E_SWE_RUN_TIMEOUT", "1800")), StepCertificates())
        if phi is None:
            return ExperimentRecord("replay", f"re-ran {len(cmds)} recorded commands",
                                    "infra failure during replay", "inconclusive", task_id)
        reproduced = (phi >= 0.5) == recorded_pass
        word = "pass" if recorded_pass else "failure"
        if reproduced:
            outcome = (f"recorded {word} REPRODUCED (phi={phi:g}) — deterministic; "
                       "step-level experiments on this trace are valid")
            verdict = "inconclusive"     # a validity check, not evidence for a specific claim
        else:
            outcome = (f"recorded {word} did NOT reproduce (phi={phi:g}) — nondeterministic "
                       "world; all step-level claims on this trace are void")
            verdict = "refutes"
        return ExperimentRecord(
            "replay", f"re-ran all {len(cmds)} recorded commands in a fresh environment",
            outcome, verdict, task_id, {"replay_phi": phi, "recorded_pass": recorded_pass})

    def ablate(self, ctx, task_id: str, step_k: int) -> ExperimentRecord:
        """Perturbation tool, leave-one-out in both directions: drop step k and replay.

        On a failed task, a fail-to-pass flip shows step k caused the failure (rare, since
        most failures are omissions). On a passing task, a pass-to-fail flip shows step k
        is necessary for the success. The passing direction is the more useful one: it
        gives executed evidence for which behaviors to distill into skills."""
        if self.remaining() <= 0 or (self._deadline is not None and time.monotonic() > self._deadline):
            return self._budget_record("ablate", task_id)
        from verse.runtime import swe_judge as _judge
        from verse.runtime.step_certificates import StepCertificates, _replay_phi
        inst, steps, recorded_pass = self._task_state(ctx, task_id)
        if inst is None:                 # invalid call — costs no budget
            return ExperimentRecord("ablate", "(not run)", f"unknown task {task_id}",
                                    "inconclusive", task_id)
        bash = [(i, s) for i, s in enumerate(steps) if s["kind"] == "bash"]
        if not (0 <= step_k < len(steps)) or steps[step_k]["kind"] != "bash":
            return ExperimentRecord("ablate", "(not run)",
                                    f"step {step_k} out of range or not a command step",
                                    "inconclusive", task_id)
        with self._lock:
            if self.remaining() <= 0:
                return self._budget_record("ablate", task_id)
            self._used += 1
        timeout = int(os.environ.get("T2E_SWE_RUN_TIMEOUT", "1800"))
        if self._is_tb(inst):
            from verse.runtime.tb_env import tb_replay_phi as _rp
            _replay = lambda cs: _rp(inst, cs, self._deadline, timeout)
        else:
            image = _judge._image_name(inst)
            cert = StepCertificates()
            _replay = lambda cs: _replay_phi(inst, image, cs, self._deadline, timeout, cert)
        # replay check first: if the recorded outcome does not reproduce, nothing is certified
        full = [s["command"] for _, s in bash]
        phi_full = _replay(full)
        if phi_full is None or (phi_full >= 0.5) != recorded_pass:
            outcome = ("infra failure during full replay" if phi_full is None else
                       "full replay did not reproduce the recorded outcome — "
                       "step claims on this trace unverifiable")
            return ExperimentRecord("ablate", "d-gate replay before ablation", outcome,
                                    "inconclusive", task_id, {"replay_phi": phi_full})
        loo = [s["command"] for i, s in bash if i != step_k]
        phi_wo = _replay(loo)
        if phi_wo is None:
            return ExperimentRecord("ablate", f"replayed without step {step_k}",
                                    "infra failure during LOO replay", "inconclusive", task_id)
        cmd = (steps[step_k].get("command") or "")[:120]
        flipped = (phi_wo >= 0.5) != recorded_pass
        if flipped and not recorded_pass:
            outcome = f"phi FLIPPED 0->1: step {step_k} is the proven culprit of the failure"
            verdict = "supports"
        elif flipped and recorded_pass:
            outcome = (f"phi FLIPPED 1->0: step {step_k} is PROVEN LOAD-BEARING for the "
                       "success — distill this behavior into the harness")
            verdict = "supports"
        elif recorded_pass:
            outcome = (f"pass survived without step {step_k}: that step is not individually "
                       "necessary (redundant or the win lives elsewhere)")
            verdict = "inconclusive"
        else:
            # No flip on a failure is not a refutation: most failures lack the correct fix,
            # and dropping one step cannot supply it.
            outcome = ("phi stayed 0: step not individually load-bearing (weak evidence — "
                       "does not refute a wrong-decision diagnosis; the fix may simply be missing)")
            verdict = "inconclusive"
        return ExperimentRecord(
            "ablate", f"replayed the {'passing' if recorded_pass else 'failed'} trajectory "
                      f"without step {step_k} (`{cmd}`)",
            outcome, verdict, task_id,
            {"phi_without": phi_wo, "step": step_k, "recorded_pass": recorded_pass})

    def substitute(self, ctx, task_id: str, step_k: int, new_command: str) -> ExperimentRecord:
        """Perturbation tool: replace step k's command with the optimizer's proposed
        alternative and replay. A fail-to-pass flip shows that a better action existed at
        step k and validates that exact replacement, which a deletion cannot show (deleting
        a step only tests whether it was harmful, which it rarely is)."""
        if self.remaining() <= 0 or (self._deadline is not None and time.monotonic() > self._deadline):
            return self._budget_record("substitute", task_id)
        from verse.runtime import swe_judge as _judge
        from verse.runtime.step_certificates import StepCertificates, _replay_phi
        inst, steps, recorded_pass = self._task_state(ctx, task_id)
        if inst is None:                 # invalid call — costs no budget
            return ExperimentRecord("substitute", "(not run)", f"unknown task {task_id}",
                                    "inconclusive", task_id)
        if recorded_pass:
            return ExperimentRecord("substitute", "(not run)",
                                    "substitute probes failed tasks only", "inconclusive", task_id)
        if not (new_command or "").strip():
            return ExperimentRecord("substitute", "(not run)", "empty replacement command",
                                    "inconclusive", task_id)
        bash = [(i, s) for i, s in enumerate(steps) if s["kind"] == "bash"]
        if not (0 <= step_k < len(steps)) or steps[step_k]["kind"] != "bash":
            return ExperimentRecord("substitute", "(not run)",
                                    f"step {step_k} out of range or not a command step",
                                    "inconclusive", task_id)
        with self._lock:
            if self.remaining() <= 0:
                return self._budget_record("substitute", task_id)
            self._used += 1
        timeout = int(os.environ.get("T2E_SWE_RUN_TIMEOUT", "1800"))
        if self._is_tb(inst):
            from verse.runtime.tb_env import tb_replay_phi as _rp
            _replay = lambda cs: _rp(inst, cs, self._deadline, timeout)
        else:
            image = _judge._image_name(inst)
            cert = StepCertificates()
            _replay = lambda cs: _replay_phi(inst, image, cs, self._deadline, timeout, cert)
        full = [s["command"] for _, s in bash]
        phi_full = _replay(full)
        if phi_full is None or phi_full >= 0.5:
            outcome = ("infra failure during full replay" if phi_full is None else
                       "full replay did not reproduce the failure — substitution unverifiable")
            return ExperimentRecord("substitute", "d-gate replay before substitution", outcome,
                                    "inconclusive", task_id, {"replay_phi": phi_full})
        swapped = [(new_command if i == step_k else s["command"]) for i, s in bash]
        phi_sub = _replay(swapped)
        if phi_sub is None:
            return ExperimentRecord("substitute", f"replayed with step {step_k} replaced",
                                    "infra failure during substituted replay",
                                    "inconclusive", task_id)
        old = (steps[step_k].get("command") or "")[:100]
        if phi_sub >= 0.5:
            outcome = (f"phi FLIPPED 0->1: replacing step {step_k} fixes the task — "
                       "a better action existed there and THIS substitute is it")
            verdict = "supports"
        else:
            outcome = (f"phi stayed 0 with the substitute: this replacement does not fix "
                       "the task (the failure lives elsewhere or needs more than this change)")
            verdict = "inconclusive"
        return ExperimentRecord(
            "substitute",
            f"replayed with step {step_k} (`{old}`) replaced by `{new_command[:100]}`",
            outcome, verdict, task_id,
            {"phi_substituted": phi_sub, "step": step_k, "new_command": new_command[:300]})

    def fix_probe(self, ctx, edits: list, target_tids: list) -> ExperimentRecord:
        """Verification tool: apply draft harness edits to a scratch copy of the executor
        harness and re-run the tasks the optimizer claims they fix.

        A flip confirmed by a second run means the draft fixes that task (supports). Zero
        flips means the prediction is refuted for the tested tasks under this exact
        change-set (refutes, stamped with edits_sha so a revised draft is not blocked by
        earlier refutations). Costs one budget unit per task run; confirmation runs are
        free.

        Drafts that edit harness_code/hooks.py are fully exercised: the scratch hooks source
        is loaded on every probe episode, and the report includes the code layer's
        execution record (per-hook call and change counts, and the first exceptions
        verbatim). Code edits are high-variance (one wrong after_tool can truncate every
        observation; one correct sanitizer can remove a whole failure class), so running
        them before submission is informative."""
        from verse.workspace import HarnessWorkspace7
        from verse.evolution.driver import run_task   # family dispatch (SWE / TB)
        from verse.evolution.hooks_runtime import (
            load_hooks_source, extract_hooks_summary)
        # Train-only guard: insts_by_tid also holds validation instances (the gate needs
        # them), but probing a validation task would optimize the round-selection metric
        # directly.
        allowed = ctx.train_tids or set(ctx.insts_by_tid)   # empty: ctx has no split info
        req = [t for t in (target_tids or []) if t in ctx.insts_by_tid]
        blocked = [t for t in req if t not in allowed]
        tids = [t for t in req if t in allowed][:3]
        if blocked and not tids:
            return ExperimentRecord("fix_probe", "(not run)",
                                    f"targets {', '.join(blocked[:3])} are validation tasks — "
                                    "probes run on TRAIN tasks only (val is the held-back "
                                    "measurement set); pick targets from the failure index",
                                    "inconclusive")
        if not edits or not tids:
            return ExperimentRecord("fix_probe", "(not run)",
                                    "need edits and at least one known target task id",
                                    "inconclusive")
        n = len(tids)
        if self.remaining() < n or (self._deadline is not None and time.monotonic() > self._deadline):
            return self._budget_record("fix_probe", ",".join(tids))
        with self._lock:
            if self.remaining() < n:
                return self._budget_record("fix_probe", ",".join(tids))
            self._used += n
        # Scratch workspace: copy the current harness, apply the draft, assemble (and load
        # hooks). The random suffix matters: concurrent best-of-N candidate episodes share
        # this object and may probe identical drafts, so same-name scratch directories would
        # race.
        import uuid as _uuid
        scratch = os.path.join(ctx.scratch_dir or "/tmp",
                               f"fixprobe_{edits_sha(edits)}_{_uuid.uuid4().hex[:6]}")
        shutil.rmtree(scratch, ignore_errors=True)
        shutil.copytree(ctx.ws_dir, scratch)
        try:
            ws = HarnessWorkspace7(scratch)
            ws.apply_changeset(edits, round_idx=-1, description="fix_probe draft")
            sp = ws.assemble()
            hooks_src = load_hooks_source(scratch)
        except Exception as e:
            shutil.rmtree(scratch, ignore_errors=True)
            with self._lock:
                self._used -= n          # invalid draft: refund, apply errors cost nothing
            return ExperimentRecord("fix_probe", "(not run)",
                                    f"draft edits failed to apply: {str(e)[:200]}",
                                    "inconclusive", ",".join(tids))
        draft_touches_code = any(
            "harness_code" in (e.get("path") or "") for e in edits)
        ex = ctx.executor or {}
        hook_evidence = []               # HARNESS_CODE_SUMMARY dicts across probe episodes
        fail_sigs = {}                   # {tid: verifier/test failure line} for 0-flip targets

        def _one(tid):
            try:
                phi, tr = run_task(ctx.insts_by_tid[tid], sp, "bedrock", ex["model"],
                                   max_turns=int(ex.get("max_turns", 100)),
                                   bedrock_region=ex.get("regions",
                                                         "us-west-2,us-east-1,us-east-2"),
                                   hooks_src=hooks_src)
                if phi == phi and phi < 0.5:
                    # return the probe episode's own verifier verdict to the optimizer:
                    # "0 flips" alone only says try again, while "still failing test_X"
                    # says what the next revision must change. SWE and TB both write this
                    # transcript line.
                    for m in reversed(tr):
                        c = m.get("content", "") if isinstance(m, dict) else ""
                        if isinstance(c, str) and (c.startswith("VERIFIER")
                                                   or c.startswith("TEST FAILURES")
                                                   or c.startswith("TEST RESULT")):
                            fail_sigs[tid] = c[:200]
                            break
                if hooks_src:
                    hs = extract_hooks_summary(tr)
                    if hs:
                        hook_evidence.append(hs)
                    else:
                        for m in tr:
                            c = m.get("content", "") if isinstance(m, dict) else ""
                            if isinstance(c, str) and c.startswith("HARNESS_CODE_LOAD_ERROR"):
                                hook_evidence.append({"load_error": c[:600]})
                                break
                return tid, phi
            except Exception:
                return tid, float("nan")
        with ThreadPoolExecutor(max_workers=min(3, n)) as pool:
            results = dict(pool.map(_one, tids))
        ran = {t: p for t, p in results.items() if p == p}
        first_flips = sorted(t for t, p in ran.items() if p >= 0.5)
        # Confirmation: a single flip on a borderline task may be noise, so a flip counts
        # only when the same draft flips the same task twice. Confirmation runs do not use
        # the count budget.
        confirm = {}
        if first_flips:
            with ThreadPoolExecutor(max_workers=min(3, len(first_flips))) as pool:
                confirm = dict(pool.map(_one, first_flips))
        shutil.rmtree(scratch, ignore_errors=True)
        flips = sorted(t for t in first_flips
                       if confirm.get(t) == confirm.get(t) and confirm.get(t, 0.0) >= 0.5)
        unconfirmed = sorted(set(first_flips) - set(flips))
        sha = edits_sha(edits)
        code_report = self._hooks_report(hook_evidence, draft_touches_code)
        if fail_sigs:
            code_report += "\n[probe verdicts] " + "; ".join(
                f"{t}: {sig}" for t, sig in sorted(fail_sigs.items()))
        detail = {"edits_sha": sha, "tested": tids, "phis": {t: (p if p == p else None)
                                                             for t, p in results.items()},
                  "confirm_phis": {t: (p if p == p else None) for t, p in confirm.items()},
                  "flips": flips, "unconfirmed_flips": unconfirmed,
                  **({"fail_signatures": fail_sigs} if fail_sigs else {})}
        if hook_evidence:
            detail["hooks"] = hook_evidence[:4]
        if not ran:
            return ExperimentRecord("fix_probe", f"draft applied; ran {n} target tasks",
                                    "infra failure on every target task" + code_report,
                                    "inconclusive", ",".join(tids), detail)
        if flips:
            outcome = (f"draft edits FIX {len(flips)}/{len(ran)} tested targets "
                       f"({', '.join(flips)}) — flip CONFIRMED by a second run"
                       + (f"; {', '.join(unconfirmed)} flipped once but did not confirm "
                          "(borderline noise, not counted)" if unconfirmed else ""))
            verdict = "supports"
        elif unconfirmed:
            outcome = (f"{', '.join(unconfirmed)} flipped on the first run but NOT on the "
                       "confirmation run — borderline/noisy, treat as unproven (probe again "
                       "or pick a more deterministic target)")
            verdict = "inconclusive"
        else:
            # The wording matters: an unqualified refutation can make the optimizer give up
            # on submitting and waste the round. A 0/N result on 1-3 noisy tasks is weak
            # evidence against those targets, not against the change-set, so the message
            # says that and states the decision rule.
            outcome = (f"draft edits fixed 0/{len(ran)} tested targets — do not claim these "
                       f"specific tasks in predicted_fixes. This does NOT prove the change-set "
                       "worthless (n is tiny and single-task flips are noisy; probe-refuted "
                       "proposals have later scored best-of-arm). Decide: if the edits rest "
                       "on evidence beyond these targets, adjust predicted_fixes and submit; "
                       "revise only if these targets were the whole rationale.")
            verdict = "refutes"
        return ExperimentRecord(
            "fix_probe", f"applied draft change-set ({sha}) to a scratch harness and re-ran "
                         f"{len(ran)} predicted-fix tasks", outcome + code_report, verdict,
            ",".join(tids), detail)

    @staticmethod
    def _hooks_report(hook_evidence: list, draft_touches_code: bool) -> str:
        """Render the code layer's execution record into the probe outcome the optimizer
        reads. Hooks that silently fell back to the identity are reported prominently:
        fail-open keeps episodes alive, but hiding the exceptions would let a broken
        hooks.py be submitted looking merely ineffective."""
        if not hook_evidence:
            if draft_touches_code:
                return ("\n[harness code] WARNING: your draft edits harness_code/hooks.py "
                        "but no code-layer activity was recorded — the hooks likely never "
                        "mounted; check the file defines `class Hooks(BaseHooks)`.")
            return ""
        load_errs = [h["load_error"] for h in hook_evidence if "load_error" in h]
        if load_errs:
            return ("\n[harness code] hooks.py FAILED TO LOAD — every probe episode ran "
                    f"WITHOUT your code (phi results measure the md layer only): "
                    f"{load_errs[0][:400]}")
        n_eps = len(hook_evidence)
        calls, changed = {}, {}
        errors = []
        for h in hook_evidence:
            for k, v in (h.get("counts") or {}).items():
                calls[k] = calls.get(k, 0) + v
            for k, v in (h.get("changed") or {}).items():
                changed[k] = changed.get(k, 0) + v
            errors.extend(h.get("errors") or [])
        eps_with_err = sum(1 for h in hook_evidence if h.get("n_errors"))
        n_llm = sum(h.get("llm_calls", 0) for h in hook_evidence)
        lines = [f"\n[harness code] executed in {n_eps} probe episode(s): "
                 + (", ".join(f"{k} called {calls[k]}x/changed {changed.get(k, 0)}x"
                              for k in sorted(calls)) or "no hook activity")
                 + (f"; self.llm used {n_llm}x (each costs one agent turn)"
                    if n_llm else "")]
        if errors:
            lines.append(f"[harness code] EXCEPTIONS in {eps_with_err}/{n_eps} episodes "
                         "(each degraded that call to identity):")
            seen = set()
            for e in errors:
                key = (e.get("hook"), e.get("error"))
                if key in seen:
                    continue
                seen.add(key)
                lines.append(f"  - {e.get('hook')} (turn {e.get('turn')}): {e.get('error')}")
                if len(seen) >= 4:
                    break
            lines.append("[harness code] fix these before submitting — a hook that throws "
                         "is dead weight at sweep time.")
        inert = [k for k in calls if calls[k] > 0 and changed.get(k, 0) == 0
                 and k in ("system_prompt", "before_llm", "after_tool", "on_turn_end")]
        if inert and not errors:
            lines.append(f"[harness code] note: {', '.join(sorted(inert))} ran but never "
                         "changed anything — intended?")
        return "\n".join(lines)

    # Helpers
    @staticmethod
    def _is_tb(inst) -> bool:
        return bool(inst) and (inst.get("task_family") or "").lower() == "tb"

    def _task_state(self, ctx, task_id: str):
        """Return (inst, steps, recorded_pass). Looks up the transcript in the failed set
        first, then in the passing set (replay and ablate work on both)."""
        from verse.runtime.b1_evidence import transcript_steps
        inst = ctx.insts_by_tid.get(task_id)
        transcript = ctx.fail_transcripts.get(task_id)
        recorded_pass = False
        if transcript is None:
            transcript = getattr(ctx, "success_transcripts", {}).get(task_id)
            recorded_pass = transcript is not None
        if inst is None or transcript is None:
            return None, [], False
        return inst, transcript_steps(transcript), recorded_pass
