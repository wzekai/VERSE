"""Attribution with trace minimization: execution-verified diagnosis.

Three judgments in the evolution loop are otherwise left to an LLM guess: (1) what the root
cause is, (2) whether an edit will work, and (3) whether the last round made progress.
fix_probe answers (2). This module answers (1) and produces the failure-mode fingerprints that
the training audit (`update_ledger`) uses to answer (3).

For each representative failure it runs failure-side trace minimization on the evolution
transcript: an LLM proposes a minimal keep-set of commands, a replay in a fresh environment
checks whether that subset still reproduces the failure, and one correction round follows.
Replays use the same executable sessions as the probes (swe_env.make_session for SWE,
TBContainerSession for TB).

Products per minimized failure:
    core            the minimal command subset that still reproduces the agent's blocking
                    error (typically 1-5 commands out of ~50)
    classification  omission: the blocking error reproduces from (near-)fresh environment
                                state; the fix was never written, so teach the missing
                                behavior
                    commission: reproduction needs earlier agent commands; those steps
                                cause the blocker, so ban or replace that action
    fingerprint     normalized error signature plus core, identifying the failure mode; the
                    training audit matches it against every later sweep at no extra cost

The LLM attribution is kept: its proposal seeds the keep-set and its reasoning goes into the
report, but an accepted core carries an executed replay record.

Runs in the evidence phase of VERSE configurations only (ctx.probes set and an `attribute:`
config section), outside the optimizer's probe budget.
"""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from verse.claims import Claim, ExperimentRecord


# Failure signatures
_ERR_TYPE_RE = re.compile(r"\b([A-Z]\w+(?:Error|Exception|Warning))\b")
_TOKEN_STRIP_RE = re.compile(r"(/[\w./-]+)|(0x[0-9a-f]+)|(\b\d+\b)|('[^']*')|(\"[^\"]*\")")


def error_signature(last_error: str) -> str:
    """Normalize an error line into a stable failure-mode key: the error class plus a residue
    with paths, numbers and literals stripped. Purely mechanical, so training-audit matching
    can be reproduced from the round journal without a model call."""
    if not last_error:
        return ""
    m = _ERR_TYPE_RE.search(last_error)
    cls = m.group(1) if m else ""
    residue = _TOKEN_STRIP_RE.sub("<X>", last_error)
    residue = re.sub(r"\s+", " ", residue).strip()[:120]
    return f"{cls}|{residue}" if cls else residue


def transcript_signature(transcript: list) -> str:
    from verse.evolution.evidence import _last_error_line
    return error_signature(_last_error_line(transcript))


# Replay
def _make_replay_session(inst: dict):
    """Fresh executable session for replaying a transcript's commands. SWE replays into a
    testbed container (swe_env), TB into the task's compose environment (tb_env). Both expose
    the start()/execute()/close() interface that minimization relies on."""
    if (inst.get("task_family") or "").lower() == "tb":
        from verse.runtime.tb_env import TBContainerSession
        return TBContainerSession(inst.get("instance_id", "?"), src=inst.get("tb_src", ""))
    from verse.runtime import swe_judge as _judge
    from verse.runtime.swe_env import make_session
    return make_session(image=_judge._image_name(inst))


def _replay_capture(inst: dict, commands: list, exec_timeout: int = 120) -> list:
    """Replay `commands` in a fresh session and return the per-command observations.
    No LLM calls. Returns None on session failure (an infrastructure problem, not a
    verdict)."""
    session = _make_replay_session(inst)
    try:
        if not session.start():
            return None
        return [session.execute(c) or "" for c in commands]
    except Exception:
        return None
    finally:
        try:
            session.close()
        except Exception:
            pass


def _tb_replay_phi(inst: dict, commands: list) -> float | None:
    """TB counterpart of step_certificates._replay_phi: replay the kept commands in a fresh TB
    session and score with the task's own run-tests.sh. TB tasks are judged on container state
    (there is no patch), so a subset is checked by re-running the tests on the resulting state.
    Returns None on infrastructure failure."""
    from verse.runtime.tb_env import TBContainerSession
    session = TBContainerSession(inst.get("instance_id", "?"), src=inst.get("tb_src", ""))
    try:
        if not session.start():
            return None
        for c in commands:
            session.execute(c)
        return session.score()
    except Exception:
        return None
    finally:
        try:
            session.close()
        except Exception:
            pass


def _reproduces(obs: list, sig: str) -> bool:
    """Whether the replayed output shows the same failure mode. Matches on the signature's
    error class when it has one (stable across path and line-number changes), otherwise on a
    prefix of the key."""
    if obs is None or not sig:
        return False
    joined = error_signature("\n".join(o[-2000:] for o in obs if o))
    cls = sig.split("|", 1)[0]
    if cls:
        return cls == joined.split("|", 1)[0] or any(
            cls in error_signature(o[-2000:]) for o in obs if o)
    return sig[:60] in joined


# LLM proposal
def _propose_core(model: str, regions: str, tid: str, steps: list, sig: str,
                  feedback: str = "") -> tuple:
    """One attribution call: the LLM reads the step table and proposes (a) the minimal
    keep-set of step indices that reproduces the blocking error and (b) a root-cause reading.
    Returns (indices or None, rationale, pre_existing). The rationale is kept as hypothesis
    text; the replay check decides whether the core is accepted."""
    from verse.runtime.executor import _invoke_bedrock, _json_candidates, _TEMP_DEPRECATED
    table = "\n".join(f"[{i}] {s.get('command', '')[:180]}"
                      for i, s in enumerate(steps) if s["kind"] == "bash")
    prompt = (
        f"A frozen coding agent FAILED task {tid}. Its blocking error signature:\n  {sig}\n\n"
        f"Its bash commands, in order:\n"
        f"{table[:6000] + (f'<<DISPLAY-CAP: table is {len(table)} chars>>' if len(table) > 6000 else '')}\n\n"
        "Following trace-attribution: propose the SMALLEST subset of these commands that, "
        "replayed in order in a fresh container, still REPRODUCES that blocking error. "
        "Include setup commands (cd/env activation) the trigger needs. If the error would "
        "reproduce from the fresh environment with just the single triggering command "
        "(pre-existing breakage — the agent never wrote the fix), keep ONLY that trigger.\n"
        + (f"\nPrior candidate failed the replay check: {feedback}\n" if feedback else "") +
        '\nReply ONLY JSON: {"keep_indices": [..], "root_cause": "<=60 words", '
        '"pre_existing": true|false}')
    body = {"max_tokens": 1500,
            "messages": [{"role": "user", "content": prompt}]}
    if not any(t in model for t in _TEMP_DEPRECATED):
        body["temperature"] = 0.0
    try:
        reply = _invoke_bedrock(model, regions, body)
        for cand in _json_candidates(reply):
            try:
                d = json.loads(cand)
                idxs = [int(i) for i in d.get("keep_indices", [])
                        if 0 <= int(i) < len(steps)]
                if idxs:
                    return idxs, str(d.get("root_cause", ""))[:400], bool(
                        d.get("pre_existing", False))
            except Exception:
                continue
    except Exception:
        pass
    return None, "", False


def _tb_fail_signature(inst: dict, transcript: list) -> str:
    """TB failure signature. SWE failures show an error line in the agent's observations
    (traceback, pytest), but a TB failure is usually an omission: the required end state was
    never reached, so the observations carry no error. run_tb_task appends a
    'TEST FAILURES: ...' / 'TEST RESULT: ...' line (from run-tests.sh) at the end of the
    transcript, and that verdict is the TB failure mode. Without it, falls back to the
    observation-based signature."""
    for m in reversed(transcript):
        c = m.get("content") or ""
        if isinstance(c, str) and c.startswith(("TEST FAILURES:", "TEST RESULT:")):
            return error_signature("tb|" + c[:200])
    return transcript_signature(transcript)


def _minimize_failure_tb(ctx, task_id: str) -> dict | None:
    """TB failure-side minimization, checked by re-running the task's tests (the
    container-state counterpart of the SWE error-reproduction check). The full-trace replay
    must reproduce the failure: if a fresh replay of every recorded command passes the tests,
    the failure is nondeterministic and nothing is certified (None), as on the SWE side. When
    the failure reproduces:
      subset also fails: omission. The required end state was never produced; the failing
                         subset is the executed core.
      subset passes:     commission. The commands excluded from the subset break the task
                         (the agent reached a working state and destroyed it); the core is
                         the complement, shown by the pass/fail contrast."""
    from verse.runtime.b1_evidence import transcript_steps
    inst = ctx.insts_by_tid.get(task_id)
    transcript = (ctx.fail_transcripts or {}).get(task_id)
    if inst is None or transcript is None:
        return None
    steps = transcript_steps(transcript)
    bash_idx = [i for i, s in enumerate(steps) if s["kind"] == "bash"]
    if not bash_idx:
        return None
    sig = _tb_fail_signature(inst, transcript)
    model = (ctx.executor or {}).get("intervener_model") or (ctx.executor or {}).get("model")
    regions = (ctx.executor or {}).get("regions", "us-west-2,us-east-1,us-east-2")
    keep, rationale, pre = _propose_core(model, regions, task_id, steps, sig or "TB task failed")
    if keep is None:
        keep, rationale, pre = bash_idx, "", False
    keep = sorted(set(keep))
    cmds = [steps[i].get("command", "") for i in keep if steps[i]["kind"] == "bash"]
    phi_sub = _tb_replay_phi(inst, cmds)
    replays = 1
    if phi_sub is None:
        return None                              # infra, not a verdict
    full_cmds = [steps[i].get("command", "") for i in bash_idx]
    phi_full = phi_sub if set(keep) == set(bash_idx) else _tb_replay_phi(inst, full_cmds)
    if set(keep) != set(bash_idx):
        replays += 1
    if phi_full is None:
        return None                              # infra, not a verdict
    if phi_full >= 0.5:
        return None       # recorded failure does not reproduce: nondeterministic, certify nothing
    if phi_sub < 0.5:
        classification = "omission"              # the core also fails: the fix is missing
        core_idx, core_cmds = keep, cmds
        outcome = (f"failure REPRODUCED (full replay phi={phi_full:g}, "
                   f"{len(keep)}-command core phi={phi_sub:g}) — the required end state was "
                   "never produced (omission)")
    else:
        classification = "commission"            # dropping the complement turns fail into pass
        core_idx = [i for i in bash_idx if i not in set(keep)]
        core_cmds = [steps[i].get("command", "") for i in core_idx]
        outcome = (f"failure REPRODUCED by the full trace (phi={phi_full:g}) but the "
                   f"{len(keep)}-command subset PASSES (phi={phi_sub:g}) — the "
                   f"{len(core_idx)} excluded command(s) break the task (commission)")
    rec = ExperimentRecord(
        "minimize",
        f"replayed the {len(keep)}/{len(bash_idx)}-command core and the full trace in fresh "
        f"TB containers and re-ran the task's tests",
        outcome, "supports", task_id,
        {"sig": sig, "core_indices": list(core_idx), "classification": classification,
         "seed_source": "llm_propose", "n_replays": replays,
         "phi_subset": phi_sub, "phi_full": phi_full})
    return {"tid": task_id, "sig": sig, "core_indices": list(core_idx),
            "core_commands": core_cmds, "classification": classification,
            "rationale": rationale, "record": rec, "replays": replays}


# Minimization, failure side
def minimize_failure(ctx, task_id: str, *, refine_rounds: int = 1) -> dict | None:
    """Failure-side minimization of an evolution transcript.

    Returns {tid, sig, core_indices, core_commands, classification, rationale, record,
    replays}, or None when the task is unusable (no signature, no commands, or an
    infrastructure failure). Steps: a one-shot LLM proposal of a minimal subset, a replay
    check, up to refine_rounds corrections, a suffix search, and finally the full command
    list (tagged, no compression)."""
    from verse.runtime.b1_evidence import transcript_steps
    inst = ctx.insts_by_tid.get(task_id)
    transcript = (ctx.fail_transcripts or {}).get(task_id)
    if inst is None or transcript is None:
        return None
    if (inst.get("task_family") or "").lower() == "tb":
        return _minimize_failure_tb(ctx, task_id)
    sig = transcript_signature(transcript)
    steps = transcript_steps(transcript)
    bash_idx = [i for i, s in enumerate(steps) if s["kind"] == "bash"]
    if not sig or not bash_idx:
        return None
    model = (ctx.executor or {}).get("intervener_model") or (ctx.executor or {}).get("model")
    regions = (ctx.executor or {}).get("regions", "us-west-2,us-east-1,us-east-2")
    replays = 0

    keep, rationale, pre = _propose_core(model, regions, task_id, steps, sig)
    seed_source = "llm_propose"
    if keep is None:
        keep, seed_source = bash_idx, "mechanical_fallback"  # full list, no compression
    feedback = ""
    accepted = False
    for _ in range(1 + max(0, refine_rounds)):
        cmds = [steps[i].get("command", "") for i in keep if steps[i]["kind"] == "bash"]
        obs = _replay_capture(inst, cmds)
        replays += 1
        if _reproduces(obs, sig):
            accepted = True
            break
        feedback = (f"kept {len(cmds)} command(s) but the replay did not show the error "
                    f"class of «{sig[:80]}»")
        keep2, rat2, pre2 = _propose_core(model, regions, task_id, steps, sig, feedback)
        if keep2 is None or set(keep2) == set(keep):
            break
        keep, rationale, pre = keep2, (rat2 or rationale), pre2
        seed_source = "llm_refined"
    if not accepted:
        # Suffix search before falling back to the full list: failures often reproduce from
        # a tail of the trace (the state that matters is what the last commands built).
        # Log-spaced suffixes can find a small core in at most 3 replays, whereas the
        # full-list fallback gives no compression.
        for k in (1, 4, 16):
            if k >= len(bash_idx):
                break
            cand = bash_idx[-k:]
            obs = _replay_capture(inst, [steps[i].get("command", "") for i in cand])
            replays += 1
            if _reproduces(obs, sig):
                keep, seed_source, accepted = cand, f"suffix_{k}", True
                break
    if not accepted:
        keep, seed_source = bash_idx, "mechanical_fallback"
        cmds = [steps[i].get("command", "") for i in keep]
        obs = _replay_capture(inst, cmds)
        replays += 1
        accepted = _reproduces(obs, sig)
        if not accepted:
            return None                       # not even the full trace reproduces

    # Classification: if the blocker reproduces from the last kept command alone (nothing
    # else the agent wrote), it is pre-existing breakage, i.e. omission. A full-length
    # fallback core supports no step-level claim (reproducing with everything kept says
    # nothing about which steps matter), so it is labeled "unknown", not "commission".
    uncompressed = seed_source == "mechanical_fallback" and len(keep) == len(bash_idx)
    classification = "unknown" if uncompressed else "commission"
    if not uncompressed:
        if len(keep) <= 1:
            classification = "omission"
        elif pre or len(keep) <= 3:
            tail = [steps[keep[-1]].get("command", "")]
            obs1 = _replay_capture(inst, tail)
            replays += 1
            if _reproduces(obs1, sig):
                classification = "omission"
                keep = [keep[-1]]
    core_cmds = [steps[i].get("command", "") for i in keep]
    outcome_by_class = {
        "omission": "pre-existing breakage: the fix was never written (omission)",
        "commission": "reproduction requires earlier agent steps: they cause the blocker "
                      "(commission)",
        "unknown": "NO COMPRESSION achieved — the error reproduces only with the full "
                   "command list, so no step-level cause is isolated (treat as "
                   "deterministic-failure confirmation only)",
    }
    rec = ExperimentRecord(
        "minimize",
        f"minimized the failed trajectory to {len(keep)}/{len(bash_idx)} commands and "
        f"replayed them in a fresh environment ({seed_source})",
        f"blocking error «{sig[:100]}» REPRODUCED by the {len(keep)}-command core — "
        + outcome_by_class[classification],
        "inconclusive" if uncompressed else "supports", task_id,
        {"sig": sig, "core_indices": list(keep), "classification": classification,
         "seed_source": seed_source, "n_replays": replays})
    return {"tid": task_id, "sig": sig, "core_indices": list(keep),
            "core_commands": core_cmds, "classification": classification,
            "rationale": rationale, "record": rec, "replays": replays}


def minimize_success(ctx, task_id: str) -> dict | None:
    """Success-side minimization: the smallest command subset that still passes (phi == 1),
    i.e. the behaviors worth distilling into skills. Subsets are scored with the cached
    replay engine (_replay_phi) on SWE and with _tb_replay_phi on TB."""
    import os
    from verse.runtime import swe_judge as _judge
    from verse.runtime.step_certificates import StepCertificates, _replay_phi
    from verse.runtime.b1_evidence import transcript_steps
    inst = ctx.insts_by_tid.get(task_id)
    transcript = (getattr(ctx, "success_transcripts", None) or {}).get(task_id)
    if inst is None or transcript is None:
        return None
    steps = transcript_steps(transcript)
    bash_idx = [i for i, s in enumerate(steps) if s["kind"] == "bash"]
    if not bash_idx:
        return None
    is_tb = (inst.get("task_family") or "").lower() == "tb"
    model = (ctx.executor or {}).get("intervener_model") or (ctx.executor or {}).get("model")
    regions = (ctx.executor or {}).get("regions", "us-west-2,us-east-1,us-east-2")
    keep, rationale, _ = _propose_core(
        model, regions, task_id, steps,
        "N/A — SUCCESS trajectory: keep the smallest set that still RESOLVES the task "
        "(all graded tests pass)")
    if keep is None:
        return None                            # no mechanical fallback on the success side
    timeout = int(os.environ.get("T2E_SWE_RUN_TIMEOUT", "1800"))
    deadline = time.monotonic() + 3600
    cert = StepCertificates()

    def _phi(indices):
        cmds = [steps[i].get("command", "") for i in indices if steps[i]["kind"] == "bash"]
        if is_tb:
            return _tb_replay_phi(inst, cmds)
        return _replay_phi(inst, _judge._image_name(inst), cmds, deadline, timeout, cert)

    phi = _phi(keep)
    if phi is None or phi < 0.5:
        keep = bash_idx                        # fall back to the full list (tagged below)
        phi = _phi(keep)
        if phi is None or phi < 0.5:
            return None
        seed_source = "mechanical_fallback"
    else:
        seed_source = "llm_propose"
    core_cmds = [steps[i].get("command", "") for i in keep if steps[i]["kind"] == "bash"]
    rec = ExperimentRecord(
        "minimize", f"minimized the PASSING trajectory to {len(keep)}/{len(bash_idx)} "
                    f"commands and replayed ({seed_source})",
        f"the {len(keep)}-command core still SOLVES the task (replay phi=1) — these "
        "behaviors are proven load-bearing; distill them into skills",
        "supports", task_id,
        {"core_indices": list(keep), "classification": "load_bearing",
         "seed_source": seed_source})
    return {"tid": task_id, "core_indices": list(keep), "core_commands": core_cmds,
            "classification": "load_bearing", "rationale": rationale, "record": rec}


# Entry point
def run_attribute_layer(ctx, *, max_fail: int = 4, max_success: int = 1,
                        wall_s: float = 1500.0, seen_sigs=None) -> tuple:
    """Evidence-phase entry point (VERSE configurations only; the driver gates on ctx.probes).

    Picks exemplar failures (largest error-signature clusters first, so each failure mode
    rather than each task is minimized) plus up to max_success successes, minimizes them
    concurrently, and returns (report_section, claims, fingerprints). Never raises; partial
    results are fine.

    seen_sigs: signatures already minimized in earlier rounds (training-audit keys). Unseen
    modes are picked first, since minimizing a persistent mode again only repeats what the
    audit already records; seen modes fill the remaining slots when there are fewer than
    max_fail new ones."""
    fails = ctx.fail_transcripts or {}
    clusters = defaultdict(list)
    # Cluster with _tb_fail_signature, the function that produced the TB fingerprints in
    # seen_sigs; otherwise unseen-first selection would never match on TB. For SWE it falls
    # back to the observation-based signature.
    for tid, tr in fails.items():
        clusters[_tb_fail_signature(None, tr)].append(tid)
    clusters.pop("", None)
    ordered = sorted(clusters.items(), key=lambda kv: -len(kv[1]))
    seen = set(seen_sigs or ())
    fresh = [(tids[0], sig, len(tids)) for sig, tids in ordered if sig not in seen]
    stale = [(tids[0], sig, len(tids)) for sig, tids in ordered if sig in seen]
    fail_targets = (fresh + stale)[:max_fail]
    succ = getattr(ctx, "success_transcripts", None) or {}
    succ_targets = sorted(succ)[:max_success]

    t0 = time.monotonic()
    results, s_results = [], []

    def _f(job):
        tid, sig, n = job
        if time.monotonic() - t0 > wall_s:
            return None
        try:
            r = minimize_failure(ctx, tid)
        except Exception:
            return None
        if r:
            r["cluster_size"] = n
        return r

    def _s(tid):
        if time.monotonic() - t0 > wall_s:
            return None
        try:
            return minimize_success(ctx, tid)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = [r for r in pool.map(_f, fail_targets) if r]
        s_results = [r for r in pool.map(_s, succ_targets) if r]

    claims, fingerprints, lines = [], [], []
    lines.append("\n\n## Attribute layer — execution-verified diagnosis (minimize)")
    lines.append("Each entry below was REPLAYED: the core is proven to reproduce the "
                 "blocking error, not guessed. omission = the fix was never written (teach "
                 "the missing behavior; validate with fix_probe). commission = the listed "
                 "steps cause the blocker (ban/replace that action; validate with "
                 "substitute). unknown = the error only reproduces with the FULL command "
                 "list (no step isolated — deterministic-failure confirmation only; do not "
                 "build a step-level diagnosis on it).")
    for r in results:
        lines.append(f"\n### {r['tid']}  [{r['classification'].upper()}]  "
                     f"(mode hits {r['cluster_size']} tasks this sweep)")
        lines.append(f"signature: {r['sig'][:140]}")
        if r["rationale"]:
            lines.append(f"attribution hypothesis (LLM): {r['rationale']}")
        lines.append(f"executed minimal core ({len(r['core_commands'])} of the trajectory's "
                     "commands):")
        lines += [f"  $ {c[:160]}" for c in r["core_commands"][:6]]
        claims.append(Claim(statement=(f"{r['tid']} blocking failure mode "
                                       f"[{r['classification']}]: {r['rationale'] or r['sig'][:120]}"),
                            task_id=r["tid"], evidence=[r["record"]],
                            tags={"attribute": True,
                                  "classification": r["classification"], "sig": r["sig"]}))
        fingerprints.append({"sig": r["sig"], "example_tid": r["tid"],
                             "classification": r["classification"],
                             "core_commands": r["core_commands"][:6],
                             "cluster_size": r["cluster_size"]})
    for r in s_results:
        lines.append(f"\n### {r['tid']}  [SUCCESS — LOAD-BEARING CORE]")
        lines.append(f"the {len(r['core_commands'])}-command core still solves the task "
                     "(replayed, phi=1) — distill exactly these behaviors:")
        lines += [f"  $ {c[:160]}" for c in r["core_commands"][:8]]
        claims.append(Claim(statement=(f"{r['tid']} minimal winning core "
                                       f"({len(r['core_commands'])} commands) — proven "
                                       "load-bearing by replay"),
                            task_id=r["tid"], evidence=[r["record"]],
                            tags={"attribute": True, "classification": "load_bearing"}))
    if not results and not s_results:
        return "", [], []
    return "\n".join(lines), claims, fingerprints


# Training audit (failure-mode ledger)
def update_ledger(ledger_path: str, round_idx: int, fail_transcripts: dict,
                  new_fingerprints: list) -> str:
    """Training audit: register new fingerprints, then record which known failure modes
    appear among this sweep's failures (mechanical signature matching over trajectories that
    were already collected; no extra execution). Returns the report section for the
    optimizer.

    The ledger feeds only the next round's optimizer report and the per-mode extinction
    curves. It never reaches the gate or the frontier selection, which read only the
    validation set."""
    import os
    ledger = {}
    if os.path.exists(ledger_path):
        try:
            ledger = json.load(open(ledger_path))
        except Exception:
            ledger = {}
    for fp in new_fingerprints or []:
        ent = ledger.setdefault(fp["sig"], {
            "first_round": round_idx, "example_tid": fp["example_tid"],
            "classification": fp["classification"],
            "core_commands": fp.get("core_commands", []), "hits": {}})
        ent.setdefault("classification", fp["classification"])
    sweep_sigs = defaultdict(list)
    for tid, tr in (fail_transcripts or {}).items():
        # same signature function as attribution, so TB failures (TEST FAILURES line) match
        # their registered fingerprints; SWE uses the observation-based signature
        sweep_sigs[_tb_fail_signature(None, tr)].append(tid)
    for sig, ent in ledger.items():
        ent.setdefault("hits", {})[str(round_idx)] = sorted(sweep_sigs.get(sig, []))
    with open(ledger_path, "w") as f:
        json.dump(ledger, f, indent=1)

    lines = ["\n\n## Failure-mode ledger (execution-derived, mechanically audited)"]
    lines.append("Per-mode attendance across sweeps. EXTINCT modes were fixed by kept "
                 "edits; GROWING/persistent modes are the live battlefield. A mode whose "
                 "tasks migrated to a new signature is partial progress phi cannot see.")
    alive = []
    for sig, ent in ledger.items():
        hist = [(int(r), len(t)) for r, t in sorted(ent.get("hits", {}).items(),
                                                    key=lambda kv: int(kv[0]))]
        if not hist:
            continue
        curve = " -> ".join(f"r{r}:{n}" for r, n in hist)
        now = hist[-1][1]
        status = "EXTINCT" if now == 0 else (
            "GROWING" if len(hist) > 1 and now > hist[-2][1] else "alive")
        alive.append((now, f"- [{status}] «{sig[:90]}» ({ent.get('classification', '?')}, "
                           f"e.g. {ent.get('example_tid', '')}): {curve}"))
    for _, ln in sorted(alive, key=lambda x: -x[0]):
        lines.append(ln)
    return "\n".join(lines) if len(lines) > 2 else ""
