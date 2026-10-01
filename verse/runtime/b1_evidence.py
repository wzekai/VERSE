"""Replay-verified evidence about failed SWE tasks, for the optimizer.

Turns a round's failure transcripts into an evidence report in which every claim comes from an
executed replay, with no model judgment in the pipeline. The optimizer reads this report instead
of the raw trajectory tail. Three layers per failed task:

  1. Replay check: re-run the recorded bash commands in a fresh container. The replayed Φ must
     equal the recorded outcome (0); otherwise the task's step claims are withheld and the
     report says so.
  2. Failure-side leave-one-out (step_certificates): if dropping step k flips replay Φ from 0 to
     1, step k is harmful (it caused the failure). If Φ stays 0, step k is not a culprit alone.
  3. Read-step probe (phi_read): for the last read step before the last write, continue the
     episode from right after its observation in two settings, with the real observation and
     with the observation withheld, probe_n fresh continuations each, and report the difference
     in solve rates. real > masked means the information was there and the failure was a
     decision error; real == masked means the observation did not matter. Without this layer
     the report would only ever credit edit steps.

Write/read detection comes from verse.provenance, replay and leave-one-out from
step_certificates, Φ from swe_judge (via step_certificates._phi_of_patch), and containers from
swe_env. This module only parses transcripts and renders the report.

Cost (all capped): per round at most T2E_B1_EV_TASKS tasks get the full analysis, each with at
most T2E_B1_EV_MAX_LOO leave-one-out replays and 2 x T2E_B1_EV_PROBE_N continuations on one probe
step. All work shares one wall-clock deadline (T2E_B1_EV_BUDGET_S); tasks beyond the cap get
only the raw trajectory tail.

Environment variables (defaults): T2E_B1_EV_TASKS (2), T2E_B1_EV_PROBE_N (2),
T2E_B1_EV_BUDGET_S (1500), T2E_B1_EV_MAX_LOO (3).
"""
from __future__ import annotations

import json
import os
import time

from verse.runtime import swe_judge as _judge
from verse.runtime.step_certificates import compute_certificates, _phi_of_patch
from verse.runtime.swe_env import make_session, parse_action

try:
    from verse.provenance import shell_writes_file as _writes
except Exception:                                    # without it, phi_read_probe is skipped
    _writes = None


# Transcript to steps
def transcript_steps(transcript: list) -> list[dict]:
    """Parse an executor transcript into step records for replay.

    Transcripts alternate assistant and user turns. Each assistant turn is re-parsed with the
    same parse_action the episode used, so replay runs exactly the commands the container ran.
    Returns step records ({kind, command or patch, a_idx, o_idx}); the message indices are used
    to truncate the transcript.
    """
    steps = []
    for i, m in enumerate(transcript):
        if m.get("role") != "assistant":
            continue
        act = parse_action(m.get("content") or "")
        kind = act.get("kind")
        if kind == "bash":
            o_idx = i + 1 if i + 1 < len(transcript) and transcript[i + 1]["role"] == "user" else None
            # A turn may contain several commands and the episode loop runs all of them, so emit
            # one step per command; otherwise replay would skip state changes.
            for cmd in act.get("commands") or [act.get("command", "")]:
                steps.append({"kind": "bash", "command": cmd, "a_idx": i, "o_idx": o_idx})
        elif kind in ("patch", "submit"):
            steps.append({"kind": kind, "patch": act.get("patch", ""), "a_idx": i})
    return steps


# Counterfactual read-step probe
def _continue_episode(inst: dict, transcript: list, prefix_cmds: list[str], chat,
                      *, max_turns: int, deadline: float) -> float | None:
    """Continue an episode from a truncated transcript and return Φ of the resulting patch.

    Container state is rebuilt by replaying prefix_cmds (no LLM calls); the executor then
    continues from the (possibly masked) transcript. Returns None on infra failure or deadline.
    """
    image = _judge._image_name(inst)
    session = make_session(image=image)
    patch = ""
    try:
        if not session.start():
            return None
        for cmd in prefix_cmds:
            if time.monotonic() > deadline:
                return None
            session.execute(cmd)
        msgs = list(transcript)
        for _ in range(max_turns):
            if time.monotonic() > deadline:
                return None
            reply = chat(msgs)
            msgs.append({"role": "assistant", "content": reply})
            act = parse_action(reply)
            kind = act.get("kind")
            if kind == "bash":
                obs = session.execute(act.get("command", ""))
                msgs.append({"role": "user", "content": obs[:4096]})
            elif kind == "patch":
                patch = act.get("patch", "")
                break
            elif kind == "submit":
                patch = session.get_patch()
                break
            else:
                msgs.append({"role": "user", "content":
                             "Reply with a ```bash``` block, a ```diff``` patch, or `submit`."})
        else:
            patch = session.get_patch()
    except Exception:
        return None
    finally:
        session.close()
    run_timeout = int(os.environ.get("T2E_SWE_RUN_TIMEOUT", "1800"))
    return _phi_of_patch(inst, patch, run_timeout)


def phi_read_probe(inst: dict, transcript: list, steps: list[dict], chat,
                   *, probe_n: int, max_turns: int, deadline: float) -> dict | None:
    """Read-step probe on the last read-only bash step before the last write step.

    Both settings start from the same container state (the replayed prefix is identical): one
    keeps the real observation, the other masks it. Each continues probe_n times. Returns the
    record, or None when the trajectory has no read step before a write.
    """
    if _writes is None:
        return None
    bash = [s for s in steps if s["kind"] == "bash"]
    w_idx = [j for j, s in enumerate(bash) if _writes(s["command"])]
    # Anchor at the last bash write, or at the end of the bash steps when there is none (e.g.
    # a direct ```diff``` submission). Reads do not change container state, so the forced
    # prefix is sound either way.
    anchor = w_idx[-1] if w_idx else len(bash)
    reads_before = [j for j in range(anchor) if j not in w_idx and bash[j].get("o_idx") is not None]
    if not reads_before:
        return None
    j = reads_before[-1]
    probe = bash[j]
    cut = probe["o_idx"] + 1                     # keep everything through the probe's observation
    prefix_cmds = [bash[k]["command"] for k in range(j + 1)]   # state after the probe step
    worlds = {}
    for world in ("real", "masked"):
        msgs = [dict(m) for m in transcript[:cut]]
        if world == "masked":
            msgs[probe["o_idx"]] = {"role": "user", "content":
                                    "(observation withheld — decide from what you already know)"}
        phis = []
        for _ in range(int(probe_n)):
            if time.monotonic() > deadline:
                break
            phi = _continue_episode(inst, msgs, prefix_cmds, chat,
                                    max_turns=max_turns, deadline=deadline)
            if phi is not None:
                phis.append(phi)
        worlds[world] = phis
    if not worlds.get("real") or not worlds.get("masked"):
        return None
    r, m = worlds["real"], worlds["masked"]
    return {"probe_step": j, "command": probe["command"][:200],
            "real_phi": r, "masked_phi": m,
            "rate_real": sum(r) / len(r), "rate_masked": sum(m) / len(m),
            "rate_diff": sum(r) / len(r) - sum(m) / len(m)}


# Report builder
def build_processed_evidence(fails: dict, insts_by_tid: dict, chat_for, *,
                             max_turns: int = 20, tail_chars: int = 1500) -> tuple[str, dict]:
    """Build the evidence report for a round's failed tasks.

    fails: {tid: transcript}; insts_by_tid: {tid: instance spec}; chat_for(tid) returns a chat
    callable bound to the executor model (continuations must use the same frozen executor as
    the sweep). Returns (report_text, records); records is the JSON-serializable replay log
    behind every claim.
    """
    max_tasks = int(os.environ.get("T2E_B1_EV_TASKS", "2"))
    probe_n = int(os.environ.get("T2E_B1_EV_PROBE_N", "2"))
    budget_s = float(os.environ.get("T2E_B1_EV_BUDGET_S", "1500"))
    max_loo = int(os.environ.get("T2E_B1_EV_MAX_LOO", "3"))
    deadline = time.monotonic() + budget_s

    chunks, records = [], {}
    for tid, transcript in list(fails.items())[:max_tasks]:
        inst = insts_by_tid.get(tid)
        steps = transcript_steps(transcript)
        rec: dict = {"tid": tid, "n_steps": len(steps)}
        lines = [f"### FAILED {tid}  ({len(steps)} recorded steps)"]

        if inst is None or not steps:
            lines.append("- (no instance spec / no parsed steps — raw tail only)")
        else:
            # layers 1 and 2: replay check and failure-side leave-one-out
            swe_steps = [{"kind": s["kind"], "command": s.get("command", "")} for s in steps]
            try:
                cert = compute_certificates(swe_steps, inst, recorded_success=False,
                                            max_loo=max_loo, deadline=deadline)
            except Exception as e:
                cert = None
                lines.append(f"- step analysis unavailable ({type(e).__name__}: {e})")
            if cert is not None:
                rec["d_gate_ok"] = cert.trajectory_ok
                rec["cert_reason"] = cert.reason
                rec["loo"] = cert.results
                if cert.trajectory_ok:
                    lines.append(f"- REPLAY CHECK PASSED: re-executing all {len(swe_steps)} recorded "
                                 f"commands in a fresh container reproduces the failure (Φ=0). "
                                 f"Step claims below are replay-verified.")
                    harmful = [k for k, r in cert.results.items() if r.get("harmful")]
                    inert = [k for k, r in cert.results.items() if not r.get("harmful")]
                    for k in harmful:
                        cmd = swe_steps[k].get("command", "")[:160]
                        lines.append(f"- step {k} VERIFIED HARMFUL: dropping it flips replay Φ 0→1 "
                                     f"(this edit IS the failure): `{cmd}`")
                    if inert:
                        lines.append(f"- steps {inert} replayed-without individually: Φ stays 0 "
                                     f"(no single-step culprit among tested writes)")
                    if not cert.results:
                        lines.append("- no write-step LOO candidates within budget")
                else:
                    lines.append(f"- replay check FAILED ({cert.reason}) — step claims withheld "
                                 f"for this task (unverifiable world)")

            # layer 3: read-step probe
            try:
                pr = phi_read_probe(inst, transcript, steps, chat_for(tid),
                                    probe_n=probe_n, max_turns=max_turns, deadline=deadline)
            except Exception as e:
                pr = None
                lines.append(f"- phi_read probe errored ({type(e).__name__}: {e})")
            if pr is not None:
                rec["phi_read"] = pr
                n_r, n_m = len(pr["real_phi"]), len(pr["masked_phi"])
                lines.append(
                    f"- read-step probe (step {pr['probe_step']} `{pr['command'][:100]}`): continuing "
                    f"the episode fresh from after this observation solves {pr['rate_real']:.0%} "
                    f"({n_r} tries) with the real observation vs {pr['rate_masked']:.0%} ({n_m} tries) "
                    f"with it withheld."
                )
                if pr["rate_real"] > pr["rate_masked"]:
                    lines.append("  -> the information was there; the failure is a DECISION error "
                                 "after good evidence (verified by resampling).")
                elif pr["rate_real"] == 0 and pr["rate_masked"] == 0:
                    lines.append("  -> unrecoverable from this point in all tries — the failure is "
                                 "locked in earlier (or the task is beyond the executor).")

        text = "\n".join(f"[{m['role']}] {m['content']}" for m in transcript[1:])
        lines.append(f"#### trajectory tail\n...{text[-tail_chars:]}")
        chunks.append("\n".join(lines))
        records[tid] = rec

    # failures beyond the task cap get only the raw tail
    for tid, transcript in list(fails.items())[max_tasks:]:
        text = "\n".join(f"[{m['role']}] {m['content']}" for m in transcript[1:])
        chunks.append(f"### FAILED {tid} (evidence budget spent — raw tail)\n...{text[-tail_chars:]}")

    return "\n\n".join(chunks), records
