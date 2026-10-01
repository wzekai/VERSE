"""Per-step importance certificates for SWE trajectories by replay and leave-one-out.

For a recorded trajectory (a list of step records whose bash commands can be re-run), decide for
each tested step whether it is load-bearing, using actual replay and the executable outcome Φ
(1 = tests pass, 0 = not) rather than a model's opinion:

    load-bearing(k):  Φ(replay of all steps) == 1  and  Φ(replay without step k) == 0
    inert(k):         Φ(replay without step k) == 1
    harmful(k):       Φ(replay of all steps) == 0  and  Φ(replay without step k) == 1

This is the trace-minimization test (a step whose removal preserves Φ is irrelevant) restricted
to single steps; the multi-step search is in evolution/attribute.py. No model call is needed.

Building blocks:
  * Candidate steps come from verse.provenance.shell_writes_file. If verse.provenance cannot be
    imported, compute_certificates returns no certificates; there is no fallback classifier.
  * Replay re-runs the recorded bash commands in a fresh container, using the same session
    code (swe_env.make_session) that produced the rollout. It makes no LLM calls.
  * Φ is swe_judge's eval script. Its internals are called directly instead of compute_score,
    so an infra failure returns None and cannot be coerced to 0 (T2E_SWE_INFRA_AS_ZERO) and
    mistaken for a Φ flip.

Determinism check: before any per-step claim, the full command list is replayed and Φ(full) must
equal the recorded outcome. If it does not (flaky tests, network commands, container drift), the
trajectory gets no certificates and `reason` says why.

Cost: one trajectory takes 1 + (number of leave-one-out candidates) replays, each re-running the
recorded commands in one container plus one Φ eval container. Callers choose which trajectories
to certify; this module caps the candidates and enforces the deadline.

Environment variables (all optional):
    T2E_CERT_MAX_LOO         max leave-one-out candidates per trajectory (default 3)
    T2E_CERT_CACHE_DIR       persist (instance, command-hash) -> Φ on disk (default: memory only)
    T2E_SWE_EXEC_TIMEOUT     per-command timeout during replay (shared with swe_env; default 120)
    T2E_SWE_RUN_TIMEOUT      per-Φ-eval docker timeout (shared with swe_judge; default 1800)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field

logger = logging.getLogger("verse.runtime")

# Write detection comes from verse.provenance; there is no local fallback.
try:
    from verse.provenance import shell_writes_file as _t2t_shell_writes_file
    _HAVE_T2T = True
except Exception:  # compute_certificates refuses to run without it
    _t2t_shell_writes_file = None
    _HAVE_T2T = False
try:  # without write tiers, candidates keep their step order
    from verse.provenance import shell_write_tier as _t2t_shell_write_tier
except Exception:
    _t2t_shell_write_tier = None

from verse.runtime import swe_judge as _judge
from verse.runtime.swe_env import make_session


@dataclass
class StepCertificates:
    """Result of certifying one trajectory. `results[k]` exists only for steps actually replayed."""
    trajectory_ok: bool = False          # the full replay reproduced the recorded outcome
    reason: str = ""                     # why not ok, or why stopped early ("budget", ...)
    phi_full: float | None = None        # Φ of the full-command replay (the baseline)
    results: dict = field(default_factory=dict)   # step_idx -> {phi_without, load_bearing|harmful}
    n_phi_evals: int = 0
    n_candidates: int = 0
    elapsed_s: float = 0.0


# Φ of one patch
def _phi_of_patch(inst: dict, patch: str, timeout: int) -> float | None:
    """Executable Φ of a candidate patch via swe_judge's script builder, docker run and log parser.

    Returns 1.0 if resolved, 0.0 if not, and None on infra failure. Infra failures are never
    coerced to 0, so they cannot look like a Φ flip.
    """
    if not (patch or "").strip():
        return 0.0   # empty patch = genuinely unresolved (a valid outcome, not infra)
    node_ids = list(inst.get("FAIL_TO_PASS", [])) + list(inst.get("PASS_TO_PASS", []))
    if not node_ids:
        return None
    script = _judge._build_eval_script(inst, node_ids)
    log_text = _judge._run_in_docker(_judge._image_name(inst), script, patch,
                                     inst.get("test_patch", ""), timeout)
    if log_text is None or "T2E_TEST_PATCH_FAILED" in log_text:
        return None
    return 1.0 if _judge._resolved_from_log(inst, log_text) else 0.0


# Replay (forced prefix)
def _replay_patch(image: str, commands: list[str], deadline: float) -> str | None:
    """Re-run the recorded bash commands in order in a fresh container and return the git diff.

    No LLM calls. Returns None on container failure or if the deadline passes mid-replay (an
    infra outcome, not a verdict).
    """
    session = make_session(image=image)
    try:
        if not session.start():
            return None
        for cmd in commands:
            if time.monotonic() > deadline:
                return None
            session.execute(cmd)   # obs discarded; only the filesystem state matters for the patch
        return session.get_patch()
    except Exception as e:
        logger.warning(f"[cert] replay failed: {e}")
        return None
    finally:
        session.close()


# Cache
_CACHE: dict = {}


def _cache_key(instance_id: str, commands: list[str]) -> str:
    h = hashlib.sha256("\n".join(commands).encode("utf-8", "replace")).hexdigest()[:24]
    return f"{instance_id}::{h}"


def _cache_get(key: str):
    if key in _CACHE:
        return _CACHE[key]
    d = os.environ.get("T2E_CERT_CACHE_DIR", "")
    if d:
        p = os.path.join(d, key.replace("/", "_") + ".json")
        if os.path.exists(p):
            try:
                v = json.load(open(p)).get("phi")
                _CACHE[key] = v
                return v
            except Exception:
                pass
    return "MISS"


def _cache_put(key: str, phi) -> None:
    _CACHE[key] = phi
    d = os.environ.get("T2E_CERT_CACHE_DIR", "")
    if d:
        try:
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, key.replace("/", "_") + ".json"), "w") as f:
                json.dump({"phi": phi}, f)
        except Exception:
            pass


def _replay_phi(inst: dict, image: str, commands: list[str], deadline: float,
                run_timeout: int, cert: StepCertificates) -> float | None:
    """Replay the commands, then return Φ of the resulting patch.

    Results are cached by (instance, command hash), so a repeated command list skips docker.
    """
    key = _cache_key(inst.get("instance_id", ""), commands)
    hit = _cache_get(key)
    if hit != "MISS":
        return hit
    patch = _replay_patch(image, commands, deadline)
    if patch is None:
        return None                      # infra failure: not cached (transient)
    phi = _phi_of_patch(inst, patch, run_timeout)
    cert.n_phi_evals += 1
    if phi is not None:
        _cache_put(key, phi)
    return phi


# Certificate runner
def compute_certificates(steps: list[dict], inst: dict, *, recorded_success: bool,
                         max_loo: int | None = None, deadline: float | None = None) -> StepCertificates:
    """Certify the steps of one trajectory by replay and leave-one-out.

    `steps` are step records in order ({"kind": "bash"|"patch"|..., "command": ...});
    `inst` is the SWE instance spec (image, FAIL_TO_PASS, PASS_TO_PASS, test_patch, ...).

    recorded_success=True: the full replay must give Φ = 1; a step whose removal flips Φ to 0
    is load-bearing (results[k]["load_bearing"]).
    recorded_success=False: the full replay must give Φ = 0; a step whose removal flips Φ to 1
    is harmful (results[k]["harmful"]). Such steps are rare, since most failures are not one
    step away from passing.
    """
    t0 = time.monotonic()
    cert = StepCertificates()
    if deadline is None:
        deadline = t0 + 900.0
    if not _HAVE_T2T:
        cert.reason = "verse.provenance not importable (shell_writes_file needed)"
        return cert
    inst = _judge._as_instance(inst) or {}
    image = _judge._image_name(inst)
    if not inst.get("instance_id") or not image:
        cert.reason = "no instance spec / image"
        return cert

    if any(s.get("kind") == "patch" for s in steps):
        # The executor submitted a ```diff``` directly, which the episode loop prefers over the
        # container's git diff. Re-running bash commands cannot reproduce it, so skip.
        cert.reason = "direct diff submission (bash replay cannot reproduce)"
        return cert
    bash_steps = [(k, s.get("command", "")) for k, s in enumerate(steps)
                  if s.get("kind") == "bash" and (s.get("command") or "").strip()]
    if not bash_steps:
        cert.reason = "no bash steps recorded"
        return cert
    commands_full = [c for _, c in bash_steps]
    run_timeout = int(os.environ.get("T2E_SWE_RUN_TIMEOUT", "1800"))

    # Determinism check: the full replay must reproduce the recorded outcome.
    phi_full = _replay_phi(inst, image, commands_full, deadline, run_timeout, cert)
    cert.phi_full = phi_full
    cert.elapsed_s = time.monotonic() - t0
    if phi_full is None:
        cert.reason = "infra during full replay"
        return cert
    expect = 1.0 if recorded_success else 0.0
    if abs(phi_full - expect) > 0.5:
        cert.reason = "nondeterministic (full replay did not reproduce recorded outcome)"
        return cert
    cert.trajectory_ok = True

    # Leave-one-out on write steps chosen by verse.provenance. Candidates are sorted by write
    # tier so that in-repo content edits (tier 2, what the patch is made of) are tested before
    # incidental writes (tier 1: pip, dd, ...); otherwise a small max_loo can be used up before
    # the actual edit step is reached.
    if max_loo is None:
        max_loo = int(os.environ.get("T2E_CERT_MAX_LOO", "3"))
    writes = [(k, c) for k, c in bash_steps if _t2t_shell_writes_file(c)]
    if _t2t_shell_write_tier is not None:
        writes.sort(key=lambda kc: -_t2t_shell_write_tier(kc[1]))   # stable: tier 2 first
    candidates = writes[:max_loo]
    cert.n_candidates = len(candidates)
    for k, _cmd in candidates:
        if time.monotonic() > deadline:
            cert.reason = "budget"
            break
        loo_commands = [c for kk, c in bash_steps if kk != k]
        phi_wo = _replay_phi(inst, image, loo_commands, deadline, run_timeout, cert)
        if phi_wo is None:
            continue   # infra failure on this candidate only; others may still certify
        if recorded_success:
            cert.results[k] = {"phi_without": phi_wo, "load_bearing": phi_wo < 0.5}
        else:
            # failure side: dropping step k fixes the run, so step k is harmful
            cert.results[k] = {"phi_without": phi_wo, "harmful": phi_wo > 0.5}
    cert.elapsed_s = time.monotonic() - t0
    return cert
