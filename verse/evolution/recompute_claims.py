"""Recompute each round's claim status from the optimizer's audit log, without rerunning.

Claim.status is REFUTED if any experiment record refutes the claim. If all records of an
episode are pooled into one claim, a refuted draft (the normal draft, refute, revise loop)
marks the round REFUTED even when the final submission was verified, and that label reaches
the next round through the history. This script recomputes the status of each round's final
submission, counting only the fix_probe records whose edits hash (edits_sha) matches the
submitted edits.

For each round it derives the sha of the final submitted edits (from the last submit_proposal
call in round_*/intervener_audit.jsonl), the fix_probe results on the final edits and on
drafts, and the corrected status. Writes <arm-dir>/claims_recompute.json and prints a table
per run. Read-only over journals: rounds.jsonl is never modified.

Usage: python -m verse.evolution.recompute_claims --arm-dir /path/to/run [--arm-dir ...]
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def _round_audit(path: str) -> list:
    events = []
    if not os.path.exists(path):
        return events
    for ln in open(path):
        try:
            events.append(json.loads(ln))
        except Exception:
            continue
    return events


def _final_edits(events: list) -> list | None:
    """Edits of the last submit_proposal call (the repair loop may resubmit)."""
    edits = None
    for e in events:
        if e.get("kind") == "tool_call" and e.get("tool") == "submit_proposal":
            edits = (e.get("args") or {}).get("edits")
    return edits


def _probe_records(events: list) -> list:
    """One {tool, tag, edits_sha, out} record per tool result (fix_probe args matched by turn)."""
    args_by_turn = {}
    out = []
    for e in events:
        if e.get("kind") == "tool_call" and e.get("tool") == "fix_probe":
            args_by_turn[e.get("turn")] = e.get("args") or {}
        if e.get("kind") != "tool_result":
            continue
        tool = e.get("tool")
        if tool not in ("fix_probe", "ablate", "substitute", "replay"):
            continue
        head = e.get("out_head", "")
        tag = head[1:head.index("]")] if head.startswith("[") and "]" in head else "?"
        sha = None
        if tool == "fix_probe":
            a = args_by_turn.get(e.get("turn")) or {}
            from verse.evolution.probes import edits_sha
            sha = edits_sha(a.get("edits") or [])
        out.append({"tool": tool, "tag": tag, "edits_sha": sha, "out": head[:200]})
    return out


def recompute_arm(arm_dir: str) -> dict:
    from verse.evolution.probes import edits_sha
    rounds = {}
    journal = os.path.join(arm_dir, "rounds.jsonl")
    if os.path.exists(journal):
        for ln in open(journal):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if isinstance(r.get("round"), int) and r["round"] > 0:
                rounds[r["round"]] = r
    result = {}
    for rd in sorted(glob.glob(os.path.join(arm_dir, "round_*", "intervener_audit.jsonl"))):
        rnd = int(rd.split("round_")[1].split(os.sep)[0])
        events = _round_audit(rd)
        edits = _final_edits(events)
        sha_final = edits_sha(edits or [])
        probes = _probe_records(events)
        on_final = [p for p in probes if p["tool"] == "fix_probe"
                    and p["edits_sha"] == sha_final]
        drafts = [p for p in probes if p["tool"] == "fix_probe"
                  and p["edits_sha"] not in (None, sha_final)]
        non_fp = [p for p in probes if p["tool"] != "fix_probe"]
        # corrected status of the final submission's claim
        tags_final = {p["tag"] for p in on_final}
        if "SUPPORTS" in tags_final and "REFUTES" not in tags_final:
            status = "VERIFIED"
        elif "REFUTES" in tags_final:
            status = "REFUTED"          # the submitted edits themselves were refuted
        else:
            status = "HYPOTHESIS"       # no conclusive fix_probe on the final edits
        old = rounds.get(rnd, {}).get("claims_summary", "")
        old_head = old.split("\n")[0] if old else ""
        v = rounds.get(rnd, {}).get("verdict") or {}
        result[rnd] = {
            "corrected_status": status, "old_summary_head": old_head,
            "final_edits_sha": sha_final,
            "fix_probe_on_final": [p["tag"] for p in on_final],
            "fix_probe_on_drafts": [p["tag"] for p in drafts],
            "other_probes": [f"{p['tool']}:{p['tag']}" for p in non_fp],
            "round_label": v.get("label"), "fixed": v.get("fixed"),
            "regressed": v.get("regressed"),
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-dir", action="append", required=True)
    args = ap.parse_args()
    for arm_dir in args.arm_dir:
        arm = os.path.basename(os.path.normpath(arm_dir))
        res = recompute_arm(arm_dir)
        out = os.path.join(arm_dir, "claims_recompute.json")
        with open(out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"\n=== {arm} (recomputed -> {out})")
        for rnd in sorted(res):
            r = res[rnd]
            was = "REFUTED" if "1 REFUTED" in r["old_summary_head"] else (
                "VERIFIED" if "1 VERIFIED" in r["old_summary_head"] else "HYPOTHESIS?")
            flag = "  <-- was mislabeled" if (was == "REFUTED" and
                                              r["corrected_status"] != "REFUTED") else ""
            print(f" r{rnd}: {was} -> {r['corrected_status']}"
                  f"  [round {r['round_label']}, final probes {r['fix_probe_on_final'] or '-'},"
                  f" draft probes {r['fix_probe_on_drafts'] or '-'}]{flag}")


if __name__ == "__main__":
    main()
