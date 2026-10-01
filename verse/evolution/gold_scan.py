"""List the SWE tasks whose reference (gold) patch fails on this execution backend.

A task can be graded on a backend only if its gold patch scores 1 there; otherwise a 0 reflects
the environment, not the agent. This is SWE-rebench's own collection filter ("the gold patch
must resolve the tests"), applied per backend. The docker backend passes almost every task. The
dockerless chroot backend fails tasks that need a TTY, a PID namespace or signal isolation (e.g.
ANSI-colour assertions, SIGINT handlers, nested pytest).

Run once per (backend, task set) before evaluation and pass the output list to
`verse.evolution.eval --invalid-tasks`, which leaves those tasks out of scoring. The full
{task_id: gold score} map is written next to it as <name>_full.json.

    python -m verse.evolution.gold_scan --parquet <swe_test.parquet> \
        --out /path/gold_invalid.json --conc 6
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from verse.evolution.driver import load_instances


def _gold_of(inst: dict, parquet_rows: dict) -> str:
    return (inst.get("gold_patch") or inst.get("patch")
            or parquet_rows.get(inst.get("instance_id"), ""))


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True, help="json list of gold-invalid task_ids")
    ap.add_argument("--conc", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    from verse.runtime.step_certificates import _phi_of_patch

    # the gold patch is in extra_info, under gold_patch or patch
    rows = {}
    df = pd.read_parquet(args.parquet)
    for _, row in df.iterrows():
        ei = row["extra_info"]
        d = ei if isinstance(ei, dict) else json.loads(ei)
        d = d.get("swe_instance", d)
        rows[d.get("instance_id")] = d.get("gold_patch") or d.get("patch") or ""

    insts = load_instances(args.parquet, 0, 42)
    print(f"[gold_scan] scoring GOLD on {len(insts)} tasks, conc={args.conc}", flush=True)

    def _one(inst):
        tid = inst.get("instance_id", "?")
        try:
            phi = _phi_of_patch(inst, _gold_of(inst, rows), args.timeout)
        except Exception as e:
            phi = None
            print(f"[gold_scan] {tid}: ERROR {type(e).__name__}", flush=True)
        print(f"[gold_scan] {tid}: gold_phi={phi}", flush=True)
        return tid, (phi if phi is not None else float("nan"))

    with ThreadPoolExecutor(max_workers=args.conc) as pool:
        results = dict(pool.map(_one, insts))

    invalid = sorted(t for t, v in results.items() if not (v == v and v >= 0.5))
    with open(args.out, "w") as f:
        json.dump(invalid, f, indent=1)
    with open(args.out.replace(".json", "_full.json"), "w") as f:
        json.dump({t: (v if v == v else None) for t, v in results.items()}, f, indent=1)
    print(f"[gold_scan] {len(results) - len(invalid)}/{len(results)} judgeable; "
          f"{len(invalid)} gold-invalid -> {args.out}", flush=True)
    print(f"[gold_scan] invalid: {invalid}", flush=True)


if __name__ == "__main__":
    main()
