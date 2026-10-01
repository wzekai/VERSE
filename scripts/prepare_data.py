#!/usr/bin/env python3
"""Build or verify the task-pool parquet files from public sources.

The engine reads each task from the `extra_info` column of a pool parquet (see load_instances
in verse/evolution/driver.py); the other columns are metadata. This script (a) builds a pool
parquet from a dump of raw task dicts and a split manifest (data/splits/*.json), and
(b) verifies that an existing parquet matches its manifest exactly (same ids, same count).

Sources of the raw task dicts (both public):
  SWE   the SWE-rebench dataset. Each task dict has instance_id, repo, base_commit,
        gold_patch, test_patch, FAIL_TO_PASS, PASS_TO_PASS and a public `swerebench/*`
        docker image. Export the tasks to a parquet or jsonl file with one task dict per
        row and pass it as --source.
  TB    Terminal-Bench task definitions. Each task dict has instance_id,
        problem_statement, task_family, tb_src, difficulty and category; the task
        environments are the upstream repositories' docker images.

Usage:
  python scripts/prepare_data.py build  --domain swe \\
      --source swe_rebench_dump.parquet \\
      --manifest data/splits/swe_train_v6.json --out data/swe_train_v6.parquet
  python scripts/prepare_data.py verify \\
      --manifest data/splits/swe_test.json --parquet data/swe_test.parquet

The manifests define the paper's splits: a rebuild from a newer upstream dump must give the
same instance_ids, or verification fails. Gold-patch screening is a separate step (see
data/README.md): run the gold scans and keep every task in the pool. Tasks whose gold patch
fails are left out of the denominator at evaluation time through their recorded id lists.
"""
from __future__ import annotations

import argparse
import json
import sys


def _json_default(o):
    """Normalize numpy scalars/arrays surfaced by parquet row.to_dict()."""
    import numpy as np
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def _load_manifest(path: str) -> dict:
    with open(path) as f:
        m = json.load(f)
    for k in ("domain", "role", "instance_ids"):
        if k not in m:
            raise SystemExit(f"manifest {path} missing key {k!r}")
    return m


def _unwrap(d: dict) -> dict:
    """Recover the raw task dict from a possibly-wrapped pool row (both source
    formats accept raw dumps and already-built pool rows)."""
    if "extra_info" not in d:
        return d
    ei = d["extra_info"]
    ei = dict(ei) if not isinstance(ei, str) else json.loads(ei)
    if "tb_instance" in ei:
        tb = ei["tb_instance"]
        ei = json.loads(tb) if isinstance(tb, str) else dict(tb)
    return ei


def _iter_source(path: str):
    """Yield raw task dicts from a parquet or jsonl dump."""
    if path.endswith(".jsonl"):
        with open(path) as f:
            for line in f:
                if line.strip():
                    yield _unwrap(json.loads(line))
        return
    import pandas as pd
    df = pd.read_parquet(path)
    for _, row in df.iterrows():
        yield _unwrap(row.to_dict())


def _pool_row(domain: str, task: dict) -> dict:
    """One pool row in the engine's schema; the driver reads the task from extra_info."""
    if domain == "tb":
        extra = {"tb_instance": json.dumps(task, default=_json_default)}
    else:
        extra = task
    return {
        "data_source": domain,
        "agent_name": "swe_agent" if domain == "swe" else "tb_agent",
        "ability": domain,
        "prompt": [{"role": "user",
                    "content": str(task.get("problem_statement", ""))[:20000]}],
        "reward_model": {"ground_truth": task.get("instance_id", ""),
                         "style": "rule"},
        "extra_info": extra,
    }


def cmd_build(args) -> int:
    import pandas as pd
    m = _load_manifest(args.manifest)
    want = list(m["instance_ids"])
    want_set = set(want)
    by_id = {}
    for task in _iter_source(args.source):
        tid = task.get("instance_id")
        if tid in want_set:
            by_id[tid] = task
    missing = [t for t in want if t not in by_id]
    if missing:
        print(f"ERROR: source is missing {len(missing)}/{len(want)} manifest tasks; "
              f"first few: {missing[:5]}", file=sys.stderr)
        return 1
    rows = [_pool_row(m["domain"], by_id[t]) for t in want]
    pd.DataFrame(rows).to_parquet(args.out)
    print(f"wrote {args.out}: {len(rows)} tasks ({m['domain']}/{m['role']})")
    return 0


def cmd_verify(args) -> int:
    m = _load_manifest(args.manifest)
    got = [task.get("instance_id") for task in _iter_source(args.parquet)]
    n_bad = got.count(None)
    if n_bad:
        print(f"MISMATCH: {n_bad} row(s) in {args.parquet} have no instance_id",
              file=sys.stderr)
        return 1
    want = list(m["instance_ids"])
    if sorted(got) == sorted(want):
        print(f"OK: {args.parquet} matches {args.manifest} "
              f"({len(want)} tasks, {m['domain']}/{m['role']})")
        return 0
    extra = sorted(set(got) - set(want))[:5]
    miss = sorted(set(want) - set(got))[:5]
    print(f"MISMATCH: {len(got)} tasks vs manifest {len(want)}; "
          f"unexpected={extra} missing={miss}", file=sys.stderr)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build a pool parquet from a source dump")
    b.add_argument("--domain", choices=["swe", "tb"], required=True)
    b.add_argument("--source", required=True,
                   help="parquet/jsonl of raw task dicts (or existing pool rows)")
    b.add_argument("--manifest", required=True)
    b.add_argument("--out", required=True)
    b.set_defaults(fn=cmd_build)
    v = sub.add_parser("verify", help="check a pool parquet against its manifest")
    v.add_argument("--manifest", required=True)
    v.add_argument("--parquet", required=True)
    v.set_defaults(fn=cmd_verify)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
