"""List the Terminal-Bench tasks whose reference solution fails on this execution backend.

The Terminal-Bench counterpart of gold_scan.py: a task can be graded only if its own reference
solution scores 1 here; otherwise a 0 reflects the environment, not the agent. Run once per
(backend, task set) before evaluation and pass the output list to
`verse.evolution.eval --invalid-tasks`.

Two Terminal-Bench solution layouts are supported:
  solution.sh / solution/solve.sh   a shell script run in the task container
  solution.yaml                     a list of {command: ...} steps, run in order

    python -m verse.evolution.tb_gold_scan --parquet data/tb_test.parquet \
        --out gold_invalid_tb.json --conc 6
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

from verse.evolution.driver import load_instances
from verse.runtime.tb_env import TBContainerSession


def _solution_steps(task_dir: str) -> list[str] | None:
    """Return the reference solution as a list of shell commands, or None if absent.

    Script solutions run as /solution/<script> in the container; the whole solution/ directory
    is copied there first, since some scripts read other files from it."""
    if os.path.exists(os.path.join(task_dir, "solution", "solve.sh")):
        return ["bash /solution/solve.sh"]
    if os.path.exists(os.path.join(task_dir, "solution.sh")):
        return ["bash /solution/solution.sh"]
    p = os.path.join(task_dir, "solution.yaml")
    if os.path.exists(p):
        import yaml
        steps = yaml.safe_load(open(p))
        return [s["command"] for s in steps if isinstance(s, dict) and s.get("command")]
    return None


def _mount_solution(sess: TBContainerSession) -> None:
    """Copy the task's solution files to /solution in the running container (docker cp)."""
    tgt = sess.exec_target
    sess._run(f"exec {shlex.quote(tgt)} mkdir -p /solution", timeout=60)
    sol_dir = os.path.join(sess.task_dir, "solution")
    if os.path.isdir(sol_dir):
        sess._run(f"cp {shlex.quote(sol_dir)}/. {shlex.quote(tgt)}:/solution/", timeout=120)
    sol_sh = os.path.join(sess.task_dir, "solution.sh")
    if os.path.exists(sol_sh):
        sess._run(f"cp {shlex.quote(sol_sh)} {shlex.quote(tgt)}:/solution/solution.sh", timeout=60)


def _scan_one(inst: dict, timeout: int) -> tuple[str, str, float | None, str]:
    """Run one task's reference solution; return (task_id, verdict, score, diag).

    Verdicts: GOLD_OK, GOLD_FAIL, NO_SOLUTION, START_FAIL, ERROR. For GOLD_FAIL and ERROR, diag
    holds the tail of the solution and test output; otherwise it is empty."""
    tid = inst["instance_id"]
    sess = TBContainerSession(tid, name=f"gold_{tid[:40]}", src=inst.get("tb_src", ""))
    steps = _solution_steps(sess.task_dir)
    if steps is None:
        return tid, "NO_SOLUTION", None, ""
    sol_tail = ""
    try:
        if not sess.start():
            return tid, "START_FAIL", None, ""
        _mount_solution(sess)
        for s in steps:
            # run from the task's working directory, as the agent does
            cp = sess._run(f"exec {shlex.quote(sess.exec_target)} bash -lc "
                           f"{shlex.quote('cd ' + sess.workdir + ' && ' + s)}", timeout=timeout)
            sol_tail = ((cp.stdout or "") + "\n" + (cp.stderr or ""))[-1500:]
        phi = sess.score()
        if phi == 1.0:
            return tid, "GOLD_OK", phi, ""
        diag = f"--- solution tail ---\n{sol_tail}\n--- test tail ---\n{sess.last_test_output[-3000:]}"
        return tid, "GOLD_FAIL", phi, diag
    except subprocess.TimeoutExpired:
        return tid, "ERROR", None, f"timeout; solution tail:\n{sol_tail}"
    except Exception as e:
        return tid, "ERROR", None, f"{e!r}; solution tail:\n{sol_tail}"
    finally:
        sess.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True, help="json list of gold-invalid task_ids")
    ap.add_argument("--report", default="", help="full {tid: {verdict, phi}} map (optional)")
    ap.add_argument("--conc", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=1800, help="per-solution-step timeout")
    args = ap.parse_args()

    insts = load_instances(args.parquet, 0, 42)
    t0 = time.time()
    print(f"TB_GOLD_SCAN_START {len(insts)} tasks conc={args.conc}", flush=True)

    results = {}
    with ThreadPoolExecutor(max_workers=args.conc) as ex:
        for tid, verdict, phi, diag in ex.map(lambda i: _scan_one(i, args.timeout), insts):
            results[tid] = {"verdict": verdict, "phi": phi, "diag": diag}
            print(f"TB_GOLD {tid} {verdict} phi={phi}", flush=True)

    invalid = sorted(t for t, r in results.items() if r["verdict"] != "GOLD_OK")
    with open(args.out, "w") as f:
        json.dump(invalid, f, indent=1)
    if args.report:
        with open(args.report, "w") as f:
            json.dump(results, f, indent=1)
    n_ok = len(results) - len(invalid)
    print(f"TB_GOLD_SCAN_DONE {n_ok}/{len(results)} pass in {time.time()-t0:.0f}s "
          f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
