"""Evaluate a harness on the held-out test set, separately from training.

Three ways to pick the harness under test:

  1. The run's val-picked round (default): read the run's journal (rounds.jsonl), check out that
     round's workspace commit, and sweep the test parquet.
  2. A specific round: --round N (-1 = last completed round) from the same run directory,
     regardless of what val picked.
  3. A standalone harness, no training run needed:
       --blank                          the seed harness (base prompt only)
       --workspace /path/to/workspace   any harness directory (e.g. results/verse_mh/harness
                                        or a hand-written harness), swept as-is.

Results are written to <arm-dir>/test110*.json (suffix _blank, _ws or _r<N> by mode):
  {"arm", "best_round", "best_val", "sha", "solved_per_rep", "solved_mean", "solved_sem",
   "task_solve_rate", "nan_tasks_per_rep", "judged_all_reps", "outcomes", ...}
Trajectories go under <arm-dir>/traj_test*.

Usage:
  python -m verse.evolution.eval --arm-dir out/verified_mh \\
      --test-parquet data/swe_test.parquet --repeats 3 --conc 24
  python -m verse.evolution.eval --arm-dir out/anything --blank ...
  python -m verse.evolution.eval --arm-dir out/anything --round -1 ...
  python -m verse.evolution.eval --arm-dir out/newdir \\
      --workspace results/verse_mh/harness --test-parquet ... --model qwen38-27b
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

from verse.evolution.driver import load_instances, sweep


def _final_record(journal: str) -> dict | None:
    if not os.path.exists(journal):
        return None
    rec = None
    for ln in open(journal):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if r.get("round") == "final":
            rec = r
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm-dir", required=True,
                    help="the arm's output dir (results and trajectories land here)")
    ap.add_argument("--test-parquet", required=True)
    ap.add_argument("--wait", action="store_true", help="poll until the final record exists")
    ap.add_argument("--blank", action="store_true",
                    help="evaluate the SEED harness (base prompt only) — no training run "
                         "or journal needed")
    ap.add_argument("--workspace", default="",
                    help="evaluate an arbitrary harness directory as-is — no training run "
                         "or journal needed")
    ap.add_argument("--round", type=int, default=None, dest="round_n",
                    help="evaluate round N's harness state (journal ws_head) instead of the "
                         "frontier pick; -1 = the last completed round")
    ap.add_argument("--base-prompt", default="",
                    help="with --blank: 'TB' for the Terminal-Bench seed, else the SWE seed")
    ap.add_argument("--conc", type=int, default=12)
    ap.add_argument("--repeats", type=int, default=1,
                    help="run the test sweep N times (fresh rollouts each) and report "
                         "per-rep solved + mean±SEM; single-run SE on 110 tasks is ~4.8, "
                         "so 3 reps is the floor for resolving small deltas")
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--regions", default="us-west-2")
    ap.add_argument("--invalid-tasks", default="",
                    help="json list of task_ids whose GOLD patch fails on this substrate "
                         "(gold-scan output); dropped from the test set as unjudgeable")
    args = ap.parse_args()

    arm = os.path.basename(os.path.normpath(args.arm_dir))
    journal = os.path.join(args.arm_dir, "rounds.jsonl")
    ws = os.path.join(args.arm_dir, "workspace")
    os.makedirs(args.arm_dir, exist_ok=True)

    from verse.workspace import HarnessWorkspace7
    from verse.evolution.hooks_runtime import load_hooks_source

    wt = None                                   # detached worktree (journal modes only)
    if args.blank:
        # standalone: create the seed harness in the run directory
        bp = args.base_prompt
        if bp == "TB":
            from verse.runtime.tb_env import TB_BASE_PROMPT as bp
        elif not bp:
            from verse.runtime.executor import BASE_PROMPT as bp
        blank_dir = os.path.join(args.arm_dir, "blank_ws")
        if not os.path.isdir(os.path.join(blank_dir, ".git")):
            HarnessWorkspace7(blank_dir).init_blank(bp)
        harness_dir, sha, best_round, best_val = blank_dir, "blank", 0, 0
    elif args.workspace:
        # standalone: sweep any harness directory as-is
        harness_dir = args.workspace
        sha, best_round, best_val = "external", -1, 0
    elif args.round_n is not None:
        recs = [json.loads(ln) for ln in open(journal)]
        rounds = [r for r in recs if isinstance(r.get("round"), int) and r.get("ws_head")]
        if not rounds:
            raise SystemExit(f"{arm}: no completed rounds in journal")
        pick = rounds[-1] if args.round_n < 0 else next(
            (r for r in rounds if r["round"] == args.round_n), None)
        if pick is None:
            raise SystemExit(f"{arm}: round {args.round_n} not in journal "
                             f"(have {[r['round'] for r in rounds]})")
        # if --round -1 is the round the final record already picked and test110.json
        # exists, the two evaluations would be identical: copy the result instead
        if args.round_n < 0:
            fin = _final_record(journal)
            main_out = os.path.join(args.arm_dir, "test110.json")
            if fin and fin.get("best_round") == pick["round"] and os.path.exists(main_out):
                import shutil as _sh
                dup = os.path.join(args.arm_dir, f"test110_r{pick['round']}.json")
                _sh.copyfile(main_out, dup)
                print(f"[eval:{arm}] last round {pick['round']} == val-picked round; "
                      f"copied existing result -> {dup} (no rerun)", flush=True)
                return
        va = pick.get("val_score_after", "")
        best_val = int(str(va).split("/")[0]) if va else 0
        sha, best_round = pick["ws_head"], pick["round"]
        harness_dir = None
    else:
        rec = _final_record(journal)
        while rec is None and args.wait:
            print(f"[eval:{arm}] waiting for final record...", flush=True)
            time.sleep(300)
            rec = _final_record(journal)
        if rec is None:
            raise SystemExit(f"{arm}: no final record yet (use --wait to poll)")
        sha, best_round, best_val = rec["best_sha"], rec["best_round"], rec["best_val"]
        harness_dir = None

    if harness_dir is None:
        # journal modes: check out the picked commit in a detached worktree, leaving the
        # run's workspace untouched. A run directory copied from another location may carry
        # a stale test_ws/ and .git/worktrees entries with the old absolute path; remove both.
        import shutil
        wt = os.path.join(args.arm_dir, "test_ws")
        subprocess.run(["git", "-C", ws, "worktree", "remove", "--force", wt],
                       capture_output=True)
        shutil.rmtree(wt, ignore_errors=True)
        subprocess.run(["git", "-C", ws, "worktree", "prune"], capture_output=True)
        subprocess.run(["git", "-C", ws, "worktree", "add", "--force", wt, sha],
                       check=True, capture_output=True)
        harness_dir = wt

    sp = HarnessWorkspace7(harness_dir).assemble()
    hooks_src = load_hooks_source(harness_dir)   # code_hooks runs: load harness_code/hooks.py
    if hooks_src:
        print(f"[eval:{arm}] harness_code/hooks.py present "
              f"({len(hooks_src)} chars) — mounting", flush=True)

    insts = load_instances(args.test_parquet, 0, 42)      # n=0 -> full parquet
    # Drop tasks whose gold patch fails on this backend (from the gold scan): a task whose
    # reference solution scores 0 cannot be judged here. SWE-rebench applies the same filter
    # when it collects tasks.
    invalid = set()
    if args.invalid_tasks and os.path.exists(args.invalid_tasks):
        invalid = set(json.load(open(args.invalid_tasks)))
        insts = [i for i in insts if i.get("instance_id") not in invalid]
        print(f"[eval:{arm}] substrate gate: dropped {len(invalid)} gold-invalid tasks "
              f"-> {len(insts)} judgeable", flush=True)
    print(f"[eval:{arm}] sweeping test {len(insts)} @ {str(sha)[:8]} "
          f"(best_round={best_round} best_val={best_val} repeats={args.repeats})", flush=True)
    executor = {"model": args.model, "regions": args.regions, "max_turns": 100}
    all_tids = {i.get("instance_id", "?") for i in insts}
    reps = []
    # one trajectory directory per mode, so evaluations do not overwrite each other
    traj_prefix = ("traj_test" if args.round_n is None and not args.workspace
                   else f"traj_test_r{best_round}" if args.round_n is not None
                   else "traj_test_ws")
    for rep in range(args.repeats):
        rep_tag = f"_rep{rep}" if args.repeats > 1 else ""
        out, _ = sweep(insts, sp, executor, concurrency=args.conc,
                       traj_dir=os.path.join(args.arm_dir, f"{traj_prefix}{rep_tag}"),
                       label=f"test:{arm}{rep_tag}", nan_retries=5, hooks_src=hooks_src)
        # A NaN task (infra failure) is missing from `outcomes` but still counts in the
        # denominator, so it scores 0. Record the NaN tasks so that comparisons can be
        # restricted to the tasks judged in every run.
        nan_tids = sorted(all_tids - set(out))
        solved = sum(1 for v in out.values() if v >= 0.5)
        reps.append({"solved": solved, "total": len(insts), "outcomes": out,
                     "nan_tasks": nan_tids})
        print(f"[eval:{arm}] rep {rep + 1}/{args.repeats}: {solved}/{len(insts)} "
              f"= {100 * solved / len(insts):.2f}% (NaN: {len(nan_tids)})", flush=True)
        # optional sync between repeats (T2E_TEST_SYNC=src:dst), not during episodes
        ts = os.environ.get("T2E_TEST_SYNC")
        if ts and ":" in ts:
            src, dst = ts.split(":", 1)
            os.system(f"aws s3 sync {src} {dst} --quiet")
    solves = [r["solved"] for r in reps]
    mean = sum(solves) / len(solves)
    sem = ((sum((s - mean) ** 2 for s in solves) / (len(solves) - 1)) / len(solves)) ** 0.5 \
        if len(solves) > 1 else 0.0
    # per-task solve rate across reps: the paired-comparison unit for McNemar-style tests
    task_rate = {tid: sum(1 for r in reps if r["outcomes"].get(tid, 0) >= 0.5) / len(reps)
                 for tid in reps[0]["outcomes"]}
    res = {"arm": arm + ("__blank" if args.blank else ""), "best_round": best_round,
           "best_val": best_val, "sha": sha,
           "test_solved": solves[0], "test_total": len(insts),   # first repeat
           "repeats": len(reps), "solved_per_rep": solves,
           "solved_mean": round(mean, 2), "solved_sem": round(sem, 2),
           # accuracy in percent, the unit reported in the paper
           "accuracy_per_rep": [round(100 * s / len(insts), 2) for s in solves],
           "accuracy_mean": round(100 * mean / len(insts), 2),
           "accuracy_sem": round(100 * sem / len(insts), 2),
           "task_solve_rate": task_rate,
           "n_gold_invalid_dropped": len(invalid),
           "n_nan_per_rep": [len(r["nan_tasks"]) for r in reps],
           "nan_tasks_per_rep": [r["nan_tasks"] for r in reps],
           "judged_all_reps": sorted(set.intersection(*[set(r["outcomes"]) for r in reps])),
           "hooks_mounted": bool(hooks_src),
           "outcomes": reps[0]["outcomes"], "ts": time.time()}
    suffix = ("_blank.json" if args.blank else
              "_ws.json" if args.workspace else
              (f"_r{best_round}.json" if args.round_n is not None else ".json"))
    path = os.path.join(args.arm_dir, "test110" + suffix)
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[eval:{arm}] TEST RESULT: {'/'.join(map(str, solves))} of {len(insts)} "
          f"mean={mean:.1f}±{sem:.1f} "
          f"accuracy={100 * mean / len(insts):.2f}%±{100 * sem / len(insts):.2f} "
          f"-> {path}", flush=True)
    if wt:
        subprocess.run(["git", "-C", ws, "worktree", "remove", "--force", wt],
                       capture_output=True)


if __name__ == "__main__":
    main()
