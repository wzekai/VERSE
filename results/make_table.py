"""Recompute the validation-selected columns of the main results table from the files in results/.

For each setting: accuracy of repeat i = 100 * (tasks solved in repeat i) / n_tasks; the table
reports the mean over the three repeats and its standard error (sample std / sqrt(3)).
Usage: python results/make_table.py
"""
import json
import math
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["blank", "meta_harness", "verse_mh", "ahe", "verse_ahe",
         "self_harness", "verse_sh", "harnessx", "verse_hx"]


def cell(path):
    d = json.load(open(path))
    n, oc = d["n_tasks"], d["outcomes"]
    assert len(oc) == n
    solved = [sum(v[i] for v in oc.values()) for i in range(d["repeats"])]
    assert solved == d["solved_per_repeat"]
    acc = [100.0 * s / n for s in solved]
    mean, sem = sum(acc) / len(acc), statistics.stdev(acc) / math.sqrt(len(acc))
    assert (round(mean, 2), round(sem, 2)) == (d["accuracy_mean"], d["accuracy_sem"])
    return f"{mean:5.2f} ± {sem:.2f}"


print(f"{'Method':40s} {'r*':>3s}   {'In-distribution':>15s}   {'Out-of-distribution':>19s}")
for name in ORDER:
    run = json.load(open(os.path.join(HERE, name, "run.json")))
    ind = cell(os.path.join(HERE, name, "test_in_distribution.json"))
    ood = cell(os.path.join(HERE, name, "test_ood.json"))
    rs = run.get("selected_round")
    rs = "--" if rs is None else str(rs)
    print(f"{run['setting']:40s} {rs:>3s}   {ind:>15s}   {ood:>19s}")
