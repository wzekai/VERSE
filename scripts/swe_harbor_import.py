#!/usr/bin/env python3
"""Convert a Harbor-packaged SWE-rebench dataset into a pool parquet and a manifest json.

The dataset has one directory per task (tests/config.json, tests/test.sh,
environment/Dockerfile, solution/solve.sh). Rows use the engine's schema
(prepare_data.py::_pool_row). Each row's extra_info is the task's tests/config.json (the
SWE-bench-style record: instance_id, repo, base_commit, patch, test_patch, problem_statement,
FAIL_TO_PASS, PASS_TO_PASS, install_config, language, image_name, created_at) plus:
  gold_patch            = patch (the field name the gold scan reads)
  docker_image          = <image_prefix>/<instance_id>:<tag>, the locally built task image
                          (environment/Dockerfile)
  repo_dir              = /<repo-name>, the SWE-rebench-V2 checkout location
                          (checked by build_swe_ood_images.sh)
  verifier_dir          = <box_root>/<task>/tests, the Harbor verifier (test.sh and parsers)
                          that the judge runs
  verifier_timeout_sec  = [verifier].timeout_sec from task.toml
Usage: swe_harbor_import.py <dataset_dir> <out.parquet> <manifest.json> --box-root PATH
       [--image-prefix sweb07]
"""
import argparse, json, os, re, sys, datetime as dt
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir"); ap.add_argument("out_parquet"); ap.add_argument("manifest")
    ap.add_argument("--box-root", required=True,
                    help="absolute path of the dataset dir on the box that runs the evals")
    ap.add_argument("--image-prefix", default="sweb07")
    a = ap.parse_args()
    rows, man = [], []
    for name in sorted(os.listdir(a.dataset_dir)):
        d = os.path.join(a.dataset_dir, name); cfg = os.path.join(d, "tests", "config.json")
        if not os.path.isfile(cfg): continue
        t = json.load(open(cfg))
        for k in ("FAIL_TO_PASS", "PASS_TO_PASS"):  # keep lists (judge takes lists or JSON)
            if isinstance(t.get(k), str):
                try: t[k] = json.loads(t[k])
                except Exception: pass
        ca = t.get("created_at")
        if isinstance(ca, (int, str)) and str(ca).isdigit():  # epoch ms to ISO, as in V1 rows
            t["created_at"] = dt.datetime.fromtimestamp(int(ca) / 1000, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        toml = open(os.path.join(d, "task.toml")).read()
        m = re.search(r"\[verifier\][^\[]*?timeout_sec\s*=\s*([0-9.]+)", toml, re.S)
        t["verifier_timeout_sec"] = float(m.group(1)) if m else 3000.0
        t["gold_patch"] = t.get("patch", "")
        t["hub_image_name"] = t.get("image_name", "")
        tag = (t.get("image_name", "").rsplit(":", 1)[1] if ":" in t.get("image_name", "") else "latest")
        t["docker_image"] = f"{a.image_prefix}/{t['instance_id'].lower()}:{tag}"
        t["repo_dir"] = "/" + t["repo"].split("/")[1]
        t["verifier_dir"] = os.path.join(a.box_root, name, "tests")
        t["task_dir_name"] = name
        rows.append({"data_source": "swe", "agent_name": "swe_agent", "ability": "swe",
                     "prompt": [{"role": "user", "content": str(t.get("problem_statement", ""))[:20000]}],
                     "reward_model": {"ground_truth": t.get("instance_id", ""), "style": "rule"},
                     "extra_info": t})
        man.append({"instance_id": t["instance_id"], "repo": t["repo"], "language": t.get("language"),
                    "hub_image": t["hub_image_name"], "docker_image": t["docker_image"], "repo_dir": t["repo_dir"],
                    "created_at": t["created_at"], "task_dir": name})
    keys = sorted({k for r in rows for k in r["extra_info"]})
    for r in rows:
        ei = r["extra_info"]
        r["extra_info"] = {k: (ei[k] if isinstance(ei.get(k), str) else json.dumps(ei[k]) if k in ei and ei[k] is not None else "")
                           for k in keys}
    pd.DataFrame(rows).to_parquet(a.out_parquet)
    json.dump({"n": len(man), "tasks": man}, open(a.manifest, "w"), indent=1)
    langs = {}
    for r in man: langs[r["language"]] = langs.get(r["language"], 0) + 1
    print(f"wrote {a.out_parquet}: {len(rows)} rows; languages {langs}; created {min(r['created_at'] for r in man)} .. {max(r['created_at'] for r in man)}")

if __name__ == "__main__":
    sys.exit(main())
