# Data

Task data are not redistributed. This folder holds the task IDs of every split; the task files
(`.parquet`) are built from the public sources with the steps below and placed here.

| Task file | Tasks | Role | Source |
|---|---|---|---|
| `swe_train_v6.parquet` | 110 | train | SWE-rebench monthly leaderboard pool |
| `swe_val_v6.parquet` | 50 | validation | SWE-rebench monthly leaderboard pool |
| `swe_test.parquet` | 110 (108 scored) | in-distribution test | SWE-rebench, March 2026 release |
| `swe_test_0726.parquet` | 111 (107 scored) | out-of-distribution test | SWE-rebench, July 2026 release |
| `tb_train_v3.parquet` | 104 | train | Terminal-Bench 1.x |
| `tb_val_v4.parquet` | 98 (50 sampled per run) | validation | Terminal-Bench 1.x + Terminal-Bench-Lite |
| `tb_test.parquet` | 89 | test | Terminal-Bench 2.x |

Files in this folder:

- `splits/*.json`: the task IDs of each split above (except the July 2026 test set).
- `swe_test_0726_manifest.json`: the 111 tasks of the July 2026 test set.
- `gold_invalid_test.json`, `gold_invalid_test_0726_final.json`: tasks whose own reference
  solution fails in the grading environment. They are not scored, which gives 108 and 107
  scored tasks.

## SWE-rebench and Terminal-Bench splits

Export the public task records to a parquet or jsonl file with one task per row, then build and
check each split against its ID list:

```bash
python scripts/prepare_data.py build --domain swe \
    --source swe_rebench_dump.parquet \
    --manifest data/splits/swe_train_v6.json --out data/swe_train_v6.parquet

python scripts/prepare_data.py verify \
    --manifest data/splits/swe_train_v6.json --parquet data/swe_train_v6.parquet
```

Repeat for every file in `splits/` (use `--domain tb` for Terminal-Bench). `verify` fails if
the task IDs differ from the list.

Terminal-Bench runs also need the task directories of the three suites under `$TBROOT`
(default `tb_tasks/`): `original-tasks/` (Terminal-Bench 1.x), `terminal-bench-2/` and
`OpenThoughts-TBLite/`.

## SWE-rebench July 2026 test set (out-of-distribution)

111 tasks in Go, Java, Python, Rust and TypeScript, all newer than the other splits. Each task is
graded by its own verifier script, which may download dependencies, so evaluation sets
`T2E_SWE_JUDGE_ALLOW_NET=1` (network for the grading container only; the agent never has
network).

```bash
# 1. download the tasks (Harbor CLI)
harbor download ibragim-badertdinov/swe-rebench-07-2026@2026-07 -o data/swe_rebench_07_2026
DS=$(pwd)/data/swe_rebench_07_2026/swe-rebench-07-2026

# 2. convert to a task file
python scripts/swe_harbor_import.py $DS data/swe_test_0726.parquet /tmp/manifest.json --box-root $DS

# 3. build the task images and add Python where it is missing
DS=$DS bash scripts/build_swe_ood_images.sh data/swe_test_0726_manifest.json
bash scripts/add_python_to_swe_ood_images.sh

# 4. (optional) check reference solutions; expected failures = gold_invalid_test_0726_final.json
T2E_SWE_ECR_PREFIX= T2E_SWE_JUDGE_ALLOW_NET=1 T2E_SWE_CONTAINER_CPUS=2 \
T2E_SWE_DOCKER_RUN_ARGS="--sysctl net.ipv6.conf.all.disable_ipv6=0" \
  python -m verse.evolution.gold_scan --parquet data/swe_test_0726.parquet \
  --out /tmp/gold_invalid_ood.json --conc 8 --timeout 3000
```
