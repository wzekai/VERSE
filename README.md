# VERSE: Verified Self-Evolving Optimizer

Code for the paper [**VERSE: Verified Self-Evolving Optimizer for Agent Harnesses**](https://arxiv.org/abs/2610.02616).

## Install

Requires Python >= 3.12 and docker.

```bash
git clone https://github.com/wzekai/VERSE.git
cd VERSE
pip install -e .
python -m pytest -q tests        # unit tests; no GPU or docker needed
```

## Prepare data

Task data are not redistributed. Build the task files in `data/` by following
[`data/README.md`](data/README.md).

## Serve the models

The executor (Qwen3.8-27B) and the optimizer (Qwen3.8-Flash-Next) are served locally with vLLM:

```bash
bash scripts/serve_optimizer.sh                              # 4 GPUs, port 8100
EXEC_GPU=4 EXEC_PORT=8101 bash scripts/serve_executor.sh     # 1 GPU, port 8101
```

To use more executor replicas, start one per GPU and list them in `T2E_VLLM_MODELS`, e.g.
`qwen38-27b=http://127.0.0.1:8101|http://127.0.0.1:8102`.

## Run harness evolution

Each method is one config under [`verse/configs/`](verse/configs/).

```bash
CONFIG=baselines/meta_harness TAG=-r1 bash scripts/run_swe.sh     # baseline (Meta-Harness)
CONFIG=self_teacher/verse_mh  TAG=-r1 bash scripts/run_swe.sh     # Meta-Harness + VERSE
CONFIG=tb/tb_verse_hx         TAG=-r1 bash scripts/run_tb.sh      # Terminal-Bench
```

A run evolves the harness, selects a round on validation, and evaluates that harness on the
test set three times. Results go to `out/<config><TAG>/`; the test result is `test110.json`.

## Evaluate a released harness

[`results/`](results/) holds the harnesses selected in the paper's main experiments and their
per-task test outcomes. To evaluate one again:

```bash
# in-distribution test set
CONFIG=self_teacher/verse_mh TAG=-eval EVAL_WORKSPACE=results/verse_mh/harness \
  bash scripts/run_swe.sh

# out-of-distribution test set
CONFIG=self_teacher/verse_mh TAG=-eval-ood EVAL_WORKSPACE=results/verse_mh/harness \
  SWE_TEST=swe_test_0726.parquet GOLD_INVALID=gold_invalid_test_0726_final.json \
  T2E_SWE_JUDGE_ALLOW_NET=1 T2E_SWE_RUN_TIMEOUT=3000 bash scripts/run_swe.sh
```

`python results/make_table.py` prints the main results table from these files.

## Repository layout

```
verse/      the engine: evolution loop, baselines, VERSE, evaluation (configs in verse/configs/)
data/       task IDs of every split and data-building instructions
results/    selected harnesses and per-task test outcomes of the main experiments
scripts/    run_swe.sh, run_tb.sh, model serving, data preparation
tests/      unit tests
```

## License

This code is released under the [MIT License](LICENSE).

## Citation

```bibtex
@article{wang2026verse,
  title={VERSE: Verified Self-Evolving Optimizer for Agent Harnesses},
  author={Wang, Zekai and Ge, Yingqiang and Wang, Zekun and Wang, Hai and Xu, Yuhui and Frandsen, Joshua and Fu, Shancong and Wilson, Ashia C and Reddy, Chandan K},
  journal={arXiv preprint arXiv:2610.02616},
  year={2026}
}
```
