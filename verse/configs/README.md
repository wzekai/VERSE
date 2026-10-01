# Configs

One yaml per method. Each is merged over the shared settings in `_base.yaml`; the launchers
only set the models, number of rounds, concurrency and data files.

| Config | Method |
|---|---|
| `baselines/{meta_harness,ahe,self_harness,harnessx}` | the four baselines |
| `self_teacher/verse_{mh,ahe,sh,hx}` | baseline + VERSE |
| `ours/verified_{mh,ahe,sh,hx}` | baseline + VERSE without optimizer self-evolution |
| `self_teacher/reflect_only` | self-evolving optimizer without verification |
| `self_teacher/bare_static` | simple verification only |
| `self_teacher/self_teacher` | self-evolving optimizer with simple verification |
| `ours/verified_ahe_{sysonly,noattr,noledger}` | ablations: remove the verification tools / attribution and minimization / the training audit |
| `ours/verified_ahe_{randprobe,fixedsched}` | ablations: random or fixed-rule scheduling of the verification tools |
| `self_teacher/self_teacher_{promptonly,notesonly,toolsonly}` | self-evolution restricted to one part of the optimizer harness |
| `tb/*` | the same methods on Terminal-Bench |

## Shared settings (`_base.yaml`)

- 6 rounds on SWE-rebench (3 on Terminal-Bench), 3 candidate edits per round. The two best
  rounds on validation are re-evaluated, and the round with the higher mean is selected.
- Every method edits the same harness parts: the markdown prompt files and
  `harness_code/hooks.py`.
- The optimizer sees training trajectories and, from validation, only a summary (accuracy per
  round, IDs of tasks whose outcome changed, harness-code error rate). Test tasks are never seen
  during evolution; the test set is evaluated at the end (three repeats).
- Verification runs are budgeted at 8 per optimizer episode (`probes.budget`); the 3 candidates
  of a round share one pool of 24.

## Where the VERSE components are

| Component | Config key | Code |
|---|---|---|
| verification tools (`fix_probe`, `replay`, `ablate`/`substitute`) | `probes` | `verse/evolution/probes.py` |
| attribution with trace minimization, training audit | `attribute` | `verse/evolution/attribute.py` |
| optimizer self-evolution | `meta_teacher` | `verse/evolution/meta_teacher.py` |

In the code, the optimizer is called *teacher* or *intervener*, the executor *student*, and the
training audit the failure-mode *ledger*. `fix_probe` is the verification tool, and `ablate`
and `substitute` form the perturbation tool.
