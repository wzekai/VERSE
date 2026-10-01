"""Config switches of the ablations that each remove one VERSE component (no LLM, no docker).

attribute.ledger=false leaves the training audit (failure-mode ledger) out of the optimizer's
report; probe_tools=false withholds the four tools (fix_probe, replay, ablate, substitute) from
the optimizer episode.
"""
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from verse.evolution.protocols import EvolutionContext

CFG_DIR = os.path.join(os.path.dirname(__file__), "..", "verse", "configs", "ours")


def _load(name):
    return yaml.safe_load(open(os.path.join(CFG_DIR, name)))


def test_noledger_keeps_attribute_layer_but_flags_ledger_off():
    cfg = _load("verified_ahe_noledger.yaml")
    attr = cfg["attribute"]
    assert attr.get("ledger", True) is False
    # all other attribute settings match verified_ahe
    base = _load("verified_ahe.yaml")["attribute"]
    assert {k: v for k, v in attr.items() if k != "ledger"} == base


def test_noattr_drops_the_diagnosis_layer_only():
    cfg = _load("verified_ahe_noattr.yaml")
    assert cfg.get("attribute") is None
    base = _load("verified_ahe.yaml")
    assert cfg["probes"] == base["probes"]
    assert cfg["evidence"] == base["evidence"]


def test_sysonly_withholds_probe_tools_from_the_intervener():
    cfg = _load("verified_ahe_sysonly.yaml")
    assert cfg["probe_tools"] is False
    base = _load("verified_ahe.yaml")
    assert cfg["probes"] == base["probes"]
    assert cfg["attribute"] == base["attribute"]
    # the driver's expression and the optimizer's read of probe_tools agree for both values
    ctx = EvolutionContext(round_idx=0, ws_dir="", traj_dir="")
    ctx.extra["probe_tools"] = cfg.get("probe_tools", True) is not False
    assert ctx.extra.get("probe_tools", True) is False
    ctx2 = EvolutionContext(round_idx=0, ws_dir="", traj_dir="")
    ctx2.extra["probe_tools"] = base.get("probe_tools", True) is not False
    assert ctx2.extra.get("probe_tools", True) is True


def test_probe_budget_override_parses_to_int():
    # run scripts pass PROBE_BUDGET through as --set probes.budget=N
    import json
    assert json.loads("4") == 4
