"""Resume tests for the evolution driver (no LLM, no docker).

A resumed driver must take its baseline outcomes and failure transcripts from the newest
training sweep, not from round 0. Otherwise every kept fix counts again as a new fix and the
optimizer reads stale round-0 failure transcripts.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from verse.evolution.driver import _newest_kept_traj_dir, _reload_train_out, _reload_fails


def _mk(out, rnd, recs):
    d = os.path.join(out, f"round_{rnd:02d}", "traj_train")
    os.makedirs(d, exist_ok=True)
    for tid, phi in recs.items():
        with open(os.path.join(d, f"{tid}.json"), "w") as f:
            json.dump({"instance_id": tid, "phi": phi,
                       "transcript": [{"role": "user", "content": f"t-{tid}-r{rnd}"}]}, f)
    return d


def test_newest_kept_traj_dir_covers_unkept_round_after_keep():
    # After a keep at round 1 the next training sweep runs in round 2's directory, even
    # though round 2 is not kept. A resume after round 2 must pick round_02 (the sweep of
    # the round-1 harness).
    with tempfile.TemporaryDirectory() as out:
        _mk(out, 0, {"a": 0.0})
        _mk(out, 2, {"a": 1.0})
        history = [{"round": 1, "kept": True}, {"round": 2, "kept": False}]
        assert _newest_kept_traj_dir(out, history).endswith("round_02/traj_train")


def test_reload_train_out_prefers_newest_sweep():
    with tempfile.TemporaryDirectory() as out:
        _mk(out, 0, {"a": 0.0, "b": 0.0})
        _mk(out, 2, {"a": 1.0, "b": 0.0})
        history = [{"round": 1, "kept": True}, {"round": 2, "kept": False}]
        assert _reload_train_out(out, history) == {"a": 1.0, "b": 0.0}


def test_reload_fails_reads_matching_transcripts():
    with tempfile.TemporaryDirectory() as out:
        _mk(out, 0, {"a": 0.0, "b": 0.0})
        _mk(out, 2, {"a": 1.0, "b": 0.0})
        history = [{"round": 1, "kept": True}, {"round": 2, "kept": False}]
        train_out = _reload_train_out(out, history)
        fails = _reload_fails(out, history, train_out)
        assert set(fails) == {"b"}
        assert fails["b"][0]["content"] == "t-b-r2"     # from the newest sweep, not round 0


def test_val_replay_reconstructs_kept_flips():
    # mirrors the resume block of run_evolution: apply kept verdicts to the round-0 val outcomes
    val = {"x": 0.0, "y": 1.0, "z": 0.0}
    history = [
        {"round": 1, "kept": True, "verdict": {"fixed": ["x"], "regressed": ["y"]}},
        {"round": 2, "kept": False, "verdict": {"fixed": ["z"], "regressed": []}},  # reverted
    ]
    for h in history:
        if not h.get("kept"):
            continue
        v = h.get("verdict") or {}
        for t in v.get("fixed", []):
            val[t] = 1.0
        for t in v.get("regressed", []):
            val[t] = 0.0
    assert val == {"x": 1.0, "y": 0.0, "z": 0.0}


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print(f"{name}: OK")
