"""verse.evolution — the config-driven harness-evolution engine.

Runs on a CPU machine: the frozen executor is served by vLLM or a hosted API, tasks are
graded in docker containers, and the optimizer is an LLM. No GPU is needed on this machine.

A method is a yaml that picks one component per slot:

    EvidenceSource.gather(ctx) -> (report, [Claim])   what the optimizer sees of the trajectories
    T2TProbeKit (probes.py)    -> ExperimentRecord    the verification, replay and perturbation
                                                      tools (fix_probe, replay, ablate,
                                                      substitute); without them, claims stay
                                                      unverified hypotheses
    EditSpace.apply/revert/assemble                   what may change in the executor harness
    Gate.judge(ctx, prop, before, after) -> GateVerdict   keep or revert
    driver.run_evolution(config)                      the round loop (single lineage)

Claims carry their own evidence (verse.claims), so whether a diagnosis is verified depends on
the data, not on a code branch. Baseline components follow the official baseline repositories.
"""
from verse.evolution.protocols import (
    EvolutionContext, GateVerdict, Proposal, EvidenceSource, Gate,
)
from verse.evolution.registry import build_component, register_component, registered

# importing the component modules registers them
from verse.evolution import gates, probes, edit_spaces, evidence, intervener, \
    probe_scheduler  # noqa: F401

__all__ = [
    "EvolutionContext", "GateVerdict", "Proposal", "EvidenceSource", "Gate",
    "build_component", "register_component", "registered",
]
