"""Evidence data model for harness diagnosis: claims and the experiments behind them.

A root-cause claim either carries records of executed experiments or it does not. Whether a
diagnosis is verified is therefore a property of the data, not a branch in the code:

    Claim(statement=..., evidence=[])                     -> rendered "[HYPOTHESIS] ..."
    Claim(statement=..., evidence=[supporting record])    -> rendered "[VERIFIED] ..."

Every evidence interface (AHE layered digest, Self-Harness clustering, HarnessX digester,
Meta-Harness free exploration) produces lists of claims, and one renderer serves them all. A
configuration without the verification tools only produces claims with empty evidence, so
adding the tools is a single switch on top of any baseline.

Pure data, rendering and JSON dump/load; no model calls.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict


@dataclass
class ExperimentRecord:
    """One executed experiment behind a claim: an execution result, not a model opinion."""
    kind: str                 # "replay" | "ablate" | "substitute" | "fix_probe" | "minimize"
    setup: str                # e.g. "re-ran all 17 recorded commands without step 14"
    outcome: str              # e.g. "replay phi flipped 0 -> 1"
    verdict: str              # "supports" | "refutes" | "inconclusive"
    task_id: str = ""
    detail: dict = field(default_factory=dict)   # raw numbers (phi values, rates, step ids)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ExperimentRecord":
        known = {k: d[k] for k in ("kind", "setup", "outcome", "verdict", "task_id", "detail")
                 if k in d}
        return ExperimentRecord(**known)


@dataclass
class Claim:
    """One root-cause assertion about a failure. Verified iff it carries supporting evidence."""
    statement: str
    task_id: str = ""
    evidence: list = field(default_factory=list)      # list[ExperimentRecord]
    tags: dict = field(default_factory=dict)          # source-specific labels (cluster sig,
                                                      # criticality, step ids, ...)

    @property
    def status(self) -> str:
        """Status of the claim, derived from its evidence.

        REFUTED:    at least one refuting experiment (the diagnosis was tested and failed).
        VERIFIED:   at least one supporting experiment and none refuting.
        HYPOTHESIS: no conclusive experiment (baselines without the tools only produce these).
        """
        if any(e.verdict == "refutes" for e in self.evidence):
            return "REFUTED"
        if any(e.verdict == "supports" for e in self.evidence):
            return "VERIFIED"
        return "HYPOTHESIS"

    def to_dict(self) -> dict:
        return {"statement": self.statement, "task_id": self.task_id,
                "evidence": [e.to_dict() for e in self.evidence], "tags": self.tags}

    @staticmethod
    def from_dict(d: dict) -> "Claim":
        return Claim(statement=d.get("statement", ""), task_id=d.get("task_id", ""),
                     evidence=[ExperimentRecord.from_dict(e) for e in d.get("evidence", [])],
                     tags=dict(d.get("tags") or {}))


def render_claim(c: Claim) -> str:
    """Render one claim as a report line followed by its indented experiment records.

    The status prefix tells the optimizer which statements were tested and which are conjecture.
    """
    head = f"[{c.status}] {c.statement}"
    if c.task_id:
        head += f"  (task: {c.task_id})"
    lines = [head]
    for e in c.evidence:
        lines.append(f"    experiment[{e.kind}/{e.verdict}]: {e.setup} -> {e.outcome}")
    return "\n".join(lines)


def render_claims(claims: list, title: str = "") -> str:
    parts = [f"## {title}"] if title else []
    n_v = sum(1 for c in claims if c.status == "VERIFIED")
    n_r = sum(1 for c in claims if c.status == "REFUTED")
    n_h = len(claims) - n_v - n_r
    if claims:
        parts.append(f"({len(claims)} claims: {n_v} VERIFIED / {n_h} HYPOTHESIS / {n_r} REFUTED)")
    parts.extend(render_claim(c) for c in claims)
    return "\n".join(parts)


def dump_claims(claims: list, path: str) -> None:
    with open(path, "w") as f:
        json.dump([c.to_dict() for c in claims], f, indent=1)


def load_claims(path: str) -> list:
    with open(path) as f:
        return [Claim.from_dict(d) for d in json.load(f)]
