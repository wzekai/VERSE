"""registry.py — component registration and assembly from yaml.

Each component slot in a method config is a dict with a `kind`, e.g. from
ours/verified_ahe.yaml (gate comes from _base.yaml):

    evidence:   {kind: push_digest_layered, max_tasks: 12}
    probes:     {kind: t2t_probes, budget: 8, budget_s: 3600}   # omitted by the baselines
    edit_space: {kind: code_hooks}
    gate:       {kind: keep_all}

build_component(slot, cfg) instantiates the class registered for (slot, kind), passing the
rest of the cfg dict as kwargs. An unknown kind or kwarg raises at assembly time, so a typo
in a yaml cannot silently run the wrong method."""
from __future__ import annotations

_REGISTRY: dict = {}      # (slot, kind) -> class


def register_component(slot: str, kind: str):
    def deco(cls):
        key = (slot, kind)
        if key in _REGISTRY and _REGISTRY[key] is not cls:
            raise ValueError(f"duplicate component registration: {key}")
        _REGISTRY[key] = cls
        return cls
    return deco


def build_component(slot: str, cfg: dict | None):
    if not cfg:
        return None
    cfg = dict(cfg)
    kind = cfg.pop("kind", None)
    if not kind:
        raise ValueError(f"component config for slot {slot!r} missing 'kind': {cfg}")
    cls = _REGISTRY.get((slot, kind))
    if cls is None:
        known = sorted(k for s, k in _REGISTRY if s == slot)
        raise ValueError(f"unknown {slot} component {kind!r}; registered: {known}")
    return cls(**cfg)


def registered(slot: str | None = None) -> list:
    if slot is None:
        return sorted(_REGISTRY)
    return sorted(k for s, k in _REGISTRY if s == slot)
