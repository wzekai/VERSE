"""The executor harness: a git-backed directory of component files.

The executor's system prompt is assembled deterministically from these files. Model weights stay
frozen; only the files evolve, with one git commit per round, so git history is the audit trail
and a revert is a git operation.

Layout (the seven components of AHE's NexAU decomposition, plus executable hooks):
    system_prompt.md   tool_notes/*.md   skills/*.md   middleware/*.md
    subagents/*.md     memory.md         workflow.md
    harness_code/hooks.py   executable (code_hooks edit space). Never added to the prompt;
                            hooks_runtime loads it into the executor loop.

`HarnessWorkspace` is the minimal 4-component base that holds the git and Manifest logic;
`HarnessWorkspace7` is the class the experiments use.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field, asdict

COMPONENTS = ("system_prompt.md", "skills", "memory.md", "tool_notes.md")
_COMPONENT_LAYER = {"system_prompt.md": "O", "skills": "L", "memory.md": "C", "tool_notes.md": "T"}


def _git(ws: str, *args: str) -> str:
    r = subprocess.run(["git", "-C", ws, *args], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {ws}: {r.stderr.strip()[:300]}")
    return r.stdout


@dataclass
class Manifest:
    """One proposed harness edit in AHE's predict-then-verify form.

    The optimizer states its predictions before the outcome is measured.
    """
    component: str                     # one of COMPONENTS (for "skills", path is skills/<file>)
    path: str                          # workspace-relative file the edit rewrites
    edit: str                          # full new file content (no diffs to misapply)
    rationale: str = ""
    predicted_fixes: list = field(default_factory=list)    # task_ids expected to flip 0->1
    at_risk: list = field(default_factory=list)            # task_ids that might regress 1->0
    layer: str = ""                    # layer label; defaults from _COMPONENT_LAYER if empty

    def __post_init__(self):
        if self.component not in COMPONENTS:
            raise ValueError(f"unknown component {self.component!r} (want one of {COMPONENTS})")
        if not self.layer:
            self.layer = _COMPONENT_LAYER[self.component]
        rel = os.path.normpath(self.path)
        if rel.startswith("..") or os.path.isabs(rel):
            raise ValueError(f"path escapes workspace: {self.path!r}")
        top = rel.split(os.sep)[0]
        if top != (self.component if self.component == "skills" else rel) and top != self.component:
            raise ValueError(f"path {self.path!r} does not live under component {self.component!r}")

    @staticmethod
    def from_json(obj) -> "Manifest":
        if isinstance(obj, str):
            obj = json.loads(obj)
        known = {k: obj[k] for k in
                 ("component", "path", "edit", "rationale", "predicted_fixes", "at_risk", "layer")
                 if k in obj}
        return Manifest(**known)


class HarnessWorkspace:
    """A git-backed directory of harness components. One evolution round = one commit."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)

    # init / assemble
    def init_blank(self, base_prompt: str = "") -> None:
        """Create the blank harness (Meta-Harness setting): a bare skeleton under git."""
        os.makedirs(os.path.join(self.root, "skills"), exist_ok=True)
        p = os.path.join(self.root, "system_prompt.md")
        if not os.path.exists(p):
            open(p, "w").write(base_prompt or "")
        for f in ("memory.md", "tool_notes.md"):
            fp = os.path.join(self.root, f)
            if not os.path.exists(fp):
                open(fp, "w").write("")
        if not os.path.isdir(os.path.join(self.root, ".git")):
            _git(self.root, "init", "-q")
            _git(self.root, "add", "-A")
            _git(self.root, "-c", "user.email=t2t@local", "-c", "user.name=t2t",
                 "commit", "-qm", "round0: blank harness", "--allow-empty")

    def assemble(self) -> str:
        """Assemble the system prompt from the components (labeled sections, empty ones skipped)."""
        parts = []
        base = open(os.path.join(self.root, "system_prompt.md")).read().strip()
        if base:
            parts.append(base)
        sk_dir = os.path.join(self.root, "skills")
        for name in sorted(os.listdir(sk_dir)) if os.path.isdir(sk_dir) else []:
            body = open(os.path.join(sk_dir, name)).read().strip()
            if body:
                parts.append(f"## Skill: {name}\n{body}")
        for fname, title in (("tool_notes.md", "Tool notes"), ("memory.md", "Memory")):
            body = open(os.path.join(self.root, fname)).read().strip()
            if body:
                parts.append(f"## {title}\n{body}")
        return "\n\n".join(parts)

    # evolve / revert
    def apply_manifest(self, m: Manifest, round_idx: int) -> str:
        """Write the edit, commit it and return the commit sha.

        The manifest JSON goes into the commit body, so the predict-then-verify record can be
        recovered from git history alone.
        """
        fp = os.path.join(self.root, m.path)
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        open(fp, "w").write(m.edit)
        _git(self.root, "add", "-A")
        msg = f"round{round_idx}: {m.component} ({m.layer})\n\n" + json.dumps(asdict(m), indent=1)
        # --allow-empty: the optimizer may propose content identical to the current file, and a
        # plain commit would then fail. The round record still goes into history; the verdict
        # grades it INERT.
        _git(self.root, "-c", "user.email=t2t@local", "-c", "user.name=t2t",
             "commit", "-qm", msg, "--allow-empty")
        return _git(self.root, "rev-parse", "HEAD").strip()

    def revert_last(self) -> None:
        try:
            _git(self.root, "-c", "user.email=t2t@local", "-c", "user.name=t2t",
                 "revert", "-n", "HEAD")
        except RuntimeError:
            # HEAD may be an empty commit (identical-content proposal) graded HARMFUL by sweep
            # noise. There is nothing to revert, so only record the verdict.
            _git(self.root, "-c", "user.email=t2t@local", "-c", "user.name=t2t",
                 "commit", "-qm", "revert: gate failed (no-op edit)", "--allow-empty")
            return
        _git(self.root, "-c", "user.email=t2t@local", "-c", "user.name=t2t",
             "commit", "-qm", "revert: gate failed", "--allow-empty")

    # candidate scoring
    def head(self) -> str:
        return _git(self.root, "rev-parse", "HEAD").strip()

    def reset_to(self, sha: str) -> None:
        """Hard-reset to a recorded commit.

        Used to score several candidates against the same base: apply a candidate, sweep, then
        reset_to(base). Unlike revert_last, this leaves no record in git history.
        """
        _git(self.root, "reset", "--hard", sha, "-q")


# Round verdict (grading adapted from AHE)
def verdict(m: Manifest, before: dict, after: dict) -> dict:
    """Grade one round: predicted versus actual per-task outcome changes.

    before/after are {task_id: 0|1} on the same held-out task set. Tasks missing from either
    side (infra failures) are ignored.

    HARMFUL   regressions (1->0), no fixes     -> the caller should revert
    EFFECTIVE fixes (0->1), no regressions     -> keep
    MIXED     fixes and regressions            -> caller's policy (default: revert)
    INERT     nothing changed                  -> keep or revert (the edit had no effect)
    """
    common = sorted(set(before) & set(after))
    fixed = [t for t in common if before[t] < 0.5 <= after[t]]
    regressed = [t for t in common if after[t] < 0.5 <= before[t]]
    label = ("INERT" if not fixed and not regressed else
             "EFFECTIVE" if fixed and not regressed else
             "HARMFUL" if regressed and not fixed else "MIXED")
    return {
        "label": label,
        "delta_phi": (sum(after[t] for t in common) - sum(before[t] for t in common)) / max(1, len(common)),
        "fixed": fixed, "regressed": regressed,
        "predicted_fixes_hit": sorted(set(fixed) & set(m.predicted_fixes)),
        "predicted_fixes_missed": sorted(set(m.predicted_fixes) - set(fixed)),
        "unpredicted_regressions": sorted(set(regressed) - set(m.at_risk)),
        "n_tasks": len(common),
    }


# component -> workspace location ('/' suffix = directory of many files)
COMPONENTS7 = {
    "system_prompt": "system_prompt.md",
    "tool_notes": "tool_notes/",
    "skills": "skills/",
    "middleware": "middleware/",
    "subagents": "subagents/",
    "memory": "memory.md",
    "workflow": "workflow.md",
    # Executable harness code (code_hooks edit space). assemble() never adds it to the
    # prompt; the executor loop loads it via hooks_runtime. It is in the path map so that
    # resolve(), changesets and revert treat it like any other component. Markdown-only
    # edit spaces reject edits to it.
    "harness_code": "harness_code/",
}
_SECTION_TITLES = {
    "workflow": "Workflow", "tool_notes": "Tool notes", "skills": "Skill",
    "middleware": "Loop directives", "subagents": "Delegation recipe", "memory": "Memory",
}


class HarnessWorkspace7(HarnessWorkspace):
    """Workspace with AHE's seven components plus harness_code/; changesets may span files."""

    # init / assemble
    def init_blank(self, base_prompt: str = "") -> None:
        for comp, rel in COMPONENTS7.items():
            p = os.path.join(self.root, rel)
            if rel.endswith("/"):
                os.makedirs(p, exist_ok=True)
            elif not os.path.exists(p):
                os.makedirs(os.path.dirname(p) or self.root, exist_ok=True)
                open(p, "w").write(base_prompt if comp == "system_prompt" else "")
        if not os.path.isdir(os.path.join(self.root, ".git")):
            _git(self.root, "init", "-q")
            _git(self.root, "add", "-A")
            _git(self.root, "-c", "user.email=t2t@local", "-c", "user.name=t2t",
                 "commit", "-qm", "round0: blank harness (7-component)", "--allow-empty")

    def assemble(self) -> str:
        parts = []
        base_p = os.path.join(self.root, "system_prompt.md")
        if os.path.exists(base_p):
            base = open(base_p).read().strip()
            if base:
                parts.append(base)
        for comp in ("workflow", "tool_notes", "skills", "middleware", "subagents", "memory"):
            rel = COMPONENTS7[comp]
            title = _SECTION_TITLES[comp]
            p = os.path.join(self.root, rel)
            if rel.endswith("/"):
                if os.path.isdir(p):
                    for name in sorted(os.listdir(p)):
                        body = open(os.path.join(p, name)).read().strip()
                        if body:
                            parts.append(f"## {title}: {name}\n{body}")
            elif os.path.exists(p):
                body = open(p).read().strip()
                if body:
                    parts.append(f"## {title}\n{body}")
        return "\n\n".join(parts)

    # changesets
    def resolve(self, rel: str) -> str:
        """Map a workspace-relative path to an absolute one, refusing paths outside components.

        A leading 'workspace/' is stripped: the optimizer sees the harness mirrored at
        workspace/ in its run directory and often writes paths relative to that.
        """
        norm = os.path.normpath(rel)
        if norm.startswith("workspace" + os.sep):
            norm = norm[len("workspace" + os.sep):]
        if norm.startswith("..") or os.path.isabs(norm):
            raise ValueError(f"path escapes workspace: {rel!r}")
        files = {r for r in COMPONENTS7.values() if not r.endswith("/")}
        dirs = {r.rstrip("/") for r in COMPONENTS7.values() if r.endswith("/")}
        if norm in files or norm.split(os.sep)[0] in dirs:
            return os.path.join(self.root, norm)
        raise ValueError(f"path {rel!r} not inside a harness component "
                         f"(want one of {sorted(COMPONENTS7.values())})")

    def apply_changeset(self, edits: list, round_idx: int, description: str = "") -> str:
        """Apply a multi-file changeset as one commit and return the new head sha.

        edits: [{"path": rel, "content": full_text} | {"path": rel, "delete": true}]. Only
        full-file writes are accepted, so there are no diffs to misapply.
        """
        if not edits:
            raise ValueError("empty changeset")
        for e in edits:
            fp = self.resolve(e["path"])
            if e.get("delete"):
                if os.path.exists(fp):
                    os.remove(fp)
                continue
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            open(fp, "w").write(e.get("content", ""))
        _git(self.root, "add", "-A")
        msg = (f"round{round_idx}: {description[:120]}\n\n"
               + json.dumps({"files": [e["path"] for e in edits]}, indent=1))
        _git(self.root, "-c", "user.email=t2t@local", "-c", "user.name=t2t",
             "commit", "-qm", msg, "--allow-empty")
        return self.head()
