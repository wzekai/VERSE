"""edit_spaces.py — the edit-space components: what a proposal may change.

Thin constraint layers over verse.workspace.HarnessWorkspace7, which does the actual file and
git work.

full_scaffold  prompt level (as in AHE): multi-file change-sets across the 7 markdown
               components. Rejects harness_code/ (markdown-only harness).
code_hooks     code level (as in AHE, HarnessX, Meta-Harness): full_scaffold plus the
               executable 8th component harness_code/hooks.py (see hooks_runtime). Used by
               every config. The static safety check runs in apply(), so a rejected hooks.py
               never enters the git lineage; the repair loop gets the violation list as the
               apply error.
typed_hooks    exactly one component file per proposal (Self-Harness's "exactly one virtual
               hook" rule, hooks.py:81-106).
"""
from __future__ import annotations

import os

from verse.workspace import HarnessWorkspace7
from verse.evolution.registry import register_component


def _is_code_path(rel: str) -> bool:
    norm = os.path.normpath(rel or "")
    if norm.startswith("workspace" + os.sep):
        norm = norm[len("workspace" + os.sep):]
    return norm.split(os.sep)[0] == "harness_code"


class _WsEditSpace:
    max_files = 100

    def _ws(self, ws_dir: str) -> HarnessWorkspace7:
        return HarnessWorkspace7(ws_dir)

    def init(self, ws_dir: str, base_prompt: str = "") -> None:
        self._ws(ws_dir).init_blank(base_prompt)

    def apply(self, payload: dict, ws_dir: str, round_idx: int) -> str:
        edits = payload.get("edits") or []
        if not edits:
            raise ValueError("proposal contains no edits")
        if len(edits) > self.max_files:
            raise ValueError(f"proposal touches {len(edits)} files > limit {self.max_files}")
        self._check(edits)
        return self._ws(ws_dir).apply_changeset(edits, round_idx,
                                                payload.get("description", ""))

    def revert_last(self, ws_dir: str) -> None:
        self._ws(ws_dir).revert_last()

    def reset_to(self, ws_dir: str, sha: str) -> None:
        self._ws(ws_dir).reset_to(sha)

    def head(self, ws_dir: str) -> str:
        return self._ws(ws_dir).head()

    def assemble(self, ws_dir: str) -> str:
        return self._ws(ws_dir).assemble()

    def _check(self, edits: list) -> None:
        for e in edits:
            if _is_code_path(e.get("path", "")):
                raise ValueError("harness_code/ is not editable in this arm "
                                 "(prompt-level edit space)")


@register_component("edit_space", "full_scaffold")
class FullScaffoldEditSpace(_WsEditSpace):
    def __init__(self, max_files: int = 100):
        self.max_files = int(max_files)


@register_component("edit_space", "code_hooks")
class CodeHooksEditSpace(_WsEditSpace):
    """full_scaffold plus the executable component. The only file allowed under harness_code/
    is hooks.py, and it must pass the static safety check before the change-set is committed."""

    def __init__(self, max_files: int = 100):
        self.max_files = int(max_files)

    def _check(self, edits: list) -> None:
        from verse.evolution.hooks_runtime import audit_hooks_source
        for e in edits:
            if not _is_code_path(e.get("path", "")):
                continue
            norm = os.path.normpath(e["path"])
            if norm.startswith("workspace" + os.sep):
                norm = norm[len("workspace" + os.sep):]
            if norm != os.path.join("harness_code", "hooks.py"):
                raise ValueError(f"only harness_code/hooks.py is editable "
                                 f"(got {e['path']!r})")
            if e.get("delete"):
                continue
            violations = audit_hooks_source(e.get("content", ""))
            if violations:
                raise ValueError("hooks.py failed the static audit — fix these and "
                                 "resubmit:\n" + "\n".join(f"- {v}" for v in violations))


@register_component("edit_space", "typed_hooks")
class TypedHooksEditSpace(_WsEditSpace):
    def __init__(self):
        self.max_files = 1

    def _check(self, edits: list) -> None:
        if len(edits) != 1:
            raise ValueError("typed_hooks requires exactly one edited file per proposal "
                             "(Self-Harness single-hook rule)")
        super()._check(edits)
