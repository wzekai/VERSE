"""Optimizer self-evolution: the optimizer also evolves its own harness.

The baselines and the `verified_*` configurations evolve the executor harness with a fixed
optimizer (called teacher in the code). Here the optimizer also rewrites its own harness,
the optimizer harness. The configurations in `verse/configs/self_teacher/` use this module
through their `meta_teacher` section.

Each round runs two nested loops:

  inner   The optimizer runs its usual episodes and edits the executor harness. Its
          execution primitive is simple verification, `run_episode(task_id, edits)`: run
          one train task once, optionally under draft harness edits. It also has the
          analysis bash sandbox. The plain `self_teacher` configuration has no
          verification, replay or perturbation tools; the `verse_*` configurations add them.

  outer   At the end of each round a meta-episode runs: the optimizer reads its own
          trajectories and the round report, then rewrites its optimizer harness, a
          git-backed directory (teacher_ws/) of prompt, notes, skills, candidate guidance
          and executable tools and hooks (teacher_code/hooks.py). The code is loaded into
          its next inner episodes.

Optimizer harness layout (portable: copy the directory to reuse a trained optimizer):

    teacher_ws/
      MANIFEST.json            format, models and settings (machine-written)
      prompt.md                self-written addition to the optimizer prompt  (<=16 KB)
      notes.md                 memory across rounds                           (<=32 KB)
      search.md                candidate guidance, one block per candidate    (<=4 KB)
      skills/*.md              optimizer skills                               (<=8 KB each)
      teacher_code/hooks.py    executable tools and hooks, audited            (<=64 KB)

Edits to the optimizer harness are not gated on validation score: a single round's
validation change cannot be credited to one optimizer-harness edit. Edits are checked only
mechanically: size limits, no hardcoded task ids in search.md or teacher_code, and a health
check of teacher_code (source audit, load, one test call per tool, test run of each hook).
Code that fails the health check is not loaded in the next round, and the event appears in
the round report. Machine-generated files (round reports, event log) live outside teacher_ws,
so the directory holds only what the optimizer wrote.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time

from verse.evolution.hooks_runtime import (
    BaseHooks, HookedRuntime, HookLoadError, audit_hooks_source, load_hooks)

# Optimizer harness size limits
MAX_PROMPT_CHARS = 16_384
MAX_NOTES_CHARS = 32_768
MAX_SKILL_CHARS = 8_192
MAX_SKILLS = 24
MAX_TEACHER_CODE_CHARS = 65_536       # 2x the executor-side hooks cap
MAX_SEARCH_CHARS = 4_096              # search.md: candidate guidance
_EDITABLE_TOP = {"prompt.md", "notes.md", "search.md", "skills", "teacher_code"}
# Self-edit channels: each channel lists the top-level entries it may write.
# prompt = guidance text added to episodes; tools = executable hooks; notes = memory
# and skills. A config that sets meta_teacher.self_edit_channels may write the union
# of its channels.
_CHANNEL_PATHS = {"prompt": {"prompt.md", "search.md"},
                  "tools": {"teacher_code"},
                  "notes": {"notes.md", "skills"}}

_META_MAX_TURNS = 30
_META_BASH_CAP = 30
_EPISODE_BUDGET = 8                   # run_episode calls per candidate (pool shared with meta)


def _git(ws: str, *args: str) -> str:
    import subprocess
    r = subprocess.run(["git", "-C", ws, *args], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()[:300]}")
    return r.stdout


# Optimizer harness directory
class TeacherWorkspace:
    """teacher_ws/: the optimizer harness as a portable, git-backed directory."""

    def __init__(self, root: str, channels: list | None = None):
        self.root = os.path.abspath(root)
        if channels is not None:
            if not channels:
                raise ValueError("self_edit_channels must be non-empty or absent; "
                                 f"valid: {sorted(_CHANNEL_PATHS)}")
            unknown = [c for c in channels if c not in _CHANNEL_PATHS]
            if unknown:
                raise ValueError(f"unknown self-edit channels {unknown}; "
                                 f"valid: {sorted(_CHANNEL_PATHS)}")
        self.channels = tuple(channels) if channels is not None else None
        self.editable = (_EDITABLE_TOP if channels is None else
                         set().union(*(_CHANNEL_PATHS[c] for c in channels)))

    def init(self, manifest: dict, seed_files: dict | None = None) -> None:
        """Create the optimizer harness directory.

        seed_files ({rel_path: content}, optional) pre-populates it before the round-0
        commit. `verse_*` configurations seed the hand-written guidance of the `verified_*`
        configurations here, so the optimizer can edit it from round 1 on; the seed is
        the root commit in git."""
        if os.path.isdir(os.path.join(self.root, ".git")):
            # resumed from a checkpoint: empty directories are not synced, so recreate them
            os.makedirs(os.path.join(self.root, "skills"), exist_ok=True)
            os.makedirs(os.path.join(self.root, "teacher_code"), exist_ok=True)
            return
        os.makedirs(os.path.join(self.root, "skills"), exist_ok=True)
        os.makedirs(os.path.join(self.root, "teacher_code"), exist_ok=True)
        with open(os.path.join(self.root, "MANIFEST.json"), "w") as f:
            json.dump({"format": "teacher_ws/v1", "created": time.time(), **manifest},
                      f, indent=1)
        for name in ("prompt.md", "notes.md", "search.md"):
            p = os.path.join(self.root, name)
            if not os.path.exists(p):
                open(p, "w").write("")
        for rel, body in (seed_files or {}).items():
            open(os.path.join(self.root, rel), "w").write(body)
        _git(self.root, "init", "-q")
        _git(self.root, "add", "-A")
        _git(self.root, "-c", "user.email=teacher@local", "-c", "user.name=teacher",
             "commit", "-qm",
             "meta round 0: seeded equipment" if seed_files else
             "meta round 0: bare equipment", "--allow-empty")

    # Reads
    def read(self, rel: str) -> str:
        p = os.path.join(self.root, rel)
        return open(p, errors="replace").read() if os.path.isfile(p) else ""

    def skills(self) -> list:
        d = os.path.join(self.root, "skills")
        if not os.path.isdir(d):
            return []
        return [(n, open(os.path.join(d, n), errors="replace").read())
                for n in sorted(os.listdir(d)) if n.endswith(".md")]

    def teacher_code(self) -> str | None:
        src = self.read("teacher_code/hooks.py")
        return src if src.strip() else None

    def head(self) -> str:
        return _git(self.root, "rev-parse", "HEAD").strip()

    def reset_to(self, sha: str) -> None:
        _git(self.root, "reset", "--hard", sha, "-q")

    # Edits
    @staticmethod
    def _rel(path: str) -> str:
        """Normalize an edit path. A leading 'teacher_ws/' is accepted and stripped,
        because the run directory the optimizer browses shows its files under that
        prefix."""
        rel = os.path.normpath(path or "")
        if rel == "teacher_ws" or rel.startswith("teacher_ws" + os.sep):
            rel = os.path.relpath(rel, "teacher_ws")
        return rel

    def validate_edits(self, edits: list, banned_strings: list) -> str:
        """Return '' when the edits are valid, else an error message for the optimizer."""
        for e in edits:
            rel = self._rel(e.get("path", ""))
            if rel.startswith("..") or os.path.isabs(rel):
                return f"path escapes teacher_ws: {e.get('path')!r}"
            top = rel.split(os.sep)[0]
            if top not in _EDITABLE_TOP:
                return (f"path {rel!r} is not editable — teacher_ws components are: "
                        f"prompt.md, notes.md, skills/*.md, teacher_code/hooks.py")
            if top not in self.editable:
                return (f"path {rel!r} is outside this arm's self-edit channels "
                        f"({', '.join(self.channels or ())}) — writable components: "
                        f"{', '.join(sorted(self.editable))}")
            body = e.get("content", "") or ""
            if e.get("delete"):
                continue
            if rel == "prompt.md" and len(body) > MAX_PROMPT_CHARS:
                return f"prompt.md is {len(body)} chars > limit {MAX_PROMPT_CHARS}"
            if rel == "notes.md" and len(body) > MAX_NOTES_CHARS:
                return f"notes.md is {len(body)} chars > limit {MAX_NOTES_CHARS}"
            if rel == "search.md":
                if len(body) > MAX_SEARCH_CHARS:
                    return f"search.md is {len(body)} chars > limit {MAX_SEARCH_CHARS}"
                hardwired = [s for s in banned_strings if s in body]
                if hardwired:
                    return ("search guidance must be GENERAL — hardwired task ids "
                            f"found: {', '.join(hardwired[:3])}")
            if top == "skills":
                if not rel.endswith(".md"):
                    return f"skills entries must be .md files: {rel!r}"
                if len(body) > MAX_SKILL_CHARS:
                    return f"{rel} is {len(body)} chars > limit {MAX_SKILL_CHARS}"
            if top == "teacher_code":
                if rel != os.path.join("teacher_code", "hooks.py"):
                    return "the only executable file is teacher_code/hooks.py"
                if len(body) > MAX_TEACHER_CODE_CHARS:
                    return f"hooks.py is {len(body)} chars > limit {MAX_TEACHER_CODE_CHARS}"
                v = audit_hooks_source(body, max_chars=MAX_TEACHER_CODE_CHARS)
                if v:
                    return "teacher_code audit failed:\n" + "\n".join(f"- {x}" for x in v)
                hardwired = [s for s in banned_strings if s in body]
                if hardwired:
                    return ("tools must be GENERAL — hardwired task ids found in "
                            f"teacher_code: {', '.join(hardwired[:3])}")
        # checkpoint syncs drop empty directories, so skills/ may be absent on a
        # resumed run even though init() created it
        skills_dir = os.path.join(self.root, "skills")
        n_existing = (sum(1 for n in os.listdir(skills_dir) if n.endswith(".md"))
                      if os.path.isdir(skills_dir) else 0)
        if n_existing + sum(1 for e in edits
                            if self._rel(e.get("path", "")).startswith("skills" + os.sep)
                            and not e.get("delete")) > MAX_SKILLS * 2:
            return f"too many skills (cap {MAX_SKILLS})"
        return ""

    def apply(self, edits: list, meta_round: int, description: str) -> str:
        for e in edits:
            fp = os.path.join(self.root, self._rel(e.get("path", "")))
            if e.get("delete"):
                if os.path.isfile(fp):
                    os.remove(fp)
                continue
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            open(fp, "w").write(e.get("content", "") or "")
        _git(self.root, "add", "-A")
        _git(self.root, "-c", "user.email=teacher@local", "-c", "user.name=teacher",
             "commit", "-qm", f"meta round {meta_round}: {description[:120]}",
             "--allow-empty")
        return self.head()


# VERSE seed
def _seed_equipment() -> dict:
    """Initial optimizer harness for `verse_*` configurations (`seed_wisdom: true`).

    It is an exact copy of the hand-written guidance the `verified_*` configurations
    receive, and nothing else:

        prompt.md   the verification-tool usage strategy (intervener.PROBE_WISDOM) and
                    driver._VERIFIED_CODE_RULE, the same texts a `verified_*` optimizer
                    receives in its prompt
        search.md   the three candidate-guidance blocks (driver._LENSES), separated by
                    `---` lines; search_lenses() delivers them under its own header

    skills/ and notes.md stay empty, because the `verified_*` configurations have no
    hand-written skills or notes. Imports are lazy because driver imports this module."""
    from verse.evolution.driver import _LENSES, _VERIFIED_CODE_RULE
    from verse.evolution.intervener import PROBE_WISDOM
    lens_blocks = "\n---\n".join(suffix.strip() for _name, suffix in _LENSES)
    return {"prompt.md": (PROBE_WISDOM.strip() + "\n\n"
                          + _VERIFIED_CODE_RULE.strip() + "\n"),
            "search.md": lens_blocks + "\n"}


# Health check
def health_check(src: str) -> str:
    """Mechanical test of teacher_code/hooks.py.

    Runs the source audit, loads the code, makes one test call per declared tool and
    test-runs each hook. Returns '' when healthy, else a failure report. It runs when
    the code is submitted and again each time it is loaded; it never evaluates on
    validation tasks."""
    v = audit_hooks_source(src, max_chars=MAX_TEACHER_CODE_CHARS)
    if v:
        return "audit: " + "; ".join(v[:3])
    try:
        rt = HookedRuntime(src)
    except HookLoadError as e:
        return f"load: {e}"
    specs = rt.extra_tool_specs()
    if rt.errors:
        return "extra_tools: " + json.dumps(rt.errors[:2])

    class _DryEnv:
        def new_tool_call(self):          # HookedRuntime.run_tool calls this per call
            pass
        def bash(self, cmd):
            return "(dry-run: bash unavailable during the health check)"
        def run_episode(self, task_id, edits=None):
            return "(dry-run: run_episode unavailable during the health check)"
        def llm(self, prompt, max_tokens=1024):
            return "(dry-run: llm unavailable during the health check)"

    for spec in specs:
        out = rt.run_tool(spec["name"], {}, _DryEnv(), turn=-1)
        if rt.errors:
            return (f"dry call of tool {spec['name']!r} raised: "
                    + json.dumps(rt.errors[:1]))
        if not isinstance(out, str):
            return f"tool {spec['name']!r} returned {type(out).__name__}, want str"
    # Test-run the pipeline hooks on representative inputs. HookedRuntime is fail-open
    # during episodes, so a hook that raises on first use would silently do nothing;
    # catching it here lets the meta-episode fix it.
    rt.system_prompt("dry-run brief")
    rt.before_llm([{"role": "user", "content": "dry"}], 0)
    rt.after_llm([{"type": "text", "text": "dry"}], 0)
    rt.before_tool("bash", {"command": "true"}, 0)
    rt.after_tool("bash", {"command": "true"}, "dry output", 0)
    rt.on_turn_end(0)
    pipeline_errs = [e for e in rt.errors if e.get("hook") != "loop"]
    if pipeline_errs:
        return "pipeline dry-run: " + json.dumps(pipeline_errs[:2])
    return ""


# Primitives available to the optimizer's own tools
class TeacherEnv:
    """The `env` argument passed to the optimizer's self-written tools (run_tool).

    It exposes three primitives: run_episode, bash (the analysis sandbox) and llm.
    Anything richer is built by the optimizer.

    `verse_*` configurations also pass the verification tools (`probes`, a loaded
    T2TProbeKit, plus `probe_ctx`). Self-written code can then call fix_probe, ablate,
    substitute and replay, for example in a tool that groups failures, runs fix_probe on
    the largest group and keeps only flips confirmed twice. The tools' internals (docker
    execution, grading, budget accounting) stay system-owned. The calls return the same
    '[VERDICT] setup -> outcome' strings as the tool versions and cost one budget unit
    each, as those do."""

    def __init__(self, run_episode_fn, bash_fn, llm_fn, probes=None, probe_ctx=None):
        self.run_episode = run_episode_fn      # (task_id, edits=None) -> str
        self.bash = bash_fn                    # (cmd) -> str (analysis sandbox)
        self.llm = llm_fn                      # (prompt, max_tokens) -> str (metered)
        if probes is not None and probe_ctx is not None:
            def _fmt(rec):
                return f"[{rec.verdict.upper()}] {rec.setup} -> {rec.outcome}"
            self.fix_probe = lambda edits, target_task_ids: _fmt(
                probes.fix_probe(probe_ctx, edits, target_task_ids))
            self.ablate = lambda task_id, step: _fmt(
                probes.ablate(probe_ctx, task_id, step))
            self.substitute = lambda task_id, step, new_command: _fmt(
                probes.substitute(probe_ctx, task_id, step, new_command))
            self.replay = lambda task_id: _fmt(probes.replay(probe_ctx, task_id))

    def new_tool_call(self) -> None:
        """Per-call reset hook; HookedRuntime.run_tool calls it before each tool call."""


class EpisodeBudget:
    """Per-round run_episode budget, shared by all candidate episodes and the
    meta-episode (thread-safe). It is sized like the verification-tool budget of the
    `verified_*` configurations: the same allowance per candidate, multiplied by the
    number of candidates."""

    def __init__(self, n: int, wall_s: float):
        self._n = n
        self._wall_s = wall_s
        self._used = 0
        self._deadline = time.monotonic() + wall_s
        self._lock = threading.Lock()

    def take(self) -> int | None:
        """Reserve one run. Returns a unique 1-based slot number, or None when the
        count or wall-clock budget is spent. Callers use the slot as a collision-free
        id, so concurrent episodes running the same task get separate scratch
        directories and transcripts."""
        with self._lock:
            if self._used >= self._n or time.monotonic() > self._deadline:
                return None
            self._used += 1
            return self._used

    def new_phase(self) -> None:
        """Restart the wall clock for the next consumer of the same count budget (as in
        T2TProbeKit.new_phase). The deadline starts at the beginning of the round, and
        candidate generation, screening and the validation sweep can outlast it; without
        a reset the meta-episode could never call run_episode."""
        with self._lock:
            self._deadline = time.monotonic() + self._wall_s

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> int:
        return max(0, self._n - self._used)


class SharedProbeBudget:
    """EpisodeBudget-compatible view over a loaded T2TProbeKit (`verse_*` configurations).

    With the verification tools loaded, the round has a single execution pool: each
    fix_probe target, replay, ablate, substitute and run_episode call costs one unit of
    the same allowance. The driver already scales probes.budget by the number of
    candidates; this adapter routes run_episode calls through it."""

    def __init__(self, kit):
        self._kit = kit

    def take(self) -> int | None:
        return self._kit.take_unit()

    def new_phase(self) -> None:
        self._kit.new_phase()

    @property
    def used(self) -> int:
        return self._kit.used

    @property
    def remaining(self) -> int:
        return self._kit.remaining()


# The method
# Tool descriptions given to the optimizer state only the contract: what the tool does,
# its arguments, its return value and its cost. Usage advice is left out of
# system-provided text; what the optimizer learns about usage lives in its own harness
# (prompt.md, skills, its own tool descriptions), where it can revise it.
_RUN_EPISODE_SPEC = {
    "name": "run_episode",
    "description": ("Run one TRAIN task once. Optional `edits` (same format as "
                    "submit_proposal edits) are applied to an isolated scratch copy of "
                    "the executor harness for this run only. Returns PASS/FAIL, the "
                    "verifier's verdict line, and the transcript path. Budget: one "
                    "shared pool per round across all episodes."),
    "input_schema": {"type": "object", "properties": {
        "task_id": {"type": "string", "description": "a TRAIN task id"},
        "edits": {"type": "array", "items": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"},
            "delete": {"type": "boolean"}}, "required": ["path"]}}},
        "required": ["task_id"]}}

_META_BRIEF = """You are a teacher agent that improves a frozen coding agent's harness, and
this is your META-EPISODE: the round of teaching just finished, and now you improve YOUR OWN
equipment before the next round.

## What just happened (your report card)
<REPORT_CARD>

## Your equipment workspace (edit via submit_proposal; current contents under teacher_ws/)
  prompt.md              your self-authored addition to your teaching prompt (<=16KB)
  notes.md               your cross-round memory (<=32KB — plenty; keep a running log)
  search.md              candidate guidance for your N parallel teaching episodes each
                         round (<=4KB). Blocks separated by `---` lines; block i is
                         appended to candidate i's prompt. Fewer blocks than candidates =
                         the rest run with no guidance (plain sampling); empty = all
                         plain. The journal and your report card record which candidate
                         won each round's screening.
  skills/*.md            teaching skills, mounted into your prompt every round (<=8KB each)
  teacher_code/hooks.py  EXECUTABLE Python that mounts into YOUR OWN next teaching episode
                         (<=64KB) — the same hook surface the executor's harness code has,
                         but running on YOUR loop. Define `class Hooks(BaseHooks)`; all
                         methods optional:
                             def system_prompt(self, assembled)       # str -> str: your brief
                             def before_llm(self, msgs, state)        # list -> list: your
                                 # message view each turn
                             def after_llm(self, content, state)      # list -> list: your raw
                                 # reply blocks before the loop reads them
                             def before_tool(self, name, args, state) # dict -> dict: your tool
                                 # args; return str to block, or (reason, synthetic_result)
                             def after_tool(self, name, args, obs, state)  # str -> str: raw
                                 # tool output before you read it
                             def on_turn_end(self, turn, state)       # str|None -> a note
                                 # injected into your next turn
                             def extra_tools(self):  # declare new tools: [{name,
                                 # description, input_schema}]
                             def run_tool(self, name, args, env, state):  # implement them;
                                 # env exposes the atomic capabilities:
                                 #   env.run_episode(task_id, edits=None) -> str
                                 #   env.bash(cmd) -> str
                                 #   env.llm(prompt) -> str<ENV_PROBES>
                             loop = {...}  # optional knobs (max_turns only-shrink, obs_cap...)
                         Imports are restricted to pure-computation modules; the audit
                         rejects file/network/process access. A raising hook degrades to
                         identity for that call; your report card shows every failure.
                         (During THIS meta-episode your pipeline hooks are unmounted and
                         only your tools are live — so you can always repair broken code.)

## Constraint
Your equipment must be GENERAL: task ids rotate between rounds and arms — hardwired ids
are rejected by the audit.

## What to do now
Read this round's evidence (your teaching episode's audit files and the round journal are
under the run directory), decide what would have made your teaching BETTER, and submit ONE
change-set to your equipment. If nothing is worth changing, submit an empty edits array
with a description starting with 'NO_CHANGES' — an empty submit without that marker is
bounced back (it usually means file contents were left out by mistake).
predicted_fixes/at_risk do not apply here — pass [].
Your equipment persists across all rounds; invest in what compounds.

IMPORTANT: files you draft in the bash sandbox (/scratch) are NOT submitted automatically —
the ONLY thing that changes your equipment is the full file content inside submit_proposal's
`edits` array. Budget your turns so the submit happens: drafting for 28 turns and running
out before calling submit_proposal loses everything (this has happened).

## Run directory map
  rounds.jsonl                    the full round journal (verdicts, val scores)
  round_NN/intervener_audit_*.jsonl  your own teaching episodes, every tool call logged
  round_NN/self_episodes/         transcripts of your run_episode calls
  report_cards/round_NN.md        all past report cards (NN = zero-padded round, e.g. round_01.md)
  teacher_ws/                     your current equipment
"""


class MetaTeacher:
    """Runs optimizer self-evolution: loads the optimizer harness and run_episode into
    each candidate episode, and runs the meta-episode at the end of each round.

    The driver calls, each round (only when the config has a `meta_teacher` section):
        prompt_suffix = mt.begin_round(rnd, ctx, rdir, n_candidates=n)
        lenses = mt.search_lenses(n)
        mt.episode_tools(ctx, rdir), mt.episode_hooks(ctx, rdir, key=lens)  # per candidate
        mt.end_round(rnd, rec, rdir, ctx)
    """

    def __init__(self, cfg: dict, out_dir: str, intervener_cfg: dict, executor: dict):
        self.cfg = cfg or {}
        self.out_dir = out_dir
        self.intervener_cfg = intervener_cfg
        self.executor = executor
        self.self_edit = bool(self.cfg.get("self_edit", True))
        # execute=False (the reflect_only ablation): self-editing stays but run_episode
        # is not offered, so the optimizer can read trajectories and validation feedback
        # but cannot execute anything. With bare_static (run_episode, no self-editing)
        # and self_teacher (both), this separates self-editing from execution.
        self.execute = bool(self.cfg.get("execute", True))
        # seed_wisdom=True (set in the `verse_*` configurations): initialize prompt.md and
        # search.md with an exact copy of the `verified_*` configurations' hand-written
        # verification strategy and candidate guidance; skills/ and notes.md stay empty.
        # Round 0 therefore uses the same guidance as the matching `verified_*`
        # configuration, and later rounds may rewrite it. Default False: the optimizer
        # starts with contract-only tool descriptions.
        self.seed_wisdom = bool(self.cfg.get("seed_wisdom", False))
        # self_edit_channels: None allows all channels; a list restricts which parts of
        # the optimizer harness the meta-episode may write
        self.channels = self.cfg.get("self_edit_channels")
        self.ws = TeacherWorkspace(os.path.join(out_dir, "teacher_ws"),
                                   channels=self.channels)
        self.ws.init({"teacher": intervener_cfg.get("model"),
                      "student": executor.get("model"),
                      "self_edit": self.self_edit, "execute": self.execute,
                      "seed_wisdom": self.seed_wisdom},
                     seed_files=(_seed_equipment() if self.seed_wisdom else None))
        os.makedirs(os.path.join(out_dir, "report_cards"), exist_ok=True)
        self._events = os.path.join(out_dir, "teacher_events.jsonl")
        self._budget: EpisodeBudget | None = None
        self._round_tools_log: list = []
        # The verification tools (`verse_*` configurations), set each round when the config
        # loads them. Self-written code reaches them through env (see TeacherEnv).
        # None for the plain self_teacher configurations.
        self._probes = None
        self._probe_ctx = None

    def _event(self, kind: str, **kw) -> None:
        with open(self._events, "a") as f:
            f.write(json.dumps({"ts": time.time(), "kind": kind, **kw}) + "\n")

    # Inner loop: setup per round and per episode
    def begin_round(self, rnd: int, ctx, rdir: str, n_candidates: int = 1) -> str:
        """Set up the round (fresh shared budget, health check of teacher_code) and
        return the prompt suffix that carries the optimizer harness.

        n_candidates scales the run_episode pool the same way the driver scales
        probes.budget for the `verified_*` configurations (base * n_candidates), so
        each candidate episode keeps the full per-episode allowance.

        `verse_*` configurations (verification tools in ctx): run_episode and the
        verification tools draw from one pool, the T2TProbeKit budget, which the driver
        has already scaled."""
        self._probes = getattr(ctx, "probes", None)
        self._probe_ctx = ctx if self._probes is not None else None
        if self._probes is not None:
            self._budget = SharedProbeBudget(self._probes)
        else:
            self._budget = EpisodeBudget(
                int(self.cfg.get("episode_budget", _EPISODE_BUDGET)) * max(1, n_candidates),
                float(self.cfg.get("episode_wall_s", 3600)))
        self._round_tools_log = []
        self._mounted_src = self.ws.teacher_code()
        if self._mounted_src:
            err = health_check(self._mounted_src)
            if err:
                self._event("equipment_unmounted", round=rnd, reason=err[:400])
                self._mounted_src = None
        return self._equipment_prompt(rnd)

    def episode_tools(self, ctx, rdir: str) -> list:
        """Tools for one episode: run_episode. The optimizer's self-written tools are
        loaded separately through episode_hooks(). The run_episode budget is one pool
        per round, shared by all episodes and the meta-episode. With execute=False
        (reflect_only) no tool is returned."""
        if not self.execute:
            return []
        return [{"spec": _RUN_EPISODE_SPEC,
                 "fn": lambda args: self._run_episode(
                     ctx, rdir, args.get("task_id", ""), args.get("edits"))}]

    def search_lenses(self, n: int) -> list:
        """Candidate guidance for the round's N parallel episodes, read from the
        optimizer's own search.md.

        The N candidates and the shared screening are fixed by the protocol for every
        configuration; only the guidance text comes from the optimizer. search.md holds
        one block per candidate, separated by lines containing only `---`. Candidates
        beyond the number of blocks get no guidance (plain sampling, as in the
        baselines); an empty search.md leaves all candidates plain. Extra blocks are
        ignored. The names (self_a, sample_b, ...) record the source in the round
        journal."""
        raw = self.ws.read("search.md").strip()
        blocks = ([b.strip()[:MAX_SEARCH_CHARS] for b in
                   __import__("re").split(r"(?m)^\s*---\s*$", raw) if b.strip()]
                  if raw else [])
        lenses = []
        for i in range(n):
            if i < len(blocks):
                lenses.append((f"self_{chr(97 + i)}",
                               "\n\n## Your candidate guidance this round (self-authored "
                               "search.md)\n" + blocks[i]))
            else:
                lenses.append((f"sample_{chr(97 + i)}", ""))
        return lenses

    def episode_hooks(self, ctx, rdir: str, key: str = "tools"):
        """Return (HookedRuntime, env) that loads the optimizer's self-written code onto
        its own episode loop.

        The code gets the same hooks the executor loop exposes (system_prompt,
        before_llm, after_llm, before_tool, after_tool, on_turn_end, loop settings) plus
        the tools it defines, which act through env. Each episode gets a fresh runtime,
        since concurrent candidates must not share hook state, and tool calls are
        recorded in the round's usage log. Returns None when no code is loaded."""
        if not self._mounted_src:
            return None
        rt = HookedRuntime(self._mounted_src, max_chars=MAX_TEACHER_CODE_CHARS)
        # `key` gives each episode its own /scratch, so concurrent candidates do not see
        # each other's drafts. The driver passes the candidate's name, which is also the
        # scratch key of that episode's bash tool, so self-written tools and bash share
        # one /scratch (self-written tools may read files staged with bash).
        env = TeacherEnv(
            run_episode_fn=(
                (lambda tid, edits=None: self._run_episode(ctx, rdir, tid, edits))
                if self.execute else
                (lambda tid, edits=None: "ERROR: run_episode is not available in "
                                         "this configuration (no execution capability)")),
            bash_fn=lambda cmd: self._analysis_bash(rdir, cmd, key),
            llm_fn=self._metered_llm,
            probes=self._probes, probe_ctx=self._probe_ctx)
        inner_run_tool = rt.run_tool
        def _logged_run_tool(name, args, cap, turn):
            r = inner_run_tool(name, args, cap, turn)
            self._round_tools_log.append(
                {"tool": name, "error": str(r).startswith("ERROR")})
            return r
        rt.run_tool = _logged_run_tool
        return rt, env

    def _equipment_prompt(self, rnd: int) -> str:
        if not self.execute:
            budget_line = ""
        elif self._probes is not None:
            # `verse_*`: one pool serves the verification tools and run_episode; state this
            # so the optimizer does not budget them as separate allowances
            budget_line = (f"### execution budget this round: "
                           f"{self._budget.remaining} runs — ONE shared pool across "
                           "probe calls (each fix_probe target / ablate / substitute / "
                           "replay costs one) and run_episode calls"
                           + (", including your meta-episode" if self.self_edit else ""))
        else:
            budget_line = (f"### run_episode budget this round: "
                           f"{self._budget.remaining} runs"
                           + (" (shared with your meta-episode — spend deliberately)"
                              if self.self_edit else ""))
        if not self.self_edit:                       # bare_static: budget line only
            return "\n\n" + budget_line
        parts = ["\n\n## Your self-authored teaching equipment (teacher_ws/)"]
        prompt = self.ws.read("prompt.md").strip()
        if prompt:
            parts.append(prompt[:MAX_PROMPT_CHARS])
        for name, body in self.ws.skills():
            if body.strip():
                parts.append(f"### Skill: {name}\n{body.strip()[:MAX_SKILL_CHARS]}")
        notes = self.ws.read("notes.md").strip()
        if notes:
            parts.append(f"### Your notes\n{notes[:MAX_NOTES_CHARS]}")
        if len(parts) == 1:
            parts.append("(empty — you have not written any equipment yet; the "
                         "meta-episode at this round's end is where you do)")
        parts.append(budget_line)
        return "\n\n".join(parts)

    # Primitives
    def _run_episode(self, ctx, rdir: str, task_id: str, edits=None) -> str:
        """Simple verification: run one train task once, optionally under draft edits
        applied to a scratch copy of the executor harness. Each call costs one budget
        unit."""
        from verse.evolution.driver import run_task
        from verse.evolution.hooks_runtime import load_hooks_source
        from verse.workspace import HarnessWorkspace7

        allowed = ctx.train_tids or set()
        if task_id not in ctx.insts_by_tid:
            return f"ERROR: unknown task id {task_id!r} — pick one from the evidence index"
        if allowed and task_id not in allowed:
            return (f"ERROR: {task_id} is a validation task — run_episode is TRAIN-only "
                    "(val is the held-back measurement set)")
        slot = self._budget.take()
        if slot is None:
            return (f"ERROR: run_episode budget exhausted "
                    f"({self._budget.used} used this round)")
        scratch = os.path.join(ctx.scratch_dir or "/tmp",
                               f"self_ep_{task_id[:40]}_{slot}")
        shutil.rmtree(scratch, ignore_errors=True)
        shutil.copytree(ctx.ws_dir, scratch)
        try:
            if edits:
                ws = HarnessWorkspace7(scratch)
                ws.apply_changeset(list(edits), round_idx=-1, description="run_episode draft")
            sp = HarnessWorkspace7(scratch).assemble()
            hooks_src = load_hooks_source(scratch)
        except Exception as e:
            return f"ERROR: draft edits failed to apply: {str(e)[:300]}"
        ex = ctx.executor or {}
        try:
            phi, tr = run_task(ctx.insts_by_tid[task_id], sp, "bedrock", ex["model"],
                               max_turns=int(ex.get("max_turns", 100)),
                               bedrock_region=ex.get("regions", "us-west-2"),
                               hooks_src=hooks_src)
        except Exception as e:
            return f"ERROR: episode infrastructure failed: {str(e)[:300]}"
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        ep_dir = os.path.join(rdir, "self_episodes")
        os.makedirs(ep_dir, exist_ok=True)
        rel = f"self_episodes/{task_id[:60]}_{slot}.json"
        with open(os.path.join(ep_dir, os.path.basename(rel)), "w") as f:
            json.dump({"task_id": task_id, "phi": phi if phi == phi else None,
                       "edits_applied": bool(edits), "transcript": tr}, f)
        verdict_line = ""
        for m in reversed(tr):
            c = m.get("content", "") if isinstance(m, dict) else ""
            if isinstance(c, str) and c.startswith(("VERIFIER", "TEST FAILURES",
                                                    "TEST RESULT")):
                verdict_line = c[:300]
                break
        status = ("NaN (infra)" if phi != phi else "PASS" if phi >= 0.5 else "FAIL")
        return (f"{task_id}: {status}"
                + (f" | {verdict_line}" if verdict_line else "")
                + f" | transcript: {rel}"
                + f" | budget left: {self._budget.remaining}")

    def _analysis_bash(self, rdir: str, cmd: str, key: str = "tools") -> str:
        from verse.evolution.intervener import _tool_teacher_bash
        return _tool_teacher_bash(rdir, os.path.join(rdir, "teacher_scratch", key), cmd)

    def _metered_llm(self, prompt: str, max_tokens: int = 1024) -> str:
        from verse.runtime.executor import _invoke_bedrock
        if not isinstance(prompt, str) or not prompt.strip():
            return "ERROR: empty prompt"
        try:
            return _invoke_bedrock(self.intervener_cfg["model"],
                                   self.intervener_cfg.get("regions", "us-west-2"),
                                   {"max_tokens": min(max(1, int(max_tokens)), 4096),
                                    "messages": [{"role": "user",
                                                  "content": prompt[:60_000]}]})
        except Exception as e:
            return f"ERROR: llm call failed: {e!r}"

    # Outer loop: the meta-episode
    def end_round(self, rnd: int, rec: dict, rdir: str, ctx) -> None:
        card = self._report_card(rnd, rec, rdir)
        with open(os.path.join(self.out_dir, "report_cards", f"round_{rnd:02d}.md"),
                  "w") as f:
            f.write(card)
        if not self.self_edit:                       # bare_static: no meta-episode
            return
        try:
            if self._budget is not None:
                # restart the wall clock for the meta-episode's remaining count budget
                # (see EpisodeBudget.new_phase)
                self._budget.new_phase()
            self._meta_episode(rnd, card, ctx)
        except Exception as e:
            # fail-open: a failed meta-episode skips this round's self-edit but does
            # not stop the run
            self._event("meta_episode_failed", round=rnd, error=str(e)[:400])
            print(f"[meta_teacher] round {rnd}: meta-episode failed ({e})", flush=True)

    def _meta_episode(self, rnd: int, card: str, ctx) -> None:
        from verse.evolution.intervener import run_intervener
        from verse.evolution.protocols import EvolutionContext

        banned = sorted(ctx.insts_by_tid)            # no hardcoded task ids in tools

        def _validate(payload):
            edits = payload.get("edits") or []
            err = self.ws.validate_edits(edits, banned)
            if err:
                return err
            # Full health check of draft teacher_code inside the episode. The source
            # audit passes code that cannot load (e.g. a missing `class Hooks(BaseHooks)`);
            # rejecting it here lets the optimizer fix it while it still has the
            # context. Pure Python, fast.
            for e in edits:
                if self.ws._rel(e.get("path", "")) == os.path.join(
                        "teacher_code", "hooks.py") and not e.get("delete"):
                    body = e.get("content", "") or ""
                    if body.strip():                 # an empty file removes the code
                        hc = health_check(body)
                        if hc:
                            return ("teacher_code/hooks.py failed its health check — "
                                    "fix and resubmit:\n" + hc)
            return ""

        meta_ctx = EvolutionContext(
            round_idx=rnd, ws_dir=self.ws.root, traj_dir="",
            insts_by_tid=ctx.insts_by_tid, train_tids=ctx.train_tids,
            probes=None, scratch_dir=ctx.scratch_dir, executor=ctx.executor)
        # plain replacement, not str.format — the brief's code examples contain braces
        prompt = _META_BRIEF.replace("<REPORT_CARD>", card)
        if self.channels is not None:
            prompt += ("\n\n## Channel restriction (this configuration)\n"
                       "Only these equipment components are writable: "
                       + ", ".join(sorted(self.ws.editable))
                       + ". Edits to any other component are rejected.")
        # `verse_*`: self-written tools can also call the verification tools through env.
        # Only the call signatures are listed, and only when the tools are loaded this
        # round.
        prompt = prompt.replace("<ENV_PROBES>", (
            "\n                                 #   env.fix_probe(edits, target_task_ids) -> str"
            "\n                                 #   env.ablate(task_id, step) -> str"
            "\n                                 #   env.substitute(task_id, step, new_command) -> str"
            "\n                                 #   env.replay(task_id) -> str"
            "\n                                 #   (probes share the round's execution pool)")
            if self._probes is not None else "")
        if not self.execute:
            # reflect_only: run_episode is unavailable, so remove its lines from the brief
            # (the env entry and the self_episodes/ line of the run-directory map)
            prompt = "\n".join(
                ln for ln in prompt.splitlines()
                if "env.run_episode" not in ln and "self_episodes/" not in ln)
        head_before = self.ws.head()
        # The meta-episode gets run_episode (drawing on the round's remaining shared
        # budget) and the tools of the current teacher_code, so it can try a tool before
        # rewriting it. hooks_pipeline=False keeps the pipeline hooks off, so broken
        # self-written hooks cannot block the episode that would repair them.
        rdir = os.path.join(self.out_dir, f"round_{rnd:02d}")
        proposal = run_intervener(
            prompt, self.out_dir, meta_ctx,
            model=self.intervener_cfg["model"],
            regions=self.intervener_cfg.get("regions", "us-west-2"),
            max_turns=int(self.cfg.get("meta_max_turns", _META_MAX_TURNS)),
            audit_path=os.path.join(self.out_dir, "report_cards",
                                    f"meta_audit_r{rnd:02d}.jsonl"),
            validate_payload=_validate, allow_empty_edits=True,
            extra_tools=self.episode_tools(ctx, rdir),
            # key gives the meta-episode its own /scratch, separate from the candidate
            # episodes' scratch directories under the same rdir
            teacher_hooks=self.episode_hooks(ctx, rdir, key=f"meta_r{rnd:02d}"),
            hooks_pipeline=False,
            bash_cap=int(self.cfg.get("meta_bash_cap", _META_BASH_CAP)),
            # submits carry full file contents, so the output limit must fit a 64 KB
            # hooks.py in one submit
            max_out_tokens=int(self.cfg.get("meta_max_out_tokens", 32768)))
        edits = proposal.meta.get("edits") or []
        if not edits:
            self._event("meta_no_edits", round=rnd)
            return
        err = self.ws.validate_edits(edits, banned)  # final turn skips in-episode check
        if err:
            self._event("meta_edits_rejected", round=rnd, reason=err[:400])
            return
        # Health check again, for the forced final turn, which skips in-episode
        # validation. If hooks.py fails, drop only that edit and commit the rest
        # (notes, prompt, skills).
        dropped = []
        code_edit = next((e for e in edits if self.ws._rel(e.get("path", "")) ==
                          os.path.join("teacher_code", "hooks.py")
                          and not e.get("delete")), None)
        if code_edit and (code_edit.get("content", "") or "").strip():
            hc = health_check(code_edit["content"])
            if hc:
                edits = [e for e in edits if e is not code_edit]
                dropped.append("teacher_code/hooks.py")
                self._event("meta_code_dropped", round=rnd, reason=hc[:400])
        if not edits:
            self._event("meta_no_edits", round=rnd)
            return
        sha = self.ws.apply(edits, rnd, proposal.description)
        # The committed teacher_code must also be healthy (e.g. after a delete edit);
        # otherwise roll back the commit
        src = self.ws.teacher_code()
        if src:
            hc = health_check(src)
            if hc:
                self.ws.reset_to(head_before)
                self._event("meta_health_check_failed", round=rnd, reason=hc[:400])
                return
        self._event("meta_committed", round=rnd, sha=sha,
                    files=[e.get("path") for e in edits],
                    **({"dropped": dropped} if dropped else {}),
                    description=proposal.description[:200])
        print(f"[meta_teacher] round {rnd}: equipment updated "
              f"({[e.get('path') for e in edits]})", flush=True)

    # Round report
    def _report_card(self, rnd: int, rec: dict, rdir: str) -> str:
        """Build the round report the optimizer reads in its meta-episode, from data the
        driver already records (no extra evaluation)."""
        v = rec.get("verdict") or {}
        lines = [f"# Teaching report card — round {rnd}",
                 f"- proposal: {rec.get('description', '(none)')[:200]}",
                 f"- verdict: {v.get('label')} (kept={rec.get('kept')}) — "
                 f"val {rec.get('val_score_before', '?')} -> {rec.get('val_score_after', '?')}"]
        # edit size versus outcome for this round (a fact, not advice)
        if rec.get("edit_bytes") is not None:
            lines.append(f"- edit size: {len(rec.get('files') or [])} file(s), "
                         f"{rec['edit_bytes']:,} bytes -> net "
                         f"{len(v.get('fixed') or [])} fixed / "
                         f"{len(v.get('regressed') or [])} regressed")
        if rec.get("error"):
            lines.append(f"- ROUND ERROR: {rec['error'][:300]}")
        fc = rec.get("flip_confirm")
        if fc:
            conf = sum(1 for x in fc if x.get("confirmed"))
            flaky = sum(1 for x in fc if x.get("claim") == "flaky")
            lines.append(f"- flip confirmation: {conf}/{len(fc)} flips reproduced"
                         + (f"; {flaky} coin-flip task(s) excluded" if flaky else "")
                         + " (unreproduced flips were noise, not your doing)")
        pa = rec.get("prediction_attribution")
        if pa:
            from collections import Counter
            lines.append(f"- your last kept proposal's predictions: {dict(Counter(pa.values()))}")
        sr = (rec.get("search") or {}).get("screen") or []
        if sr:
            lines.append("- candidate screening: " + "; ".join(
                f"{s.get('lens')}={s.get('score')}" for s in sr if isinstance(s, dict)
                and "lens" in s))
        winner = (rec.get("search") or {}).get("winner_lens")
        if winner:
            lines.append(f"- screening winner: {winner}"
                         + (" (self-authored search.md guidance)"
                            if str(winner).startswith("self_") else
                            " (no guidance — plain sample)"))
        # tool usage in this round's candidate episodes, from the intervener audit logs
        usage, errors = {}, {}
        for name in os.listdir(rdir):
            if not name.startswith("intervener_audit"):
                continue
            for ln in open(os.path.join(rdir, name), errors="replace"):
                try:
                    e = json.loads(ln)
                except Exception:
                    continue
                if e.get("kind") == "tool_call":
                    usage[e.get("tool")] = usage.get(e.get("tool"), 0) + 1
                elif e.get("kind") == "tool_result" and str(
                        e.get("out_head", "")).startswith("ERROR"):
                    errors[e.get("tool")] = errors.get(e.get("tool"), 0) + 1
        if usage:
            lines.append("- your tool usage this round: " + ", ".join(
                f"{k}x{n}" + (f" ({errors[k]} errors)" if errors.get(k) else "")
                for k, n in sorted(usage.items(), key=lambda kv: -kv[1])))
        if self._budget is not None and self.execute:
            pool = ("shared execution pool (probes + run_episode)"
                    if self._probes is not None else "run_episode budget")
            lines.append(f"- {pool}: {self._budget.used} used / "
                         f"{self._budget.remaining} left")
        # The report states facts only (call counts, errors, changes between rounds);
        # diagnosis is left to the optimizer.
        if self._round_tools_log:
            n_err = sum(1 for t in self._round_tools_log if t["error"])
            lines.append(f"- your self-written tools were called "
                         f"{len(self._round_tools_log)}x ({n_err} errors)")
        elif self.ws.teacher_code():
            lines.append("- your self-written tools were mounted; calls this round: 0")
        # Optimizer-harness events (code not loaded, rejected edits, health-check
        # rollbacks) from rounds rnd-1 and rnd. Round N's meta-episode runs after round
        # N's report is written and logs its events with round=N, so they first appear
        # in round N+1's report.
        if os.path.exists(self._events):
            evs = [json.loads(x) for x in open(self._events, errors="replace")]
            recent = [e for e in evs if e.get("round") in (rnd, rnd - 1)
                      and e["kind"] != "meta_committed"]
            for e in recent:
                when = " (last round's meta-episode)" if e.get("round") == rnd - 1 else ""
                lines.append(f"- EQUIPMENT EVENT{when}: {e['kind']}: "
                             f"{e.get('reason', '')[:200]}")
        return "\n".join(lines)
