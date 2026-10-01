"""Executable harness layer: the `code_hooks` edit space.

The optimizer writes `harness_code/hooks.py` through the same submit_proposal / fix_probe edit
channel as the markdown components, and it is versioned and reverted the same way. Unlike the
prompt components, it is never concatenated into the system prompt. It is loaded in a
restricted namespace and its hooks are called at fixed points of the executor loop. With
hooks_src=None (markdown-only configurations) the loop is unchanged.

Scope: in their published artifacts, the baselines that allow executable edits (AHE middleware
and tools, HarnessX processors and tools, Meta-Harness agent overrides) act through four
mechanisms: observation processing, context management, tool implementations, and cross-turn
state. All four are reachable through these hooks. The loop structure itself (Meta-Harness's
`_run_agent_loop`) is deliberately not exposed: one turn stays one LLM call in every
configuration, so all configurations run under the same compute budget.

Sandboxing (four layers; the threat model is reward hacking by the optimizer):
  1. Static audit (`audit_hooks_source`): imports limited to pure-computation modules; no
     I/O, no eval/exec/open, no dunder access, no dynamic attribute access. Runs when the
     edit is submitted and again at every load.
  2. Restricted load: the module is executed with a builtins table without open, eval and
     exec, whose __import__ only admits the allowed modules.
  3. Limited visibility: hooks see only the conversation (which contains the problem
     statement), the turn index and their own state. The task instance (FAIL_TO_PASS test
     ids, test patch, gold patch) never reaches them.
  4. Action channel: hook-defined tools act only through a BashCapability, whose single
     method runs a command in the same network-isolated, git-scrubbed task container as the
     model's own bash tool. Hook code cannot reach the host filesystem, the network or
     credentials, and reaches the LLM only through the metered `Hooks.llm` capability.

Failure handling: each call fails open (a hook that raises acts as the identity for that
call). A per-episode error log flows into the trajectory, the fix_probe report and the round
journal, so a broken hook is visible to the next round instead of silently doing nothing or
ending the episode with a NaN score.
"""
from __future__ import annotations

import ast
import json
import re
import time

# Static audit (AST allowlist)
ALLOWED_IMPORTS = {
    "re", "json", "math", "string", "textwrap", "difflib", "collections", "itertools",
    "functools", "heapq", "bisect", "statistics", "copy", "fnmatch", "shlex",
}
_BANNED_NAMES = {
    "eval", "exec", "compile", "open", "input", "__import__", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "breakpoint", "exit", "quit", "help",
    "memoryview", "classmethod", "staticmethod", "super", "type",
}
_ALLOWED_DUNDER_DEFS = {"__init__"}

MAX_HOOKS_SOURCE = 32_000        # chars


def audit_hooks_source(src: str, max_chars: int = MAX_HOOKS_SOURCE) -> list:
    """Static allowlist audit. Returns a list of violation strings (empty when clean).

    Violations are worded for the optimizer, which receives them verbatim. `max_chars`
    defaults to the cap for executor hooks; the optimizer's own hooks (meta_teacher) pass a
    larger cap."""
    v = []
    if len(src) > max_chars:
        v.append(f"hooks.py is {len(src)} chars > limit {max_chars}")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = (node.module if isinstance(node, ast.ImportFrom) else None)
            names = [mod] if mod else [a.name for a in node.names]
            for name in names:
                root = (name or "").split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    v.append(f"line {node.lineno}: import of '{root}' not allowed "
                             f"(allowed: {sorted(ALLOWED_IMPORTS)})")
        elif isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
            v.append(f"line {node.lineno}: use of '{node.id}' not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            v.append(f"line {node.lineno}: dunder attribute access "
                     f"'.{node.attr}' not allowed")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("__") and node.name not in _ALLOWED_DUNDER_DEFS:
                v.append(f"line {node.lineno}: defining '{node.name}' not allowed")
            if isinstance(node, ast.AsyncFunctionDef):
                v.append(f"line {node.lineno}: async not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            v.append(f"line {node.lineno}: global/nonlocal not allowed")
    return v


# Restricted load
_SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "bool", "bytes", "callable", "chr", "dict", "divmod",
    "enumerate", "filter", "float", "format", "frozenset", "hash", "hex", "int",
    "isinstance", "issubclass", "iter", "len", "list", "map", "max", "min", "next",
    "object", "oct", "ord", "pow", "print", "range", "repr", "reversed", "round",
    "set", "slice", "sorted", "str", "sum", "tuple", "zip",
    # exception types hooks may raise or catch
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError", "AttributeError",
    "RuntimeError", "StopIteration", "ZeroDivisionError", "ArithmeticError",
    "LookupError", "NotImplementedError",
]


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in ALLOWED_IMPORTS:
        raise ImportError(f"import of '{root}' is not allowed in harness hooks")
    return __import__(name, globals, locals, fromlist, level)


def _safe_builtins() -> dict:
    import builtins as _b
    table = {n: getattr(_b, n) for n in _SAFE_BUILTIN_NAMES}
    table["__import__"] = _safe_import
    # `class` statements need __build_class__ and __name__; without them `class Hooks(...)`
    # raises NameError under a custom builtins table.
    table["__build_class__"] = _b.__build_class__
    table["__name__"] = "harness_code.hooks"
    return table


class BaseHooks:
    """The hook surface. Every method defaults to the identity, so an empty subclass is a
    valid, inert harness. The optimizer-facing contract is HOOKS_API_DOC below."""

    # Declarative loop knobs, validated against LOOP_KNOBS. Illegal keys or values are
    # dropped one by one and logged. This mirrors AHE's configuration registry.
    loop: dict = {}

    def llm(self, prompt: str, max_tokens: int = 1024) -> str:
        """Metered model access, replaced per episode by HookedRuntime.attach_llm.

        Each call consumes one agent turn from the episode budget (turns + LLM calls
        <= max_turns). This default runs only when no capability is attached (test run
        or markdown-only configuration)."""
        return "ERROR: llm capability not attached (dry-run or no episode context)"

    def system_prompt(self, assembled: str) -> str:
        return assembled

    def before_llm(self, msgs: list, state: dict) -> list:
        return msgs

    def after_llm(self, content: list, state: dict) -> list:
        return content

    def before_tool(self, name: str, args: dict, state: dict):
        return args        # return the (possibly modified) args dict, or a str to block

    def after_tool(self, name: str, args: dict, obs: str, state: dict) -> str:
        return obs

    def extra_tools(self) -> list:
        return []

    def run_tool(self, name: str, args: dict, env, state: dict) -> str:
        return f"ERROR: unknown tool {name}"

    def on_turn_end(self, turn: int, state: dict):
        return None


def load_hooks_source(ws_dir: str) -> str | None:
    """Read harness_code/hooks.py from a workspace directory; None when absent or blank.

    Callers pass the source string around, not a live object, because every episode must
    instantiate its own Hooks: sweeps run episodes concurrently and a shared instance would
    race on `state`."""
    import os
    p = os.path.join(ws_dir, "harness_code", "hooks.py")
    if not os.path.isfile(p):
        return None
    src = open(p, errors="replace").read()
    return src if src.strip() else None


def load_hooks(src: str, max_chars: int = MAX_HOOKS_SOURCE):
    """Audit, execute under restricted builtins, and instantiate `Hooks`.

    Raises HookLoadError with a message the optimizer can act on (audit violations, an
    import-time exception, or a missing class)."""
    violations = audit_hooks_source(src, max_chars=max_chars)
    if violations:
        raise HookLoadError("audit failed:\n" + "\n".join(f"- {x}" for x in violations))
    ns = {"__builtins__": _safe_builtins(), "BaseHooks": BaseHooks}
    try:
        exec(compile(src, "harness_code/hooks.py", "exec"), ns)     # noqa: S102 — audited + restricted
    except Exception as e:
        raise HookLoadError(f"hooks.py failed to execute at import time: {e!r}")
    cls = ns.get("Hooks")
    if not isinstance(cls, type):
        raise HookLoadError("hooks.py must define a class named `Hooks` "
                            "(subclass BaseHooks; all methods optional)")
    if not issubclass(cls, BaseHooks):
        # A bare `class Hooks:` is a common mistake; its AttributeErrors on every hook
        # call are hard to read, so state the fix.
        raise HookLoadError("`Hooks` must subclass BaseHooks: write "
                            "`class Hooks(BaseHooks):` — BaseHooks is predefined in "
                            "the namespace, do not define or import it yourself")
    try:
        inst = cls()
    except Exception as e:
        raise HookLoadError(f"Hooks() constructor raised: {e!r}")
    return inst


class HookLoadError(Exception):
    pass


# Loop knobs
# AHE's configuration lever (its code_agent.yaml registry), enforced in code. Knobs are
# validated at the test run and validated again at episode start. max_turns may only shrink,
# so a harness cannot raise its own compute budget (HarnessX also keeps max_steps out of its
# configuration lever; AHE allows growth).
# Specs are (type, lo, hi) for ints and (type, max_len) for strings.
LOOP_KNOBS = {
    "max_turns":       ("int", 10, None),      # hi = the config's max_turns (may only shrink)
    "obs_cap":         ("int", 1024, 12288),   # observation truncation (chars)
    "keep_full_turns": ("int", 4, 32),         # sliding-window turns kept at full length
    "spam_streak":     ("int", 3, 10),         # consecutive unparseable turns before abort
    "nudge_no_tool":   ("str", 400),           # nudge when the model calls no tool
    "nudge_bad_markup": ("str", 400),          # nudge on tool-call markup emitted as text
}


def validate_loop_config(loop, config_max_turns: int = 100) -> tuple:
    """Return (clean_config, violations).

    Validation is per key: an illegal key, type or value is dropped and reported, and the
    remaining knobs still apply. Strings are stripped and length-capped."""
    clean, viol = {}, []
    if loop is None:
        return clean, viol
    if not isinstance(loop, dict):
        return clean, [f"loop must be a dict, got {type(loop).__name__}"]
    for k, v in loop.items():
        spec = LOOP_KNOBS.get(k)
        if spec is None:
            viol.append(f"loop['{k}']: unknown knob (valid: {sorted(LOOP_KNOBS)})")
            continue
        if spec[0] == "int":
            lo, hi = spec[1], spec[2]
            if hi is None:
                hi = int(config_max_turns)
            if not isinstance(v, int) or isinstance(v, bool):
                viol.append(f"loop['{k}']: must be an int, got {v!r}")
                continue
            if not (lo <= v <= hi):
                viol.append(f"loop['{k}']={v}: out of range [{lo}, {hi}]"
                            + (" (max_turns may only shrink)" if k == "max_turns" else ""))
                continue
            clean[k] = v
        else:
            if not isinstance(v, str) or not v.strip():
                viol.append(f"loop['{k}']: must be a non-empty str")
                continue
            clean[k] = v.strip()[: spec[1]]
    return clean, viol


# Capability objects
class LLMCapability:
    """Metered model access for hooks (the object behind Hooks.llm).

    Supports in-harness LLM calls, which the executable baselines also use (AHE middleware,
    HarnessX sub-harness providers, Meta-Harness summarizers), and, combined with
    extra_tools, subagent-style tools. Every call consumes one unit of the same budget as
    an agent turn: the executor loop counts turns plus hook LLM calls against max_turns.
    Calls use the episode's own executor model; the model and its parameters are not
    selectable."""

    def __init__(self, invoke_fn, max_calls: int):
        self._invoke = invoke_fn         # (prompt, max_tokens) -> str
        self._max = max(0, int(max_calls))
        self.calls = 0                   # calls made this episode (read by the loop)

    def __call__(self, prompt: str, max_tokens: int = 1024) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            return "ERROR: empty prompt"
        if self.calls >= self._max:
            return (f"ERROR: hook LLM budget exhausted ({self._max} calls/episode — "
                    "each call costs one agent turn)")
        self.calls += 1
        try:
            return self._invoke(prompt[:60_000], min(max(1, int(max_tokens)), 4096))
        except Exception as e:
            return f"ERROR: llm call failed: {e!r}"


class BashCapability:
    """The only action channel for hook-defined tools: run a command in the episode's task
    container, with the same network isolation, git scrub and CPU cap as the model's own
    bash tool. Commands are recorded and written to the trajectory as ```bash blocks, so
    replay, perturbation and minimization also cover them."""

    def __init__(self, execute_fn, deadline: float, max_cmds: int = 16):
        self._execute = execute_fn
        self._deadline = deadline
        self._max = max_cmds
        self.commands: list = []          # every command run through this capability
        self._calls_this_tool = 0

    def new_tool_call(self):
        self._calls_this_tool = 0

    def bash(self, cmd: str) -> str:
        """Run one shell command in the task container; returns combined output."""
        if not isinstance(cmd, str) or not cmd.strip():
            return "ERROR: empty command"
        if time.monotonic() > self._deadline:
            return "ERROR: episode wall-clock exhausted"
        self._calls_this_tool += 1
        if self._calls_this_tool > self._max:
            return (f"ERROR: per-tool-call command limit ({self._max}) reached — "
                    "return your result now")
        self.commands.append(cmd)
        try:
            return self._execute(cmd)
        except Exception as e:
            return f"ERROR: command failed to execute: {e!r}"


# Fail-open runtime
_MAX_ERRORS_KEPT = 20
_MAX_EXTRA_TOOLS = 4
_MAX_NOTE_CHARS = 1200
_MAX_OBS_TO_HOOK = 65_536        # bounds hook compute on very large outputs


class HookedRuntime:
    """Per-episode wrapper: fail-open calls, return-type validation, call counts and an
    error log. One instance per episode, built inside run_task from the source string."""

    def __init__(self, src: str, max_chars: int = MAX_HOOKS_SOURCE):
        self.hooks = load_hooks(src, max_chars)   # raises HookLoadError; the caller decides
        self.state: dict = {}
        self.errors: list = []                    # [{hook, turn, error}]
        self.counts: dict = {}                    # {hook: calls}
        self.changed: dict = {}                   # {hook: calls that returned a change}
        self._tools = None                        # validated extra tool specs (lazy)
        self.llm_cap: LLMCapability | None = None # attached by the loop (attach_llm)

    # Episode capabilities
    def loop_config(self, config_max_turns: int = 100) -> dict:
        """Validated loop knobs from the Hooks class (schema: LOOP_KNOBS). Violations go to
        the error log (visible in fix_probe and hooks_stats) and the bad keys are dropped,
        so a mistyped knob falls back to its default instead of ending the episode."""
        clean, viol = validate_loop_config(getattr(self.hooks, "loop", None),
                                           config_max_turns)
        for v in viol:
            if len(self.errors) < _MAX_ERRORS_KEPT:
                self.errors.append({"hook": "loop", "turn": -1, "error": v[:300]})
        if clean:
            self.changed["loop"] = len(clean)
        return clean

    def attach_llm(self, invoke_fn, max_calls: int) -> None:
        """Attach the metered LLM capability to the hooks instance. The loop reads
        llm_cap.calls each turn and counts them against the turn budget."""
        self.llm_cap = LLMCapability(invoke_fn, max_calls)
        try:
            self.hooks.llm = self.llm_cap
        except Exception:
            pass                                   # not logged; llm_cap.calls stays 0

    # Internals
    def _guard(self, hook: str, turn: int, fn, fallback, validate=None):
        self.counts[hook] = self.counts.get(hook, 0) + 1
        try:
            out = fn()
        except Exception as e:
            if len(self.errors) < _MAX_ERRORS_KEPT:
                self.errors.append({"hook": hook, "turn": turn, "error": repr(e)[:300]})
            return fallback
        if validate is not None:
            ok, out_or_msg = validate(out)
            if not ok:
                if len(self.errors) < _MAX_ERRORS_KEPT:
                    self.errors.append({"hook": hook, "turn": turn,
                                        "error": f"bad return: {out_or_msg}"})
                return fallback
            out = out_or_msg
        return out

    # Hook surfaces
    def system_prompt(self, assembled: str) -> str:
        def _val(out):
            if not isinstance(out, str) or not out.strip():
                return False, "system_prompt must return a non-empty str"
            return True, out
        out = self._guard("system_prompt", -1,
                          lambda: self.hooks.system_prompt(assembled), assembled, _val)
        if out != assembled:
            self.changed["system_prompt"] = self.changed.get("system_prompt", 0) + 1
        return out

    def before_llm(self, msgs: list, turn: int) -> list:
        def _val(out):
            if not isinstance(out, list) or not out:
                return False, "before_llm must return a non-empty list of messages"
            for m in out:
                if not (isinstance(m, dict) and m.get("role") in ("user", "assistant")
                        and "content" in m):
                    return False, "each message needs role in {user,assistant} and content"
            if out[-1].get("role") != "user":
                return False, "the last message must stay a user message"
            return True, out
        out = self._guard("before_llm", turn,
                          lambda: self.hooks.before_llm(msgs, self.state), msgs, _val)
        if out is not msgs:
            self.changed["before_llm"] = self.changed.get("before_llm", 0) + 1
        return out

    def after_llm(self, content: list, turn: int) -> list:
        """The model's raw reply blocks, before the loop reads them.

        This is the parse-rescue surface (Meta-Harness `_parse_tool_calls`, HarnessX
        `on_after_model`). Typical use: a weak executor emits tool-call markup as text
        (DSML tags, XML); a hook can parse it and return a real tool_use block, so the turn
        acts instead of being wasted. Malformed tool calls are a common mechanical failure
        of weak executors."""
        # The Messages API rejects an assistant message with any block after a tool_use
        # ("tool_use ids were found without tool_result blocks immediately after"). One
        # such reply breaks the message history, every later call fails, and the episode
        # scores NaN. A note appended after the model's tool_use is harmless in intent, so
        # the blocks are reordered instead of rejected (stable partition: other blocks keep
        # their order, tool_use blocks move to the end) and the fix is logged for the
        # optimizer.
        def _reorder(blocks):
            first_tu = next((i for i, b in enumerate(blocks)
                             if b.get("type") == "tool_use"), None)
            if first_tu is None or all(b.get("type") == "tool_use"
                                       for b in blocks[first_tu:]):
                return blocks, False
            if len(self.errors) < _MAX_ERRORS_KEPT:
                self.errors.append({
                    "hook": "after_llm", "turn": turn,
                    "error": ("block(s) after a tool_use reordered to before it — the "
                              "Messages API rejects any block after tool_use in an "
                              "assistant message")})
            return ([b for b in blocks if b.get("type") != "tool_use"]
                    + [b for b in blocks if b.get("type") == "tool_use"]), True
        def _val(out):
            if out is content:
                # identity return: nothing to re-validate, but a hook may have appended
                # in place (content.append(note); return content), so the block order is
                # still checked
                out, _ = _reorder(out)
                return True, out
            if not isinstance(out, list) or not out:
                return False, "after_llm must return a non-empty list of content blocks"
            fixed, seen_ids = [], set()
            for i, b in enumerate(out[:12]):
                if not isinstance(b, dict):
                    return False, "each content block must be a dict"
                t = b.get("type")
                if t == "text":
                    fixed.append({"type": "text", "text": str(b.get("text", ""))})
                elif t == "tool_use":
                    if not b.get("name"):
                        return False, "tool_use blocks need a name"
                    bid = str(b.get("id") or f"hk_{turn}_{i}")
                    if bid in seen_ids:
                        bid = f"hk_{turn}_{i}"
                    seen_ids.add(bid)
                    fixed.append({"type": "tool_use", "id": bid,
                                  "name": re.sub(r"[^a-zA-Z0-9_-]", "_", str(b["name"]))[:64],
                                  "input": b.get("input") if isinstance(b.get("input"), dict)
                                  else {}})
                else:
                    return False, f"unknown block type {t!r} (want text|tool_use)"
            fixed, _ = _reorder(fixed)
            return True, fixed
        out = self._guard("after_llm", turn,
                          lambda: self.hooks.after_llm(content, self.state), content, _val)
        if out is not content:
            self.changed["after_llm"] = self.changed.get("after_llm", 0) + 1
        return out

    def before_tool(self, name: str, args: dict, turn: int) -> tuple:
        """Return (args, block_reason, synthetic).

        block_reason=None: execute with the (possibly rewritten) args. A str: do not
        execute. When blocking, the hook may return a tuple (reason, synthetic_result); the
        synthetic string is given to the model as the tool observation instead of the
        '[harness] command blocked' notice (as in HarnessX's synthetic_result), e.g. to
        answer a cached query without running a container command. This hook enforces rules
        in code (a markdown rule such as 'never rm -rf' is only advice; a before_tool guard
        blocks the command) and is where commands are rewritten (adding timeouts,
        normalizing paths)."""
        def _val(out):
            if isinstance(out, str):
                return True, out                       # block, with reason
            if isinstance(out, tuple):
                if (len(out) == 2 and isinstance(out[0], str)
                        and isinstance(out[1], str)):
                    return True, out                   # block, with synthetic result
                return False, ("a tuple return must be (reason: str, "
                               "synthetic_result: str)")
            if isinstance(out, dict):
                if "command" in out and not isinstance(out["command"], str):
                    return False, "args['command'] must be a str"
                return True, out
            return False, ("before_tool must return the args dict, a str to block, "
                           "or (reason, synthetic_result) to block with a synthetic "
                           "observation")
        out = self._guard("before_tool", turn,
                          lambda: self.hooks.before_tool(name, dict(args or {}), self.state),
                          args, _val)
        if isinstance(out, tuple):
            self.changed["before_tool"] = self.changed.get("before_tool", 0) + 1
            return args, out[0][:500], out[1][:_MAX_OBS_TO_HOOK]
        if isinstance(out, str):
            self.changed["before_tool"] = self.changed.get("before_tool", 0) + 1
            return args, out[:500], None
        if out != args:
            self.changed["before_tool"] = self.changed.get("before_tool", 0) + 1
        return out, None, None

    def after_tool(self, name: str, args: dict, obs: str, turn: int) -> str:
        raw = obs if len(obs) <= _MAX_OBS_TO_HOOK else obs[:_MAX_OBS_TO_HOOK]
        def _val(out):
            if not isinstance(out, str):
                return False, "after_tool must return a str"
            return True, out
        out = self._guard("after_tool", turn,
                          lambda: self.hooks.after_tool(name, args, raw, self.state),
                          obs, _val)
        if out != raw and out != obs:
            self.changed["after_tool"] = self.changed.get("after_tool", 0) + 1
        return out

    def extra_tool_specs(self) -> list:
        """Validated, cached list of {name, description, input_schema}. Names are sanitized
        to the API's tool-name charset, the count is capped, and bash/submit are never
        shadowed."""
        if self._tools is not None:
            return self._tools
        def _val(out):
            if not isinstance(out, list):
                return False, "extra_tools must return a list"
            specs, seen = [], set()
            for t in out[:_MAX_EXTRA_TOOLS]:
                if not (isinstance(t, dict) and t.get("name") and t.get("description")):
                    continue
                nm = re.sub(r"[^a-zA-Z0-9_-]", "_", str(t["name"]))[:64]
                if nm in ("bash", "submit", "submit_proposal") or nm in seen:
                    continue
                seen.add(nm)
                schema = t.get("input_schema")
                if not isinstance(schema, dict):
                    schema = {"type": "object", "properties": {}, "required": []}
                specs.append({"name": nm, "description": str(t["description"])[:1000],
                              "input_schema": schema})
            return True, specs
        self._tools = self._guard("extra_tools", -1,
                                  lambda: self.hooks.extra_tools(), [], _val)
        return self._tools

    def run_tool(self, name: str, args: dict, cap: BashCapability, turn: int) -> str:
        cap.new_tool_call()
        def _val(out):
            if not isinstance(out, str):
                return False, "run_tool must return a str"
            return True, out
        return self._guard("run_tool", turn,
                           lambda: self.hooks.run_tool(name, args or {}, cap, self.state),
                           f"ERROR: tool {name} raised", _val)

    def on_turn_end(self, turn: int) -> str | None:
        def _val(out):
            if out is None:
                return True, None
            if not isinstance(out, str):
                return False, "on_turn_end must return str or None"
            return True, out[:_MAX_NOTE_CHARS]
        out = self._guard("on_turn_end", turn,
                          lambda: self.hooks.on_turn_end(turn, self.state), None, _val)
        if out:
            self.changed["on_turn_end"] = self.changed.get("on_turn_end", 0) + 1
        return out

    # Summary
    def summary(self) -> dict:
        out = {"counts": self.counts, "changed": self.changed,
               "n_errors": len(self.errors), "errors": self.errors[:6]}
        if self.llm_cap is not None and self.llm_cap.calls:
            out["llm_calls"] = self.llm_cap.calls
        return out


# Summary record in the transcript
_SUMMARY_PREFIX = "HARNESS_CODE_SUMMARY: "


def summary_transcript_entry(rt: HookedRuntime) -> dict:
    """Transcript record, appended at episode end, of what the code layer did.

    It is saved in the trajectory files, read by the next round's optimizer, and parsed back
    by fix_probe for its report."""
    return {"role": "system", "content": _SUMMARY_PREFIX + json.dumps(rt.summary())}


def extract_hooks_summary(transcript: list) -> dict | None:
    for entry in reversed(transcript or []):
        c = entry.get("content", "") if isinstance(entry, dict) else ""
        if isinstance(c, str) and c.startswith(_SUMMARY_PREFIX):
            try:
                return json.loads(c[len(_SUMMARY_PREFIX):])
            except Exception:
                return None
    return None


# Optimizer-facing documentation (prompt text)
HOOKS_API_DOC = """## harness_code/hooks.py — the executable harness layer
Beyond the markdown components you may write PYTHON that runs inside the executor's agent
loop. Submit it as a normal edit with path `harness_code/hooks.py`. Define:

```python
class Hooks(BaseHooks):        # BaseHooks is predefined; all methods optional
    loop = {}                                      # OPTIONAL loop knobs (see table below)
    def system_prompt(self, assembled):            # str -> str
        return assembled                            # rewrite/extend the final prompt
    def before_llm(self, msgs, state):             # transform the message view each turn
        return msgs                                 # e.g. compress old observations smartly
    def after_llm(self, content, state):           # the model's RAW reply blocks — the
        return content                              # PARSE-RESCUE surface: if the model
                                                    # emits tool-call markup as plain text
                                                    # (XML/DSML/json), parse it and return
                                                    # a real {"type":"tool_use","name":...,
                                                    # "input":{...}} block; a rescued turn
                                                    # acts instead of dying. Keep text
                                                    # blocks BEFORE tool_use blocks: the
                                                    # API rejects an assistant message with
                                                    # anything after a tool_use (a text
                                                    # appended after one is auto-moved
                                                    # before it and ledgered)
    def before_tool(self, name, args, state):      # rewrite a command before it runs
        return args                                 # (auto-add timeouts, normalize paths);
                                                    # return a STR to BLOCK it — mechanical
                                                    # commission enforcement ("never rm -rf"
                                                    # as a guard, not a plea); or return
                                                    # (reason, synthetic_result) to block
                                                    # AND answer with a synthetic
                                                    # observation (cached/idempotent
                                                    # queries without a container command)
    def after_tool(self, name, args, obs, state):  # rewrite RAW tool output before the
        return obs                                  # model sees it (annotate errors, dedup
                                                    # tracebacks, extract test failures...)
    def extra_tools(self):                         # add tools: [{name, description,
        return []                                   #   input_schema}]
    def run_tool(self, name, args, env, state):    # implement them; env.bash(cmd) runs a
        return "..."                                # command in the task container
    def on_turn_end(self, turn, state):            # str|None — a note injected into the
        return None                                 # next user turn (cross-turn reminders)
```

### self.llm(prompt, max_tokens=1024) — metered model access (available in ANY hook)
One call = one agent turn deducted from the episode's turn budget (turns + llm calls <=
max_turns) — intelligence is budgeted, not free. Uses the episode's own executor model;
model/temperature are not selectable. Enables smart middleware the baselines ship:
observation summarizers (compress a 60KB log to the 5 lines that matter), pre-submit
self-checks, error-message explainers, and — composed with extra_tools/run_tool —
subagent-shaped tools (an extra tool whose run_tool implementation calls self.llm with a
focused prompt plus env.bash output). Spend it where a turn of thinking is worth a turn
of acting; a hook that calls llm every turn halves the executor's action budget.

### loop knobs (class attribute `loop = {...}` — validated, illegal keys dropped+ledgered)
| knob | default | range | use |
| max_turns | config (100) | 10..config, ONLY SHRINK | early-stop economy; growth = bought compute, rejected |
| obs_cap | 4096 | 1024..12288 | observation truncation length |
| keep_full_turns | 12 | 4..32 | sliding-window turns kept full |
| spam_streak | 6 | 3..10 | unparseable-turn abort threshold |
| nudge_no_tool | (default text) | str<=400 | nudge when the model calls no tool |
| nudge_bad_markup | (default text) | str<=400 | nudge on tool-markup-as-text turns — pair with after_llm rescue |

Rules (enforced by a static audit — violations bounce your proposal back):
- imports limited to: re, json, math, string, textwrap, difflib, collections, itertools,
  functools, heapq, bisect, statistics, copy, fnmatch, shlex. No file/network/process
  access — `env.bash()` inside run_tool acts in the sandboxed task container, and
  `self.llm()` is the only (metered) model access.
- `state` is a per-episode dict you own (starts empty each task).
- A raising hook silently degrades to identity FOR THAT CALL and is logged; the per-episode
  error ledger appears at the end of every trajectory (HARNESS_CODE_SUMMARY) — check it.
- Budget note: a run_tool call may run several bash commands but the episode's wall-clock
  and container CPU caps still apply; llm calls share the turn budget (see above).

Two capabilities people miss:
- FILE ASSETS: run_tool + env.bash can WRITE files into the task container (helper
  scripts, checklists: `env.bash("cat > /tmp/helper.py <<'EOF' ...")`) — a tool can
  install its own machinery on first use, then reuse it all episode.
- PERSISTENT CONTEXT SURGERY: before_llm sees the full message list EVERY turn and state
  persists across turns — applying the same transform each turn IS durable history
  rewriting (summarize-and-replace old observations, pin key facts to the front).

When to use code instead of markdown: mechanical failure modes that instructions cannot
reliably fix — malformed model output, noisy/oversized observations, missing structure in
error messages, repeated manual workflows worth packaging as a tool. If the same failure
class survived a markdown fix, that is the signal to fix it here instead."""

# Contract-only version for the self_teacher configurations: the same hook surface,
# metering, knob table and audit rules, without the usage guidance (the parse-rescue
# recipe, middleware examples, the "capabilities people miss" tips, and when to prefer
# code over markdown). The self-teaching optimizer works out what each hook is for.
HOOKS_API_DOC_CONTRACT = """## harness_code/hooks.py — the executable harness layer
Beyond the markdown components you may write PYTHON that runs inside the executor's agent
loop. Submit it as a normal edit with path `harness_code/hooks.py`. Define:

```python
class Hooks(BaseHooks):        # BaseHooks is predefined; all methods optional
    loop = {}                                      # OPTIONAL loop knobs (see table below)
    def system_prompt(self, assembled):            # str -> str: the assembled system
        return assembled                            # prompt before the episode starts
    def before_llm(self, msgs, state):             # list -> list: the message view sent
        return msgs                                 # to the model each turn
    def after_llm(self, content, state):           # list -> list: the model's raw reply
        return content                              # blocks before the loop reads them;
                                                    # text blocks must come BEFORE tool_use
                                                    # blocks (API constraint; violations
                                                    # are auto-reordered and ledgered)
    def before_tool(self, name, args, state):      # dict -> dict: tool args before the
        return args                                 # call runs; return a STR to block the
                                                    # call, or (reason, synthetic_result)
                                                    # to block and supply the observation
    def after_tool(self, name, args, obs, state):  # str -> str: raw tool output before
        return obs                                  # the model sees it
    def extra_tools(self):                         # declare new tools: [{name,
        return []                                   #   description, input_schema}]
    def run_tool(self, name, args, env, state):    # implement them; env.bash(cmd) runs a
        return "..."                                # command in the task container
    def on_turn_end(self, turn, state):            # str|None — a note injected into the
        return None                                 # next user turn
```

### self.llm(prompt, max_tokens=1024) — metered model access (available in ANY hook)
One call = one agent turn deducted from the episode's turn budget (turns + llm calls <=
max_turns). Uses the episode's own executor model; model/temperature are not selectable.

### loop knobs (class attribute `loop = {...}` — validated, illegal keys dropped+ledgered)
| knob | default | range |
| max_turns | config (100) | 10..config, ONLY SHRINK |
| obs_cap | 4096 | 1024..12288 |
| keep_full_turns | 12 | 4..32 |
| spam_streak | 6 | 3..10 |
| nudge_no_tool | (default text) | str<=400 |
| nudge_bad_markup | (default text) | str<=400 |

Rules (enforced by a static audit — violations bounce your proposal back):
- imports limited to: re, json, math, string, textwrap, difflib, collections, itertools,
  functools, heapq, bisect, statistics, copy, fnmatch, shlex. No file/network/process
  access — `env.bash()` inside run_tool acts in the sandboxed task container, and
  `self.llm()` is the only (metered) model access.
- `state` is a per-episode dict you own (starts empty each task).
- A raising hook silently degrades to identity FOR THAT CALL and is logged; the per-episode
  error ledger appears at the end of every trajectory (HARNESS_CODE_SUMMARY).
- Budget note: a run_tool call may run several bash commands but the episode's wall-clock
  and container CPU caps still apply; llm calls share the turn budget (see above)."""
