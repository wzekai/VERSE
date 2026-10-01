"""harness_code/hooks.py - mechanical guardrails for the frozen executor (round 3).

Diagnosed on the sweep run UNDER the round-2 hooks (110 episodes / 74 failures), so every
guard below is known to execute. This round repairs two of my OWN guards that were silently
damaging the executor's test loop - the failures they cause look exactly like model failures.

  BUG 1 (harness-inflicted, the big one) - the verified-interpreter rewrite MANGLED every
      `python -m pytest`. before_tool used
          re.sub(r"(?<![\w./-])pytest(?=[\s|;&)]|$)", "'<interp>' -m pytest", cmd)
      and the bare word `pytest` INSIDE `-m pytest` (and inside quoted args and here-doc
      bodies) matches that pattern, so
          python -m pytest tests/x.py
      became
          '<interp>' -m '<interp>' -m pytest tests/x.py
      which answers `No module named /opt/conda/envs/testbed/bin/python`. Measured: 334
      mangled commands in 85/110 episodes, 304 of them returning a broken observation, plus
      68 corrupted `python -c "import <interp> -m pytest"` (26 eps) and 48 corrupted quoted
      grep / `command -v` args. The mangled text lands in the message history and the models
      COPY it (brightway-lca__brightway2-io-294: 8 mangled runs in a row, steps 22-29;
      aws-cloudformation__cfn-python-lint-4368: every test run mangled). Now: substitute only
      at a SEGMENT HEAD, never on `-m <module>`, never on a token that already carries a
      path, never after a `<<` here-doc - plus _repair_mangle(), which collapses the mangled
      forms back before the command runs so copies of the corruption heal instead of re-failing.
  BUG 2 (harness-inflicted) - here-doc edits were invisible. The dominant edit style is
      python - <<'EOF' ... open(path,'w').write(s) ... EOF
      which matched neither _ANY_WRITE_RE (no redirect, no sed -i) nor _OPEN_W_RE (the path is
      a variable). 28/110 episodes therefore ran with edits=0 while editing: the coaching lied,
      the verify-gate never fired for here-doc editors, and the DUPLICATE CACHE WAS NOT
      INVALIDATED by the edit, so a test re-run after an edit could be served stale output as
      if current. Fixed with _WRITE_BODY_RE in _mutates() and _src_edit().
  BUG 3 - ANSI SGR sequences survive into observations and break `^=+ ... =+$`: only 24 of 70
      pytest digests recovered the summary line. after_tool strips them before parsing.

  KEPT (measured working in the head): offline install block; duplicate-command replay; pytest
  log digest; node-id quoting hint; the `@@PATCHSTAT` patch ground-truth probe; and the bounded
  dead-turn resurrection, which this round fired in 24 episodes and took empty-diff failures
  from 12 to 4 and no-action episode endings from 20 eps to 7.
"""
import re
import json

_TEST_RE = re.compile(r"(\bpytest\b|\bpy\.test\b|-m\s+pytest\b|-m\s+unittest\b|\btox\b)", re.I)
_READ_START_RE = re.compile(
    r"^\s*(?:cd\s+\S+\s*(?:&&|;)\s*)*(?:echo\s+\S+\s*&&\s*)*"
    r"(cat|sed|grep|egrep|rg|ls|find|head|tail|wc|awk|nl|stat|file|tree|which|diff|"
    r"realpath|basename|dirname|sort|uniq|jq|git|test|\[)\b")
_PIP_RE = re.compile(r"^\s*(?:cd\s+\S+\s*(?:&&|;)\s*)*(?:sudo\s+)?(?:python[0-9.]*\s+-m\s+)?"
                     r"(pip[23]?|conda)\s+(install|download|create)\b(.*)$", re.S)

# stderr/scratch redirections are NOT edits: strip them before looking for a redirect.
_NOISE_RE = re.compile(r"\d*>&+\d+|\d*>>?\s*/dev/\w+|\d*>>?\s*/tmp/\S+|\d*>>?\s*/var/tmp/\S+",
                       re.I)
_ANY_WRITE_RE = re.compile(r"(sed\s+-i|perl\s+-\w*i\b|git apply|git checkout\s+--|git restore\b|"
                           r"\btee\b|\bmv\b|\bcp\b|\brm\b|\bmkdir\b|\btouch\b|\btruncate\b|"
                           r"\bpatch\b|>>?\s*\S)", re.I)
# a script that writes through the interpreter (here-doc or -c): the dominant edit style.
_WRITE_BODY_RE = re.compile(r"(\.write\(|\.writelines\(|write_text\(|write_bytes\(|"
                            r"open\([^()\n]{0,60}?['\"][wax]|shutil\.(?:copy|copyfile|move)|"
                            r"os\.replace\(|os\.rename\()", re.I)
_QPATH_RE = re.compile(r"['\"]([\w./\-]+\.[A-Za-z]{1,5})['\"]")
_TARGET_RE = re.compile(">>?\s*['\"]?([^\s'\"|&;<>()]+)")
_SRC_EXT_RE = re.compile(r"\.(?:py|pyx|pxd|pyi|js|ts|jsx|java|go|rs|c|h|cc|cpp|hpp|f90|ftl|"
                         r"toml|cfg|ini|txt|md|rst|json|ya?ml|lock)$", re.I)
_OPEN_W_RE = re.compile(r"open\(\s*['\"][^'\"]+['\"]\s*,\s*['\"][wax]")

_MOD_ERR_RE = re.compile(r"(?:ModuleNotFoundError|ImportError):\s*"
                         r"(?:No module named|cannot import name)\s*['\"]?([\w.]+)")
_NET_ERR_RE = re.compile(r"(Max retries exceeded|NameResolutionError|Temporary failure in "
                         r"name resolution|Failed to resolve|NewConnectionError|"
                         r"Connection aborted|ConnectionError|ECONNREFUSED)", re.I)
_GOODPY_RE = re.compile(r"GOODPY\s+(/\S+)")
_PATCHSTAT_RE = re.compile(r"@@PATCHSTAT\s+(\d+)\s+lines")
_NODEID_RE = re.compile(r"ERROR: not found: (\S+)")
_ERR_RE = re.compile(r"(Traceback \(most recent call last\)|ModuleNotFoundError|ImportError|"
                     r"command not found|ERROR: not found|No such file or directory|"
                     r"error: unrecognized arguments|usage: pytest|No module named /|"
                     r"Error while finding module specification)", re.I)
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_SUMMARY_RE = re.compile(r"^=+[^=\n]*(?:passed|failed|error|no tests ran)[^=\n]*=+\s*$", re.M)
_SUM_ANY_RE = re.compile(r"[^\n]*(?:\d+ (?:passed|failed)\b|no tests ran|\bran \d+ tests)[^\n]*")
_FAIL_LINE_RE = re.compile(r"^(?:FAILED|ERROR)\b.*$", re.M)
_FAILCOUNT_RE = re.compile(r"\b\d+ (?:failed|errors?)\b")
_PASSED_RE = re.compile(r"\d+ (?:passed|ok)\b|^OK\b|ran \d+ tests", re.M)

_HARNESS_TAG = "# harness-verify"
_REVIVE_CMD = ("cd /testbed && git status --porcelain | head -20 && git diff --stat | tail -4 "
               "&& echo \"@@PATCHSTAT $(git diff | wc -l) lines\" " + _HARNESS_TAG)
_TRUTH_HEAD = 'echo "@@PATCHSTAT $(cd /testbed && git diff | wc -l) lines"; '
_EDIT_TURNS = (12, 16, 20, 26, 35, 50, 70, 85)
_PROBE_HEAD = ('echo "== env probe =="; '
               'for p in /opt/conda/envs/testbed/bin/python /testbed/.venv/bin/python '
               '/usr/local/bin/python "$(command -v python3)"; do '
               '[ -x "$p" ] || continue; "$p" -c "import %s" >/dev/null 2>&1 '
               '&& echo "GOODPY $p"; done; echo "== end probe =="')
_PROBE_MARK = 'echo "== env probe =="'

# ---------------------------------------------------------------- interpreter rewrite
_SEG_RE = re.compile(r"(&&|\|\||;|\||\n)")
_PREFIX_RE = re.compile(r"^\s*(?:(?:sudo|time|nohup|command)\s+|timeout\s+\S+\s+"
                        r"|env\s+(?:\S+\s+)*)*", re.I)
_HEAD_RE = re.compile(r"(python[\d.]*|pytest)(?=[\s'\"|;&)]|$)")
_MID = r"(?:'[^']*python[\d.]*'|\"[^\"]*python[\d.]*\"|[\w./-]*python[\d.]*)"
_MANGLE_A = re.compile(r"-m\s+" + _MID + r"\s+-m\s+(pytest|unittest)")
_MANGLE_B = re.compile(r"(?<![\w./-])import\s+" + _MID + r"\s+-m\s+(pytest|unittest)")
_MANGLE_C = re.compile(r"command -v '((/[\w./-]*)?python[\d.]*)'")


def _scrub(cmd):
    return _NOISE_RE.sub(" ", cmd)


def _norm(cmd):
    return " ".join(cmd.split())


def _unmangle(cmd):
    """Collapse the corrupted interpreter forms back to a runnable command. The mangled text
    is in the message history, so models re-emit it verbatim; heal it instead of re-running it."""
    for _ in range(3):
        new = _MANGLE_A.sub(r"-m \1", cmd)
        new = _MANGLE_B.sub(r"import \1", new)
        new = _MANGLE_C.sub(r"command -v \1", new)
        if new == cmd:
            break
        cmd = new
    return cmd


def _fix_seg(seg, gp):
    m = _PREFIX_RE.match(seg) or _PREFIX_RE.match("")
    off = m.end()
    hm = _HEAD_RE.match(seg, off)
    if not hm:
        return seg                      # not an interpreter command: never touch it
    tok = hm.group(1)
    rep = ("'%s'" % gp) if tok.startswith("python") else ("'%s' -m pytest" % gp)
    return seg[:off] + rep + seg[hm.end():]


def _interp_fix(cmd, gp):
    """Substitute a BARE python/pytest at a command position with the verified interpreter.
    Structurally incapable of mangling: only the FIRST token of a `&&`/`;`/`|` segment is a
    candidate (so `-m pytest` is never a target), a token that already carries a path is left
    alone, and everything after a `<<` here-doc marker is copied verbatim."""
    cut = cmd.find("<<")
    head, body = (cmd[:cut], cmd[cut:]) if cut >= 0 else (cmd, "")
    parts = _SEG_RE.split(head)
    out = []
    for i, p in enumerate(parts):
        out.append(p if i % 2 else _fix_seg(p, gp))
    return "".join(out) + body


def _mutates(cmd):
    """Anything that can change what a later command would see (cache invalidation)."""
    c = _scrub(cmd)
    return bool(_ANY_WRITE_RE.search(c)) or bool(_WRITE_BODY_RE.search(cmd))


def _is_scratch(t):
    return t.startswith(("/tmp", "/dev", "/var/tmp", "/proc"))


def _src_edit(cmd):
    """True when the command plausibly edits a repo source file - i.e. builds THE PATCH."""
    c = _scrub(cmd)
    if re.search(r"sed\s+-i|perl\s+-\w*i\b|git apply|git checkout\s+--|git restore\b", c, re.I):
        return True
    if _WRITE_BODY_RE.search(c):
        paths = [t for t in _QPATH_RE.findall(c) if not _is_scratch(t)]
        if paths:
            return True
        if not any(t in c for t in ("/tmp", "/dev", "/var/tmp")):
            return True
    scratch_only = True
    hit = False
    for t in _TARGET_RE.findall(c):
        if _is_scratch(t):
            continue
        scratch_only = False
        if _SRC_EXT_RE.search(t) or "/" in t:
            hit = True
    if hit:
        return True
    if _OPEN_W_RE.search(c) or (".write(" in c and not scratch_only):
        return True
    if (".write(" in c or _OPEN_W_RE.search(c)) and "/" not in c:
        return True
    return False


def _tu(cmd):
    return {"type": "tool_use", "name": "bash", "input": {"command": cmd[:4000]}}


def _text_cmd(text):
    """Pull a shell command out of a reply that wrote the tool call as TEXT."""
    cmd = None
    m = re.search(r"```(?:bash|sh|shell)\n(.*?)```", text, re.S)
    if m:
        cmd = m.group(1)
    if not cmd:
        m = re.search(r"<(bash|command|tool_call|antml:function_calls)>(.*?)</\1>", text, re.S)
        if m:
            cmd = re.sub(r"<[^>]+>", "", m.group(2))
    if not cmd:
        m = re.search(r'\{[^{}]*"(?:command|name)"[^{}]*\}', text, re.S)
        if m:
            try:
                blob = json.loads(m.group(0))
            except Exception:
                blob = None
            if isinstance(blob, dict):
                cmd = blob.get("command") or ""
                if not cmd and isinstance(blob.get("input"), dict):
                    cmd = blob["input"].get("command", "")
    if not cmd:
        return None
    cmd = "\n".join(l for l in cmd.splitlines()
                    if l.strip() and not l.strip().startswith(("$", "#"))).strip()
    return cmd or None


class Hooks(BaseHooks):
    loop = {
        "obs_cap": 6144,
        "keep_full_turns": 14,
        "spam_streak": 8,
        "nudge_no_tool": ("Call a tool now: `bash` to edit the source (sed -i or a here-doc "
                          "python script) or to run the affected test file, or `submit` once "
                          "the diff is non-empty. A reply with no tool call ENDS the episode "
                          "and scores the current diff."),
        "nudge_bad_markup": ("That tool call did not parse. Emit a NATIVE tool call "
                             "(bash tool, input {\"command\": \"...\"}), never XML/DSML "
                             "markup in your text."),
    }

    # ------------------------------------------------------------ prompt
    def system_prompt(self, assembled):
        return assembled + (
            "\n\n## Operating rules (several are enforced mechanically by the harness)\n"
            "- LEAVE A PATCH. Grading is the `git diff` of /testbed: an empty diff scores "
            "zero however good the analysis was. Make a first real edit within ~12 commands, "
            "then refine it against tests. A reply with no tool call ENDS the episode.\n"
            "- ONE LOOP: explore (grep -n, sed -n 'A,Bp' ranges - not whole-file cats) -> edit "
            "the source -> run the AFFECTED test file -> fix -> `git diff` -> submit.\n"
            "- NEVER run the same command twice: the harness replays your earlier output "
            "instead of re-running it, so a repeat buys nothing.\n"
            "- Quote parametrised test node ids ('...::test_x[a-b]') or filter with `-k`; "
            "unbracketed/shell-split ids come back as `ERROR: not found`.\n"
            "- Run tests as `python -m pytest <file> -q` with ONE interpreter for the whole "
            "session (the repo env interpreter, often /opt/conda/envs/testbed/bin/python). If "
            "an import fails the harness prints `GOODPY <path>` - use that path from then on.\n"
            "- No network in this sandbox: never pip/conda install; a test that needs the "
            "network cannot pass here and is not the grade - deselect it and move on.\n"
            "- Implement the ISSUE LITERALLY: the exact public names, argument names, "
            "defaults, return types, exception types and messages the issue shows. Before "
            "submitting, grep the changed symbol to catch every code path sharing it.\n"
            "- When the harness runs a check for you (an observation starting `[harness]`), "
            "read it and act on it: edit the source, or submit. Do not answer with prose.\n")

    def _st(self, state):
        st = state.get("h")
        if st is None:
            st = {"calls": 0, "edits": 0, "patch_lines": -1, "test_runs": 0,
                  "last_edit_call": -1, "last_test_call": -1, "last_test_cmd": None,
                  "last_pass": None, "seen": {}, "out": {}, "bad": {}, "gen": 0,
                  "blocked": 0, "notes": {}, "flags": [], "probe": None,
                  "good_py": None, "revive": 0, "vgate": 0, "fgate": 0}
            state["h"] = st
        return st

    # ------------------------------------------------------------ command in
    def before_tool(self, name, args, state):
        st = self._st(state)
        if name != "bash":
            return args
        cmd = (args or {}).get("command") or ""
        if not cmd.strip():
            return args
        if _HARNESS_TAG in cmd:
            return args                      # the harness's own injected check

        fixed = _unmangle(cmd)
        if fixed != cmd:
            args = dict(args)
            args["command"] = fixed
            cmd = fixed
            st["flags"] = st["flags"] + ["unmangle"]

        gp = st["good_py"]
        if gp and not _WRITE_BODY_RE.search(cmd) and "GOODPY" not in cmd:
            new = _interp_fix(cmd, gp)
            if new != cmd:
                args = dict(args)
                args["command"] = new
                cmd = new
                st["flags"] = st["flags"] + ["py-rewrite"]

        pm = _PIP_RE.match(cmd)
        if pm:
            rest = pm.group(3) or ""
            local = re.search(r"(-e\b|--no-deps|--no-index|--find-links|\S+\.(whl|tar\.gz)|"
                              r"(?:^|\s)\.{1,2}(?:\s|$)|/)", rest)
            if not local and st["notes"].get("pip", 0) < 2:
                st["notes"]["pip"] = st["notes"].get("pip", 0) + 1
                return ("install blocked: no network",
                        "[harness] BLOCKED: this sandbox has no network, so installing '%s' "
                        "can only fail. The env is pre-provisioned - work with what is "
                        "installed. Judge your change by the tests that DO run here."
                        % " ".join(rest.split())[:80])

        mod = st["probe"]
        if mod and not _mutates(cmd) and "GOODPY" not in cmd:
            st["probe"] = None
            safe = re.sub(r"[^0-9A-Za-z_.]", "", mod)
            args = dict(args)
            args["command"] = (_PROBE_HEAD % safe) + "; " + cmd
            st["flags"] = st["flags"] + ["probe"]
            return args

        # ground truth for the patch state, riding on a command the model runs anyway
        if (st["edits"] == 0 and st["calls"] >= 8 and st["calls"] % 7 == 0
                and "<<" not in cmd and "@@PATCHSTAT" not in cmd
                and _READ_START_RE.match(cmd)):
            args = dict(args)
            args["command"] = _TRUTH_HEAD + cmd
            return args

        norm = _norm(cmd)
        key = (st["gen"], norm)
        nrun = st["seen"].get(key, 0)
        limit = 2 if st["bad"].get(key) else 1
        if nrun >= limit and not _mutates(cmd):
            if _READ_START_RE.match(cmd) or _TEST_RE.search(cmd):
                st["seen"][key] = nrun + 1
                st["blocked"] += 1
                st["flags"] = st["flags"] + ["dup"]
                prev = st["out"].get(key) or "(output not captured)"
                if len(prev) > 2600:
                    prev = prev[:2000] + "\n... [middle elided] ...\n" + prev[-500:]
                return ("duplicate command served from cache",
                        "[harness] This EXACT command already ran earlier in this session and "
                        "cannot return anything new, so it was not re-run. Its output:\n" + prev
                        + "\n[harness] Do NOT repeat commands. Two ways forward: (a) make the "
                        "source edit you are circling (sed -i, or a here-doc python script), "
                        "or (b) read a file/range you have not read yet.")
        return args

    # ------------------------------------------------------------ observation in
    def after_tool(self, name, args, obs, state):
        st = self._st(state)
        if name != "bash" or not isinstance(obs, str):
            return obs
        if _ANSI_RE.search(obs):
            obs = _ANSI_RE.sub("", obs)
        cmd = (args or {}).get("command") or ""
        if cmd.startswith(_PROBE_MARK):
            cmd = cmd.split("== end probe ==", 1)[-1].lstrip("; ")
        key = (st["gen"], _norm(cmd))
        st["seen"][key] = st["seen"].get(key, 0) + 1
        if len(obs) <= 5000:
            st["out"][key] = obs
        if _ERR_RE.search(obs):
            st["bad"][key] = True

        pm = _PATCHSTAT_RE.search(obs)
        if pm:
            try:
                st["patch_lines"] = int(pm.group(1))
            except Exception:
                pass
            obs = obs.replace(pm.group(0), "[harness: git diff of /testbed is %s lines]"
                              % pm.group(1))
            if st["patch_lines"] > 0 and st["edits"] == 0:
                st["edits"] = 1
                st["last_edit_call"] = st["calls"]

        if _src_edit(cmd):
            st["edits"] += 1
            st["gen"] += 1
            st["last_edit_call"] = st["calls"]
        elif _mutates(cmd):
            st["gen"] += 1
        if _TEST_RE.search(cmd):
            st["test_runs"] += 1
            st["last_test_call"] = st["calls"]
            if "\n" not in cmd and len(cmd) < 300 and _HARNESS_TAG not in cmd:
                st["last_test_cmd"] = cmd
            st["last_pass"] = bool(_PASSED_RE.search(obs)) and not _FAILCOUNT_RE.search(obs)

        if not st.get("good_py") and not _MOD_ERR_RE.search(obs):
            im = re.search(r"(/[\w./-]*bin/python[0-9.]*)\s+-m\s+(?:pytest|unittest)\b", cmd)
            if im and re.search(r"\d+ (?:passed|failed)|ran \d+ tests|^OK", obs):
                st["good_py"] = im.group(1)
        m = _GOODPY_RE.search(obs)
        if m and not st.get("good_py"):
            st["good_py"] = m.group(1)
            obs += ("\n[harness] Working interpreter for this repo: %s - use it for every "
                    "python/pytest run from now on (other interpreters lack the project "
                    "dependencies).\n" % st["good_py"])

        if re.search(r"No module named /|No module named '/|Error while finding module "
                     r"specification", obs) and st["notes"].get("mangle", 0) < 2:
            st["notes"]["mangle"] = st["notes"].get("mangle", 0) + 1
            obs += ("\n[harness] That error came from a BROKEN command, not from your code: "
                    "`python -m <path-to-python> -m pytest ...` names an interpreter as a "
                    "module. The runnable form is exactly one of:\n"
                    "    /opt/conda/envs/testbed/bin/python -m pytest <file> -q\n"
                    "    /opt/conda/envs/testbed/bin/python -c \"...\"\n"
                    "Write it fresh - do not copy the previous command.\n")

        mm = _MOD_ERR_RE.search(obs)
        if mm and st["probe"] is None and not st.get("good_py") and "env probe" not in cmd:
            st["probe"] = mm.group(1).split(".")[0]
            obs += ("\n[harness] Wrong interpreter: bare `python` has no project deps. Use "
                    "the repo env interpreter (e.g. "
                    "/opt/conda/envs/testbed/bin/python -m pytest ...).\n")

        nf = _NODEID_RE.search(obs)
        if nf and st["notes"].get("nodeid", 0) < 2:
            st["notes"]["nodeid"] = st["notes"].get("nodeid", 0) + 1
            obs += ("\n[harness] That test id was destroyed by the shell, not by pytest: "
                    "bracketed/parametrised ids contain characters that word-split. Either "
                    "QUOTE the whole node id in single quotes, or drop the bracket part and "
                    "filter with `-k <test_name>`, or run the whole test FILE. Never retry the "
                    "same unquoted id.\n")

        if _NET_ERR_RE.search(obs) and st["notes"].get("net", 0) < 2:
            st["notes"]["net"] = st["notes"].get("net", 0) + 1
            obs += ("\n[harness] This sandbox has NO network. Tests needing the network or "
                    "live credentials cannot pass here and are not the grade - deselect them "
                    "(--deselect / -k 'not ...') and keep working on the code change.\n")

        if _HARNESS_TAG in cmd and st["notes"].get("vguide", 0) < 6:
            st["notes"]["vguide"] = st["notes"].get("vguide", 0) + 1
            if st["edits"] == 0:
                head = ("[harness] Your last reply contained NO tool call, which ends this "
                        "episode with the current git diff - which is EMPTY. An empty diff is "
                        "a guaranteed zero, so the harness ran this check instead. Decide from "
                        "what you have already read which function owns the behaviour the issue "
                        "describes and WRITE the change now with `sed -i` or a here-doc python "
                        "script, then run the affected test file. A rough patch can pass; "
                        "nothing never does.\n")
            elif st["last_pass"] is False:
                head = ("[harness] You were about to stop while the tests you last ran were "
                        "still RED, so the harness re-ran them for you. Fix the source until "
                        "these pass (or until you have proven the failure is a network/env "
                        "issue), then `git diff` and submit. Re-reading files changes nothing.\n")
            else:
                head = ("[harness] You changed source but never ran a test after that edit, so "
                        "the harness re-ran your own last test command. If it passes, run the "
                        "test file that covers the code you edited, then `git diff` and submit; "
                        "if it fails, fix the source. Do not reply in prose.\n")
            obs = head + obs

        if len(obs) > 5200 and _READ_START_RE.match(cmd):
            obs = (obs[:4000] + "\n... [harness: %d chars elided - this file is long; read "
                   "ranges with sed -n 'A,Bp' or locate lines with grep -n instead of catting "
                   "whole files] ...\n" % (len(obs) - 4900) + obs[-900:])
        if _TEST_RE.search(cmd) and len(obs) > 1600:
            dig = self._digest(obs)
            if dig:
                obs = dig
        return obs

    def _digest(self, obs):
        """Condense a long pytest/unittest log to summary + failing nodes + the first real
        traceback, so failure detail survives the observation cap."""
        summ = _SUMMARY_RE.findall(obs)
        if not summ:
            summ = _SUM_ANY_RE.findall(obs)[-2:]
        fails = _FAIL_LINE_RE.findall(obs)
        m = re.search(r"^=+ (?:FAILURES|ERRORS) =+\s*$", obs, re.M)
        if m:
            head = obs[m.start():m.start() + 1600]
        elif "Traceback (most recent call last)" in obs:
            i = obs.rfind("Traceback (most recent call last)")
            head = obs[max(0, i - 400):i + 1200]
        elif fails:
            head = obs[-1400:]
        else:
            head = obs[:700]
        out = "[test output, condensed by the harness]\n"
        if summ:
            out += "SUMMARY: " + " | ".join(s.strip("= ") for s in summ[-3:]) + "\n"
        if fails:
            out += "FAILING NODES (%d): %s\n" % (len(fails),
                                                 "; ".join(f[:150] for f in fails[:10]))
        return (out + head)[:3200]

    # ------------------------------------------------------------ model reply
    def after_llm(self, content, state):
        st = self._st(state)
        st["calls"] += 1
        if not isinstance(content, list):
            return content
        if any(isinstance(b, dict) and b.get("type") == "tool_use" for b in content):
            return content                    # a real tool call: identity, nothing to fix
        # Rebuild from text blocks ONLY: the validator rejects any other block type and
        # would drop the whole return (that is what silently killed round 1).
        keep = [b for b in content if isinstance(b, dict) and b.get("type") == "text"]
        text = "\n".join(b.get("text", "") or "" for b in keep)

        # (a) PARSE-RESCUE: the tool call was written as TEXT -> make it a real action.
        cmd = _text_cmd(text)
        if cmd:
            st["flags"] = st["flags"] + ["rescue"]
            return keep + [_tu(cmd)]

        # (b) ANTI-DEATH: a reply with no tool call ENDS the episode. Only intervene while the
        #     episode is objectively in a losing state, and only a bounded number of times.
        if st["calls"] < 4:
            return content
        has_patch = st["edits"] > 0 or st["patch_lines"] > 0
        if not has_patch and st["revive"] < 3:
            st["revive"] += 1
            st["flags"] = st["flags"] + ["revive"]
            return keep + [_tu(_REVIVE_CMD)]
        if has_patch and st["last_edit_call"] > st["last_test_call"] and st["vgate"] < 2:
            st["vgate"] += 1
            st["flags"] = st["flags"] + ["vgate"]
            base = st["last_test_cmd"] or ("cd /testbed && git diff --stat | tail -5")
            return keep + [_tu(base + (" " + _HARNESS_TAG))]
        if (has_patch and st["last_pass"] is False and st["fgate"] < 2
                and st["last_test_cmd"]):
            st["fgate"] += 1
            st["flags"] = st["flags"] + ["fgate"]
            return keep + [_tu(st["last_test_cmd"] + (" " + _HARNESS_TAG))]
        return content

    # ------------------------------------------------------------ cross-turn note
    def on_turn_end(self, turn, state):
        st = self._st(state)
        flags, st["flags"] = st["flags"], []
        note = None
        if "unmangle" in flags:
            note = ("The harness rewrote your last command: it contained a broken interpreter "
                    "form (`python -m <path> -m pytest`). Use `python -m pytest <file> -q`. "
                    "Keep going: fix the source, run the affected test file, or submit.")
        elif "revive" in flags:
            note = ("Your last reply had no tool call and /testbed had NO source edit - that "
                    "combo scores zero. Pick the file you have already read that owns the "
                    "behaviour in the issue and write the fix NOW (`sed -i`, or a here-doc "
                    "python script that does s/old/new/ on the lines you have seen). Then run "
                    "the affected test file with the repo interpreter.")
        elif "vgate" in flags:
            note = ("You stopped with an edited but untested patch, so the harness ran your "
                    "last test command. Read the result above, then either fix the source, run "
                    "the covering test file, or submit. A reply with no tool call ends the run.")
        elif "fgate" in flags:
            note = ("The tests were still failing when you stopped. Change the SOURCE, not the "
                    "reading: the failing assertion names the symbol to fix. Then `git diff` "
                    "and submit.")
        elif "rescue" in flags:
            note = ("Your previous reply wrote a command as text, so it never ran. Always make "
                    "a NATIVE bash tool call; a prose reply ends the episode.")
        elif "dup" in flags:
            note = ("You are re-running commands instead of changing the repo. State the one "
                    "fact you now know, then ACT: edit the file that owns the behaviour, or "
                    "run the affected test file with the repo interpreter.")
        elif st["edits"] == 0 and turn in _EDIT_TURNS:
            note = ("No file has been modified yet (turn %d). Stop exploring and make the "
                    "first edit now with what you already know: `sed -i` or a here-doc python "
                    "script on a file you have read. A rough patch can still pass; an empty "
                    "one never does. Then run the affected test file." % turn)
        elif (st["edits"] and st["last_test_call"] < st["last_edit_call"]
              and turn - st["last_edit_call"] >= 2 and st["notes"].get("vnote", 0) < 3):
            st["notes"]["vnote"] = st["notes"].get("vnote", 0) + 1
            note = ("You changed source and have not run a test since that edit. Run the "
                    "affected test file (repo interpreter, scoped with -k or the test FILE, "
                    "not the whole suite) before submitting; regressions in existing tests are "
                    "graded too.")
        elif turn >= 6 and turn % 9 == 0:
            note = ("[status] edits=%d test_runs=%d turn=%d. Loop: grep/read a range -> edit "
                    "source -> run the affected test file -> git diff -> submit. Never repeat "
                    "a command; always leave a non-empty patch."
                    % (st["edits"], st["test_runs"], turn))
        if note is None:
            return None
        if st["notes"].get(note, 0) >= 2:
            return None
        st["notes"][note] = st["notes"].get(note, 0) + 1
        return note
