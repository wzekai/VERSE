"""harness_code/hooks.py - environment normalisation, patch protection, verification.

Over 110 episodes/79 failures: 41 burned turns on the BASE interpreter (no pytest) ->
python/pytest calls are rewritten to the repo conda env; 16 hit unquoted parametrized
node ids -> quoted; 0/110 ever installed a package (offline) yet 45 tried -> remote
fetches are refused with a synthetic observation; 13 ended on an EMPTY diff ->
patch-destroying git commands and empty submits are refused, actionless replies revived.
New this round: pytest output is truncated from the FRONT, so the failure block at the
end never reaches the model -> _test_digest rebuilds it; and 30 episodes were graded
failed with a green visible run -> the diff is reviewed against the ISSUE's own wording
(fresh-eyes llm pass on `git diff` + a check that the issue names appear in it).
"""

import re

TP = "/opt/conda/envs/testbed/bin/python"
_TP_SETUP = ('TP=' + TP + '; [ -x "$TP" ] || TP=$(command -v python3 || command -v python'
             ' || echo python); ')
_STATUS = ("cd /testbed && git status --porcelain=v1 | head -20 && git diff --stat | tail -10")


def _norm(s):
    return re.sub(r"\s+", " ", (s or "").strip())


_REMOTE = re.compile(
    r"(?:^|[;&|]\s*|\s)(?:pip[23]?|pipx|uv|poetry|conda|mamba)\s+(?:install|download|wheel"
    r"|sync|update|add|resolve)\b|-m\s+pip\s+(?:install|download|wheel)\b"
    r"|\b(?:curl|wget)\b|\b(?:apt|apt-get|dnf|yum|apk)\s+(?:install|update|add)\b"
    r"|\b(?:npm|yarn|pnpm)\s+(?:install|add|ci)\b|\bgit\s+(?:clone|fetch|pull)\b", re.I)
_LOCAL_OK = re.compile(r"--no-index|--editable|\s-e\s|-r\s+\S|find-links|--no-build-isolation", re.I)
_OFFLINE_SIG = re.compile(
    r"NewConnectionError|Max retries exceeded|Could not resolve host|Temporary failure in "
    r"name resolution|Network is unreachable|Failed to connect|ConnectTimeoutError|"
    r"Name or service not known|HTTPSConnectionPool|Errno -[13]\b|Could not find a version"
    r"|Could not fetch url", re.I)
_SEDI = re.compile(r"\bsed\s+(?:-[a-zA-Z]+\s+)*-i\b")
_PERLI = re.compile(r"\bperl\s+-[a-zA-Z]*i[a-zA-Z]*\b")
_PATCHW = re.compile(r"\bpatch\s+-p|\bgit apply\b|\bapply_patch\b", re.I)
_TEE = re.compile(r"\btee\b\s+-?\w*\s+\S")
_OPENW = re.compile(r"open\(\s*[\"'][^\"']+[\"']\s*,\s*[\"'][wax]|\.write_text\(|\.write_bytes\(", re.I)
_REDIR = re.compile(r">>?\s*([^\s|&;<>)|]+)")
_MOVEIN = re.compile(r"(?:^|[;&|]\s*)(?:cp|mv|rsync)\s+\S+\s+(?:[^\s]*/)?[\w.\-]+\.\w+")
_TMPISH = ("/dev/", "/tmp/", "/var/tmp/", "/proc/", "/sys/", "/run/", "stderr", "stdout")
_READ_ONLY = re.compile(
    r"^\s*(?:cd\s+[^\n&;]+(?:&&|;)\s*)?(?:ls|cat|head|tail|sed|grep|egrep|fgrep|rg|ag|find|fd"
    r"|wc|tree|file|stat|readlink|realpath|sort|uniq|cut|nl|jq"
    r"|git\s+(?:status|log|diff|show|blame|ls-files|ls-tree|cat-file|describe)"
    r"|pip(?:[23])?\s+(?:list|show|freeze)|python[0-9.]*\s+--version|echo|printf|which"
    r"|whereis|type|conda\s+env\s+list)\b", re.I)
_TESTRUN = re.compile(r"\bpytest\b|\bpy\.test\b|python[0-9.]*\s+-m\s+(?:pytest|unittest)|"
                      r"\bnosetests?\b|\btox\b|make\s+test|\bgo\s+test\b|\bnpm\s+test|\bjest\b", re.I)
_TESTS_OK = re.compile(r"(\d+)\s+passed", re.I)
_TESTS_BAD = re.compile(r"(\d+)\s+(?:failed|error)", re.I)
_MISSING_PYTEST = re.compile(r"No module named ['\"]?pytest", re.I)
_MISSING_MOD = re.compile(r"ModuleNotFoundError: No module named ['\"]?([\w.]+)", re.I)
_NOTFOUND = re.compile(r"ERROR:\s*not found|no tests ran|file or directory not found", re.I)
_TESTFILE = re.compile(r"(^|[/\s'\"])(tests?/|test_[a-z0-9_]+\.py|[a-z0-9_]+_test\.py)", re.I)
_DESTROY = re.compile(r"\bgit\s+(?:stash|restore|clean|switch|reset\s+--(?:hard|merge))"
                      r"|\bgit\s+checkout\b(?!\s+-b\b)", re.I)
_PY_AT_CMD = re.compile(r'(?<![\w./$-])(python[0-9]*)(?=\s)')
_PYTEST_AT_CMD = re.compile(r'(?<![\w./$-])(?:command\s+run\s+)?(py\.test|pytest)(?=\s)')
_LEAD_READER = re.compile(
    r"^\s*(?:cd\s+[^\n&;]+(?:&&|;)\s*)?(?:grep|egrep|fgrep|rg|ag|sed|cat|echo|printf|awk|find"
    r"|ls|man|which|type|head|tail|less|wc|diff|strings|history)\b", re.I)
_INSTALLY = re.compile(r"\b(?:pip|conda|uv|poetry)\b.*\b(?:install|upgrade)\b", re.I)

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)```", re.S | re.I)
_TAGGED = re.compile(r"<\s*(?:bash|command|execute)\s*>(.*?)<\s*/\s*(?:bash|command|execute)\s*>",
                     re.S | re.I)
_SHELLY = re.compile(r"^(?:cd |ls |cat |grep|find|sed|awk|python|pytest|git |echo|make|cp |mv |"
                     r"head|tail|/opt/|export |mkdir|rm |touch|pip|source|./|tox)", re.I)


def _edit_signal(cmd):
    if not cmd:
        return False
    if _SEDI.search(cmd) or _PERLI.search(cmd) or _PATCHW.search(cmd) or _OPENW.search(cmd):
        return True
    if _TEE.search(cmd) or _MOVEIN.search(cmd):
        return True
    for m in _REDIR.finditer(cmd):
        p = m.group(1)
        if p.startswith("&") or p == "/dev/null" or any(p.startswith(t) for t in _TMPISH):
            continue
        if re.search(r"\.\w+$", p) or "/" in p:
            return True
    return False


def _quote_node_ids(cmd):
    """Quote pytest node ids containing '[' - the shell splits them otherwise and pytest
    answers 'ERROR: not found' (observed in 16 episodes)."""
    if "::" not in cmd:
        return cmd, False
    out, pos, changed = [], 0, False
    for m in re.finditer(r"[^\s|&;]+", cmd):
        tok, s = m.group(0), m.start()
        if "::" not in tok or "[" not in tok or tok.startswith(("'", '"')):
            continue
        if cmd[:s].count("'") % 2 or cmd[:s].count('"') % 2:
            continue                                  # already inside quotes
        if "]" not in tok:                            # node id containing a space
            j = cmd.find("]", m.end())
            if j < 0 or j - s > 220:
                continue
            tok = cmd[s:j + 1]
        out.append(cmd[pos:s])
        out.append("'" + tok + "'")
        pos = s + len(tok)
        changed = True
    if not changed:
        return cmd, False
    out.append(cmd[pos:])
    return "".join(out), True


def _fix_interpreter(cmd, state):
    """Point python/pytest at the repo's conda env: the ambient `python` is a different
    interpreter with no pytest and none of the repo's dependencies."""
    if state.get("tp_bad") or ("pytest" not in cmd and "python" not in cmd):
        return cmd, False
    if TP in cmd or "$TP" in cmd or "conda activate" in cmd or "uv run" in cmd \
            or "poetry run" in cmd or "hatch run" in cmd or ".venv" in cmd or "conda run" in cmd:
        return cmd, False
    if _INSTALLY.search(cmd) or _LEAD_READER.match(cmd):
        return cmd, False
    sents = {}

    def _mask(m):
        k = "\x02%d\x02" % len(sents)
        sents[k] = m.group(0)
        return k

    new = re.sub(r"-m\s+(?:pytest|py\.test|unittest)", _mask, cmd)
    new = _PY_AT_CMD.sub('"$TP"', new)
    new = _PYTEST_AT_CMD.sub('"$TP" -m \\1', new)
    for k, v in sents.items():
        new = new.replace(k, v)
    if new == cmd:
        return cmd, False
    return _TP_SETUP + new, True


def _requirements(issue):
    """The issue's explicit asks - a checklist to hold the diff against."""
    if not issue:
        return []
    pats = re.compile(r"\b(?:should|must|expected|expect to|needs to|shall|returns?|raises?"
                      r"|support|add|allow|instead|preserve|fails?|works?|behaves?|option"
                      r"|deprecat|defaults?|error|ignores?|matches?)\b", re.I)
    lines, seen = [], set()
    for chunk in re.split(r"(?<=[.!?])\s+|\n", issue.replace("\\r", "")):
        c = chunk.strip().strip("-*# ").strip()
        if not (14 < len(c) < 220) or c in seen or not pats.search(c):
            continue
        if c.lower().startswith(("issue:", "title:", "```")):
            continue
        seen.add(c)
        lines.append(c[:200])
        if len(lines) >= 6:
            break
    return lines


def _markup_commands(texts):
    """Commands the model wrote as text/markup instead of as a native tool call."""
    found = []
    for t in texts:
        if not t:
            continue
        for m in _TAGGED.finditer(t):
            c = m.group(1).strip()
            if c and len(c) < 6000:
                found.append(c)
        for m in _FENCE.finditer(t):
            lang, body = (m.group(1) or ""), m.group(2).strip()
            if not body or len(body) > 6000:
                continue
            first = body.splitlines()[0].lstrip("$+ ").strip()
            if lang.lower() in ("python", "py"):
                if _edit_signal(body):
                    found.append("python3 - <<'PY_RESCUE'\n" + body + "\nPY_RESCUE")
                continue
            if lang or _SHELLY.match(first) or "$ " in body[:60]:
                found.append(body)
    keep = []
    for c in found:
        if c.startswith("python3 - <<'PY_RESCUE'"):
            keep.append(c)
            continue
        lines = [l for l in c.splitlines() if l.strip()]
        if not lines:
            continue
        shellish = sum(1 for l in lines if _SHELLY.match(l.lstrip("$+ ")) or l.startswith("cd "))
        if shellish >= max(1, len(lines) // 2) or len(lines) == 1:
            keep.append(c)
    return keep


# ---------------------------------------------------------------- test digest
_SUMROW = re.compile(r"^=+[^=\n]*(?:passed|failed|error|no tests ran)[^=\n]*=+\s*$", re.M)
_COUNTS = re.compile(r"\d+ (?:passed|failed|errors?|skipped|xfailed|deselected)")
_SHORT = re.compile(r"^(?:FAILED|ERROR)\s+\S.{0,190}$", re.M)
_HDRLINE = re.compile(r"^_{5,}\s*(\S.{0,150}?)\s*_{5,}\s*$", re.M)
_ELINE = re.compile(r"^E\s+(\S.{0,170})$", re.M)
_LOC = re.compile(r"^\s*(\S+\.py):(\d+)", re.M)
_COLLBAD = re.compile(r"errors? during collection|ImportError|ModuleNotFoundError|"
                      r"SyntaxError|fixture .* not found", re.I)
_GITDIFF = re.compile(r"\bgit\b[^|;&]*\bdiff\b")
_IDNAME = re.compile(r"[`\"]([A-Za-z_][A-Za-z0-9_]{2,40}|--[a-z][\w-]{2,30})[`\"]")
_KNOWN = {"python", "pytest", "git", "bash", "true", "false", "none", "null", "str", "int",
          "bool", "list", "dict", "self", "args", "kwargs", "http", "api", "issue", "test",
          "tests", "json", "yaml", "setup"}


def _test_digest(cmd, obs):
    """Test output is truncated from the FRONT (obs_cap): rebuild the failure block that
    sits past the cut."""
    if len(obs) < 4200 or not _TESTRUN.search(cmd):
        return None
    if not (_SUMROW.search(obs) or _SHORT.search(obs) or _HDRLINE.search(obs)):
        return None
    head = []
    rows = _SUMROW.findall(obs)
    cnt = rows[-1].strip().strip("= ") if rows else ", ".join(_COUNTS.findall(obs)[-4:])
    if cnt:
        head.append("[harness digest of that test run] " + " ".join(cnt.split())[:200])
    shorts = _SHORT.findall(obs)
    if shorts:
        head.append("failing/erroring tests:\n" +
                    "\n".join("- " + x.strip()[:200] for x in shorts[:12]))
    else:
        hdrs = list(_HDRLINE.finditer(obs))
        items = []
        for i, h in enumerate(hdrs[:10]):
            end = hdrs[i + 1].start() if i + 1 < len(hdrs) else min(len(obs), h.end() + 4000)
            seg = obs[h.end():end]
            e = _ELINE.findall(seg)
            loc = _LOC.findall(seg)
            items.append("- " + h.group(1)[:150] + ("  -> " + e[0][:150] if e else "")
                         + ("  @ " + loc[-1][0] + ":" + loc[-1][1] if loc else ""))
        if items:
            head.append("failures (test -> first error line):\n" + "\n".join(items))
    cb = _COLLBAD.search(obs)
    if cb and not shorts:
        head.append("[harness] collection problem: "
                    + " ".join(obs[max(0, cb.start() - 150):cb.start() + 300].split())[:380])
    if not head:
        return None
    return ("\n".join(head)[:1900] +
            "\n--- raw tail (middle trimmed; for a full traceback re-run just that test: "
            "`-m pytest <file>::<name> -q --tb=long`) ---\n" + obs[-1100:])


def _api_names(issue):
    """Backticked identifiers the ISSUE spells out - hidden tests use those names."""
    out = []
    for m in _IDNAME.finditer(issue or ""):
        t = m.group(1)
        if t.lower() in _KNOWN or t in out:
            continue
        if not (t.startswith("--") or "_" in t
                or (len(t) > 3 and any(c.isupper() for c in t[1:]))):
            continue
        out.append(t)
        if len(out) >= 5:
            break
    return out


_DIRTEST = re.compile(r"(?:pytest|py\.test|unittest)(?P<rest>[^|;&]*)")

def _wide_target(cmd):
    """True when a test-run command targets a DIRECTORY (a whole package of tests) rather
    than one file - the only cheap defence against regressions next door."""
    m = _DIRTEST.search(cmd)
    if not m:
        return False
    for tok in m.group("rest").split():
        if tok.startswith("-") or "=" in tok:
            continue
        if tok.endswith(".py") or "::" in tok:
            return False
        if "/" in tok or tok in ("tests", "test", ".", "src"):
            return True
    return False

_REVIEW_PROMPT = ("You are the last reviewer of a candidate patch for a GitHub issue. "
    "It will be graded by the project's HIDDEN tests, written from the issue text: they "
    "use the exact names, defaults, error messages and edge cases the issue implies, and "
    "they also re-run the existing suite.\n\nISSUE:\n%s\n\nCANDIDATE PATCH (git "
    "diff):\n%s\n\nReply with AT MOST 6 lines, each starting 'GAP:' (an issue "
    "requirement this patch does not implement, plus the file/function where it belongs) "
    "or 'RISK:' (an existing behaviour, caller or test this patch breaks, named). "
    "Check: sibling code paths (sync/async, other classes, CLI, other callers), "
    "None/empty/edge inputs, the exact spelling of new names/defaults/error messages, "
    "backwards compatibility. If the patch is complete and low-risk reply exactly: "
    "COVERED. No praise, no restating the patch.")


class Hooks(BaseHooks):

    loop = {
        "keep_full_turns": 16,          # stalled episodes re-read files: wider full window
        "nudge_no_tool": ("Reply with a NATIVE tool call, not prose. Run one `bash` command "
                          "now (write your fix with `sed -i` or a python patch script), or "
                          "`submit` once `git diff` shows it. Prose-only ends the episode and "
                          "an empty diff scores zero."),
    }

    # ---------------------------------------------------------------- prompt tail
    def system_prompt(self, assembled):
        return (assembled or "") + (
            "\n\nDECIDING FACTS (re-read):\n"
            "1. Tests: `cd /testbed && /opt/conda/envs/testbed/bin/python -m pytest <path> "
            "-q 2>&1 | tail -40`; quote node ids; NO network (pip/curl/clone never work).\n"
            "2. The score IS `git diff`: draft patch by turn ~10, empty diff = zero, and "
            "`git stash` / `git checkout -- <file>` throws your work away.\n"
            "3. Graded = the project's HIDDEN tests for this issue: existing tests passing "
            "is not the bar. New names/options/messages must be spelled as the issue spells "
            "them, and every sibling code path must behave.")

    # ---------------------------------------------------------------- issue capture
    def before_llm(self, msgs, state):
        try:
            if not state.get("issue") and msgs:
                c = msgs[0].get("content") if isinstance(msgs[0], dict) else None
                if isinstance(c, str) and "ssue" in c[:12]:
                    state["issue"] = c[:6000]
        except Exception:
            pass
        return msgs

    # ---------------------------------------------------------------- tool gate
    def before_tool(self, name, args, state):
        try:
            cmd = ((args or {}).get("command") or "")
        except Exception:
            return args
        state["seen"] = state.get("seen", 0) + 1
        if name == "submit":
            return self._submit_gate(args, state)
        if not cmd.strip():
            return args

        # 1. no network in this sandbox
        if _REMOTE.search(cmd) and not _LOCAL_OK.search(cmd):
            return ("no network in this container",
                    "BLOCKED (no network): this sandbox is offline - that command can never "
                    "succeed, do NOT retry it. Dependencies are already installed in the repo "
                    "env: run `cd /testbed && /opt/conda/envs/testbed/bin/python -m pytest "
                    "<file> -q 2>&1 | tail -40`, list them with `-m pip list`. If a test needs "
                    "a download or an uninstalled package, skip it and use a /tmp repro.")

        # 2. protect the patch (it is the score)
        if state.get("edits") and _DESTROY.search(cmd):
            return ("would destroy the patch under test",
                    "BLOCKED: that git command would stash or discard the working-tree "
                    "changes that ARE your score. To compare with the pristine file safely:\n"
                    "  git show HEAD:relative/path.py > /tmp/orig.py; cp relative/path.py "
                    "/tmp/mine.py; diff /tmp/orig.py /tmp/mine.py")

        key = _norm(cmd)
        cached = state.get("reads", {}).get(key)
        if cached is not None and _READ_ONLY.match(cmd) and not _edit_signal(cmd) \
                and state.get("readcache_hits", 0) < 4:
            state["readcache_hits"] = state.get("readcache_hits", 0) + 1
            body = cached if len(cached) < 1000 else cached[:1000] + "\n...[cut]"
            return ("repeat of an earlier read-only command",
                    "[harness] You already ran this exact command; recorded output (nothing "
                    "re-ran):\n" + body + "\nStop re-reading: open the file and WRITE the "
                    "fix. `git diff` must not be empty when the episode ends.")

        # 4. node-id quoting + interpreter rewrite
        new, ch1 = _quote_node_ids(cmd)
        new, ch2 = _fix_interpreter(new, state)
        if ch1 or ch2:
            args = dict(args or {})
            args["command"] = new
            bits = []
            if ch1:
                bits.append("quoted the pytest node id (unquoted [ ] -> 'ERROR: not found')")
            if ch2:
                bits.append("used the repo env interpreter $TP (" + TP + ")")
            state["rewrite_note"] = "; ".join(bits)

        # 5. pacing book-keeping
        if _edit_signal(cmd):
            state["edits"] = state.get("edits", 0) + 1
            state["edit_turn"] = state["seen"]
            state["reads"] = {}
            if _TESTFILE.search(cmd):
                state["touched_tests"] = state.get("touched_tests", 0) + 1
        if _TESTRUN.search(cmd):
            state["ran_tests"] = state.get("ran_tests", 0) + 1
        return args

    def _submit_gate(self, args, state):
        """Never end on an empty diff; make the first submit prove the patch was run."""
        if not state.get("edits"):
            state["empty_submits"] = state.get("empty_submits", 0) + 1
            if state["empty_submits"] > 2:
                return args
            return ("nothing to grade",
                    "REFUSED: `git diff` is empty, so this scores ZERO however good the "
                    "analysis was. Turns remain: 1) grep the symbol/message from the issue to "
                    "pick the responsible function; 2) write your best-effort fix into it "
                    "(`sed -i` or a python patch script) - a partial fix scores, nothing does "
                    "not; 3) run that area's test file, then submit again.")
        if not state.get("submit_gate"):
            state["submit_gate"] = 1
            reqs = _requirements(state.get("issue", ""))
            msg = ("NOT YET - one verification pass, then call submit again:\n"
                   "a) run WHOLE test files of every module you edited, then one DIRECTORY run "
                   "(regressions hide next door):\n"
                   "   cd /testbed && /opt/conda/envs/testbed/bin/python -m pytest "
                   "tests/<area>/ -q 2>&1 | tail -40\n"
                   "b) implement every GAP line the fresh-eyes review put on your `git diff`;\n"
                   "c) hold each explicit request below against `git diff` - the grader runs "
                   "hidden tests written from these lines:")
            msg += ("\n" + "\n".join("- " + r for r in reqs)) if reqs else \
                   ("\n- (nothing machine-extractable: re-read the issue sentence by sentence)")
            if state.get("tests_green") is False:
                msg += ("\n- your LAST test run still had failures: fix them, or show they "
                        "are pre-existing on the pristine file (git show HEAD:<path> > /tmp/o.py).")
            return ("first submit: verification pass required", msg)
        return args

    # ---------------------------------------------------------------- observations
    def after_tool(self, name, args, obs, state):
        if not isinstance(obs, str):
            return obs
        try:
            cmd = ((args or {}).get("command") or "")
        except Exception:
            return obs
        extra = ""
        note = state.pop("rewrite_note", None)
        if note:
            extra += "\n\n[harness] " + note
        if _norm(cmd) != _norm(_STATUS) and _READ_ONLY.match(cmd) and not _edit_signal(cmd) \
                and len(obs) < 20000:
            state.setdefault("reads", {})[_norm(cmd)] = obs[:4000]
        if TP in cmd and "No such file or directory" in obs:
            state["tp_bad"] = True
        if not state.get("offline") and _OFFLINE_SIG.search(obs):
            state["offline"] = True
            extra += ("\n\n[harness] Network failure: this container has NO internet. Never "
                      "retry a fetch; use what is installed in /opt/conda/envs/testbed and a "
                      "local /tmp repro.")
        if _TESTRUN.search(cmd):
            if not state.get("wide_ok") and _wide_target(cmd):
                state["wide_ok"] = 1
            if not state.get("wide_ok") and _TESTS_OK.search(obs) \
                    and max([int(x) for x in re.findall(r"(\d+) passed", obs)] or [0]) >= 25:
                state["wide_ok"] = 1
            dig = _test_digest(cmd, obs)
            if dig:
                obs = dig
            if _TESTS_BAD.search(obs):
                state["tests_green"] = False
                state["red_seen"] = 1
            elif _TESTS_OK.search(obs):
                state["tests_green"] = True
                state["red_seen"] = 0
                state["green_turn"] = state.get("seen", 0)
        if _MISSING_PYTEST.search(obs) and "envs/testbed" not in cmd:
            extra += ("\n\n[harness] `python`/`pytest` there is the BASE interpreter without "
                      "pytest. Always use the repo env:\n  cd /testbed && "
                      "/opt/conda/envs/testbed/bin/python -m pytest <path> -q 2>&1 | tail -40")
        m = _MISSING_MOD.search(obs)
        if m and state.get("modwarn", 0) < 2 and "envs/testbed" not in cmd:
            state["modwarn"] = state.get("modwarn", 0) + 1
            extra += ("\n\n[harness] '%s' is missing from THAT interpreter - the repo's "
                      "dependencies live in the conda env `testbed`: re-run with "
                      "/opt/conda/envs/testbed/bin/python. Installing is impossible (offline)."
                      % m.group(1))
        if _NOTFOUND.search(obs) and state.get("notfound_warn", 0) < 2:
            state["notfound_warn"] = state.get("notfound_warn", 0) + 1
            extra += ("\n\n[harness] pytest matched no test for that selection. From the "
                      "repo root, quote the id or use -k:\n  cd /testbed && "
                      "/opt/conda/envs/testbed/bin/python -m pytest "
                      "\"tests/test_x.py::test_name[param]\" -q")
        if state.get("touched_tests") and _TESTFILE.search(cmd) and _edit_signal(cmd) \
                and state.get("testfile_warn", 0) < 1:
            state["testfile_warn"] = 1
            extra += ("\n\n[harness] That wrote to a test file. The grader restores its own "
                      "copies of the repo's tests, so the edit is discarded - and a test file "
                      "that fails to import turns every graded test into 'not-run' = zero. "
                      "Experiments belong in /tmp; change the SOURCE.")
        if _GITDIFF.search(cmd) and "diff --git" in obs and state.get("edits"):
            extra += self._diff_review(obs, state)
        return obs + extra if extra else obs


    def _diff_review(self, diff, state):
        """Hold this diff against the issue's sentences with fresh eyes. 1 metered turn."""
        out = ""
        try:
            issue = (state.get("issue") or "")[:4500]
            names = [n for n in _api_names(issue) if n.strip("-") not in diff]
            if names and state.get("name_warn", 0) < 1:
                state["name_warn"] = 1
                out += ("\n\n[harness] The issue spells these identifiers but your diff does "
                        "not contain them: " + ", ".join("`%s`" % n for n in names) +
                        "\nIf the issue asks for a NEW option/argument/function/attribute, the "
                        "hidden tests use THAT EXACT NAME - an equivalent-but-renamed thing "
                        "scores zero. Check: grep -rn \"NAME\" /testbed --include='*.py'; add "
                        "whatever is genuinely missing.")
            if state.get("reviews", 0) >= 2 or not issue or len(diff) < 150:
                return out
            state["reviews"] = state.get("reviews", 0) + 1
            r = self.llm(_REVIEW_PROMPT % (issue, diff[:9000]), max_tokens=700)
            if r and not r.startswith("ERROR:") and "COVERED" not in r[:40]:
                out += ("\n\n[harness] Fresh-eyes review of your diff against the issue "
                        "(1 turn spent). Fix every GAP before submitting - a patch that "
                        "half-implements the issue is the most common way a green run still "
                        "fails grading:\n" + r.strip()[:1600])
            elif r and not r.startswith("ERROR:"):
                out += "\n\n[harness] Fresh-eyes review: no gap found. Verify + submit."
        except Exception:
            pass
        return out

    # ---------------------------------------------------------------- reply repair
    def after_llm(self, content, state):
        """An episode ends on a reply without tool_use: run commands the model wrote as
        text markup, and refuse to end on an untouched repo (both capped)."""
        if not isinstance(content, list):
            return content
        try:
            tools = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
            names = [b.get("name") for b in tools]
            if "bash" in names or "submit" in names:
                return content          # before_tool's submit gate covers empty submits
            strings = [str(b.get("text", "")) for b in content
                       if isinstance(b, dict) and b.get("type") == "text"]
            if state.get("rescues", 0) < 4:
                cmds = _markup_commands(strings)
                if cmds:
                    state["rescues"] = state.get("rescues", 0) + 1
                    for i, c in enumerate(cmds[:2]):
                        # append in place and return the SAME object: a new list gets
                        # shape-validated and would reject thinking blocks
                        content.append({"type": "tool_use",
                                        "id": "hk_r%d_%d" % (state["rescues"], i),
                                        "name": "bash", "input": {"command": c}})
                    return content
            if state.get("revivals", 0) >= 3:
                return content
            if state.get("edits"):
                note = ("[harness] That reply had no tool call, which ENDS the episode. The "
                        "grade comes from `git diff` after a verification run - confirm your "
                        "patch is on disk and the area's tests pass before you start "
                        "summarising. One more command beats a paragraph.")
            else:
                note = ("[harness] That reply had no tool call, which ENDS the episode with "
                        "an EMPTY diff (score 0). No file has been modified yet: pick the "
                        "function responsible for the issue and write your best-effort fix "
                        "now, then run its tests.")
            content.append({"type": "text", "text": note})
            content.append({"type": "tool_use", "id": "hk_rev%d" % state.get("revivals", 0),
                            "name": "bash", "input": {"command": _STATUS}})
            state["revivals"] = state.get("revivals", 0) + 1
            return content
        except Exception:
            return content

    # ---------------------------------------------------------------- pacing
    def on_turn_end(self, turn, state):
        edits = state.get("edits", 0)
        seen = state.get("seen", 0)
        if not edits:
            if turn == 8:
                return ("~9 turns spent reading and NOT ONE file written. Stop exploring: "
                        "name the function responsible for the issue and write a draft fix "
                        "into it now (`sed -i` or a python patch script). Refine it after - "
                        "the episode is graded on `git diff` and an empty diff is zero.")
            if turn == 14:
                return ("STILL no edits after 15 turns. More reading will not tell you the "
                        "fix: make the most plausible change to the code the issue names, "
                        "run /opt/conda/envs/testbed/bin/python -m pytest <area test file> "
                        "-q 2>&1 | tail -30 and let the failures teach you the rest.")
            if turn >= 20 and (turn - 20) % 5 == 0:
                return ("TURN %d and `git diff` is still empty - a guaranteed zero. Write "
                        "ANY plausible fix to the source now, run the area's tests, then "
                        "submit. Further reading scores nothing." % (turn + 1))
            return None
        if not state.get("ran_tests") and seen - state.get("edit_turn", 0) >= 3 \
                and state.setdefault("nudge_test", 0) < 2:
            return ("You changed source but have not run a single test:\n"
                    "cd /testbed && /opt/conda/envs/testbed/bin/python -m pytest <the test "
                    "file> -q 2>&1 | tail -40\nplus a 10-line repro of the issue. An "
                    "unverified fix is usually wrong, and submit asks for this run anyway.")
        if state.get("tests_green") and turn >= state.get("green_turn", 0) + 3 \
                and state.setdefault("nudge_submit", 0) < 2:
            return ("The tests you ran pass and a patch is on disk. Further polishing rarely "
                    "adds points and risks losing the thread: run the rest of that test FILE "
                    "once (regressions hide in the same module), `git diff` to confirm the "
                    "change is what you meant, then `submit`.")
        if state.get("red_seen") and not state.get("tests_green") \
                and state.setdefault("nudge_red", 0) < 2:
            state["red_seen"] = 0
            return ("Your last test run still had failures. Read the FIRST failing assertion, "
                    "fix that cause (or narrow an over-broad change) and re-run - do not "
                    "finish while the tests covering your own change are red.")
        return None
