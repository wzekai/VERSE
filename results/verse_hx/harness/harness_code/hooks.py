"""hooks.py v4 - the G1-G4 guards (no-patch fails 16/77->8/74, repeat loops 13/77->2/74) plus a
deterministic submit-time completeness review. Measured on the last sweep: the metered reviewer call
failed in 78/110 episodes (only 14 produced any review text), so the review reached the agent as
boilerplate AFTER the harness paid up to 2 metered turns for it; it now runs no model call at all
and always delivers the issue-vs-diff token coverage. FORCE_RO tracks this sweep's read-spiral
ceiling (worst passing streak 32, failing 47).
"""
import re

_PATH_TOK = r"[A-Za-z0-9_./$~+*()-]+"
_REDIRECT_RE = re.compile(r">{1,2}\s*[\"']?(" + _PATH_TOK + r")")
_TEE_RE = re.compile(r"\btee\b\s+(?:-a\s+)?[\"']?(" + _PATH_TOK + r")")
_OPEN_W_RE = re.compile(r"open\(\s*[\"']([^\"']{1,160})[\"']\s*,\s*[\"'][wa]")
_SED_I_RE = re.compile(r"\bsed\b[^|;&]*\s-i\S*")
_PERL_I_RE = re.compile(r"\bperl\s+-[ip]e\b")
_PATCH_RE = re.compile(r"apply_patch|git\s+apply|\bpatch\s+(?:-p\s*\d+|--batch|--forward|-i\s)"
                       r"|<\s*\S+\.(?:patch|diff)\b")
_WRITEFN_RE = re.compile(r"write_text|\.write\(")
_EXT_RE = re.compile(r"(" + _PATH_TOK +
                     r"\.(?:py|pyx|pxd|c|h|cc|cpp|f90|java|js|ts|go|rs|rst|txt|md|cfg|toml|ini|yaml|yml|json))\b")
_TESTPATH_RE = re.compile(r"(?:^|/)(?:tests?/|[^/]*test_[^/]*\.py$|[^/]*_test\.py$)")
_SCRATCH_RE = re.compile(r"^(?:/tmp/|/dev/|/var/tmp/|tmp/)")
_TESTRUN_RE = re.compile(r"\bpytest\b|\bpy\.test\b|-m\s+unittest|\btox\b|\bnosetests\b")
_INSTALL_RE = re.compile(r"\bpip[0-9.]*\s+install|\bconda\s+install|\buv\s+pip\s+install"
                         r"|-m\s+pip[3]?\s+install|\beasy_install\b")
_DEVNULL_RE = re.compile(r"\d?>>\s*/dev/null")
_NETFAIL_RE = re.compile(r"Temporary failure in name resolution|NewConnectionError|"
                         r"Max retries exceeded|Could not find a version|Errno -3|"
                         r"Could not resolve host|Network is unreachable|"
                         r"Name or service not known|Connection refused")
_NOMOD_RE = re.compile(r"No module named [\"']?([A-Za-z0-9_.\-]+)")
_NOTFOUND_BIN_RE = re.compile(r"command not found: (?:/opt[^\s]*/)?py?test\b")
_CMD_PY_RE = re.compile(r"(?<![\w/.\-])(python3?|pytest)(?=\s|$)")
_PROBE_PY_RE = re.compile(r"PROBE_PY (\S+)")
_READPATH_RE = re.compile(r"((?:/testbed/)?[\w./\-]*[\w\-]+\.(?:py|pyx|c|h|rst|md|txt|cfg|toml|ini|yaml|yml|json))")

PROBE = ("{ echo '[harness env-probe]'; for p in /opt/conda/envs/testbed/bin/python "
         "/usr/local/bin/python /opt/conda/bin/python; do [ -x \"$p\" ] && echo \"PROBE_PY $p\"; "
         "done; echo 'NO NETWORK here: pip/conda install always fails - never retry it.'; "
         "echo 'Run the repo with the first interpreter above: <that python> -m pytest -x -q <file>'; "
         "echo '[end env-probe]'; } 2>&1\n")

INJECT_CMD = ("echo '[harness] diff state:'; git --no-pager diff --stat | tail -n 6; "
              "git status --porcelain | head -n 12; echo '[end diff state]'")

_WRITE_HOWTO = ("Do it in ONE bash call now: cd /testbed && python - <<'PY' ... "
                "p='/testbed/<file>.py'; s=open(p).read(); s=s.replace(OLD, NEW); "
                "open(p,'w').write(s) ... PY  (or cat > file <<'EOF'). An edit you are unsure of "
                "scores more than an empty diff; verify it with the project interpreter after.")

NOTE_EMPTY = ("[harness] your reply had NO tool call, which ENDS the episode, and your diff is "
              "EMPTY - the grader scores the diff, so this is an automatic 0. Prose is never "
              "executed. " + _WRITE_HOWTO)

NOTE_PATCH = ("[harness] your reply had no tool call, which ends the episode as shown below. If "
              "the change is complete and verified call `submit`, else keep working in bash.")

NOTE_PROSE = ("[harness] that shell command was TEXT, not a bash tool call, so NOTHING ran. "
              "Call the bash tool with it now.")

GATE_EMPTY = ("[harness] `submit` blocked: your `git diff` is EMPTY and the grader scores exactly "
              "that diff, so this is a guaranteed 0. Test-file edits do not count (the grader "
              "checks out its own). Implement the ISSUE in the SOURCE, then submit again. "
              + _WRITE_HOWTO)

DRAFT_NOTE = ("[harness] your diff is still EMPTY and this episode is about to score 0, so the "
              "harness is drafting a first source edit for you to fix. Not a substitute for your "
              "own fix - refine it, verify it, then submit.")

GATE_REVIEW = ("[harness] `submit` intercepted once for a review of the patch against the issue "
               "(the graded tests are NOT in your checkout, so a green local run proves nothing). "
               "Fix anything real in the review below, then `submit` again - it goes through.")

PIP_MSG = ("[harness] blocked: NO network here, so every pip/conda install fails on a DNS "
           "timeout - nothing needs installing. The repo is already installed in its own conda env: "
           "use /opt/conda/envs/testbed/bin/python -m pytest ... (or the PROBE_PY path). A missing "
           "dependency is the environment, not your patch: still write the fix and submit.")

REP_MSG = ("[harness] blocked: you already ran this EXACT command (identical output) - repeats "
           "spend the turn budget your fix needs. Decide from what you have read and ACT: "
           + _WRITE_HOWTO)

IMIT_MSG = ("[harness] blocked: that is the harness's own status command and it changes nothing - "
            "your diff is still empty. Write the source edit now: " + _WRITE_HOWTO)

WRITE_MSG = ("[harness] blocked: many turns read, diff STILL empty, so read-only commands are "
             "paused until you write. Writing an edit or running a test unblocks immediately. "
             + _WRITE_HOWTO)

WRITE_HINT = ("[harness] {n} read-only commands, diff STILL EMPTY - the grader scores the diff: "
              "write the fix in this next command and verify it after.")

TEST_BLOCK_MSG = ("[harness] blocked: that writes to a TEST file. The grader checks out its own "
                  "test files, so this edit is DISCARDED - and a test that passes only because you "
                  "changed it hides a wrong fix. Put the behaviour change in the SOURCE. (Blocks "
                  "once; re-run it if you truly need it.)")

HINT_NOMOD = ("[harness] this interpreter lacks that module and nothing can be installed (no "
              "network). The repo's env is {py}: re-run as `{py} -m pytest ...` / `{py} -c ...`.")
HINT_NET = ("[harness] no network: pip/conda installs always fail. The dependencies already "
            "exist in {py} (`{py} -m pip list` to check).")
HINT_NODEID = ("[harness] that node id is not in the working tree - the grader ADDS its new tests "
               "itself. Run the existing covering test file and prove the issue's behaviour with a "
               "short script in /tmp.")

MAX_INJ = 6
FORCE_RO = 34          # above this sweep's worst PASSING streak (32), below the failing 47

REVIEW_TOOL = {
    "name": "review_patch",
    "description": ("Pre-submit self-check: runs `git diff` and reports as short bullets which requirements "
                    "of the ISSUE the patch misses (wrong names/defaults/messages, missing code "
                    "paths, sibling implementations or callers needing the same change, unhandled "
                    "empty/blank/None inputs, an introduced crash). Call it instead of `submit` "
                    "when you think the fix is done. The harness calls it on your first "
                    "submit."),
    "input_schema": {"type": "object", "properties": {}, "required": []},
}

CHECKLIST_FALLBACK = (
    "- check EVERY sentence of the issue against the diff: names, defaults, messages, return "
    "values must be in the source (not in tests)\n"
    "- grep for other callers/implementations of the symbol you changed and change them too\n"
    "- re-run the WHOLE covering test file (not one node id) with the project interpreter")

ANALYST_HEAD = (
    "You are a second pair of eyes on a repo task: an engineer edited /testbed and their test run "
    "has failed TWICE on the same failure. You see the issue, their edit commands, and the failing "
    "output. Do not restate the error - give the most likely ROOT CAUSE in the SOURCE. The grader "
    "checks out its OWN test files, so NEVER advise editing or deleting a test: a test asserting "
    "the behaviour the issue removes is expected to fail and is not their bug. At most 4 short "
    "bullets, each naming a source file/symbol; end with one line `NEXT: <one shell command>`.\n\n")

ANALYST_TAIL = ("\n\nNow the bullets, then the NEXT: line.")

DRAFT_TOOL = {
    "name": "draft_patch",
    "description": ("Last-resort scaffold, run when the episode is ending with an EMPTY git diff "
                    "(an automatic 0): drafts one minimal source edit for the issue and applies it "
                    "so there is something to grade. No arguments. Call it yourself if you are "
                    "stuck with nothing written, then refine what it wrote."),
    "input_schema": {"type": "object", "properties": {}, "required": []},
}

DRAFT_HEAD = (
    "Produce ONE minimal source edit that moves this repo toward implementing the issue. Below: "
    "the issue, then the files the engineer spent their time reading (with excerpts). Output ONLY "
    "a single shell command (a python heredoc that rewrites part of a file, or a sed -i), no "
    "prose, no fences. At most ~15 lines, syntactically valid, exact names from the issue, never a "
    "file under tests/ or named test_*.py. Pick the change most likely to be what the issue asks.\n\n")

DRAFT_GUARD = ("rm ", " mv ", "git ", "pip ", "conda ", "curl ", "wget ", "chmod ", "shred ")

_STOP = set(
    "the a an and or of to in is are was were be been for on with that this it its from as at by "
    "not no but if then than so such can could will would shall should may might must do does did "
    "have has had you your we our they their there here when where which who what how why also "
    "more most other some any each both new old current existing add change changes changed make "
    "using used use via test tests issue fix fixes bug problem kwarg kwargs default defaults "
    "param params".split())


def _issue_gaps(issue, diff_text):
    """Names the ISSUE states that appear NOWHERE in the diff: the 'fixed only the symptom I
    reproduced' miss. Deterministic - the metered reviewer used to catch these and failed 78/110."""
    if not issue:
        return ""
    dl = (diff_text or "").lower()
    cand = []
    for rx in (r"[`'\"]([A-Za-z_][\w.\-]{2,40})[`'\"]",
               r"\b([A-Za-z_][A-Za-z0-9_]{3,32})\b"):
        for m in re.finditer(rx, issue):
            t = m.group(1)
            if len(t) < 4 or t.lower() in _STOP or t in cand:
                continue
            cand.append(t)
    miss = []
    for t in cand:
        key = t.split(".")[-1].lower()
        if len(key) < 4 or key in dl:
            continue
        miss.append(t)
        if len(miss) >= 9:
            break
    return "\n".join("- " + t for t in miss)[:900] if miss else ""


def _selfcheck_cmd(diff_text):
    """Grep command built from the diff (changed files + symbols) so the agent looks for
    sibling implementations/callers - how a patch passes its own repro yet fails hidden tests."""
    files = [m.group(1) for m in re.finditer(r"^\+\+\+ b/(\S+)", diff_text, re.M)][:6]
    syms = []
    for m in re.finditer(r"^[+-]\s*(?:def|class)\s+([A-Za-z_]\w*)", diff_text, re.M):
        if m.group(1) not in syms:
            syms.append(m.group(1))
    dirs = []
    for f in files:
        d = f.rsplit("/", 1)[0] if "/" in f else "."
        if d and d not in dirs:
            dirs.append(d)
    pat = "|".join((syms[:5] + [f.split("/")[-1].rsplit(".", 1)[0] for f in files[:2]]) or ["."])
    q = chr(39)
    return ("cd /testbed && grep -rn -E " + q + pat + q + " --include=" + q + "*.py" + q
            + " " + " ".join(dirs[:3] or ["."]) + " | head -n 40")


def red_analyse(state):
    """True once per stuck episode (max 2): red run after a red run, patch exists."""
    return (state.get("red") and _g(state, "redruns", 0) >= 2
            and _g(state, "analyst", 0) < 2 and bool(state.get("issue")))


def _fail_excerpt(obs):
    """Verdict + last traceback from a test-run observation."""
    lines = obs.splitlines()
    keep = [l for l in lines if l.strip().startswith(("E ", "FAILED", "ERROR", "assert"))
            or "Error" in l or "error" in l][:40]
    tail = "\n".join(lines[-60:])
    return ("\n".join(keep) + "\n--- tail ---\n" + tail)[:3500]


def _write_targets(cmd):
    """Paths a command writes to: feeds only 'is this an edit' + 'is it a test file'."""
    out = []
    for rx in (_REDIRECT_RE, _TEE_RE, _OPEN_W_RE):
        for m in rx.finditer(cmd):
            out.append(m.group(1))
    if _SED_I_RE.search(cmd) or _PERL_I_RE.search(cmd) or _PATCH_RE.search(cmd):
        for m in _EXT_RE.finditer(cmd):
            out.append(m.group(1))
    return [t for t in out if t and len(t) < 200]


def _is_test_path(path):
    return bool(_TESTPATH_RE.search(path.replace("\\", "/")))


def _block_text(b):
    if not isinstance(b, dict):
        return ""
    return str(b.get("text") or b.get("thinking") or b.get("reasoning") or "")


def _runs_interpreter(cmd, py):
    if py and py in cmd:
        return False
    return bool(_CMD_PY_RE.search(cmd))


def _g(state, key, default):
    v = state.get(key, default)
    return default if v is None else v


class Hooks(BaseHooks):

    loop = {
        "nudge_no_tool": ("Nothing happened because no tool was called. Call the bash tool with "
                          "ONE command that ACTS (edit a file or run a test), or `submit` once the "
                          "fix is written and verified. Never describe a command in prose."),
        "nudge_bad_markup": ("That tool call did not parse. Emit a native bash tool call holding "
                             "one shell command (no XML/JSON markup in your text); write large "
                             "files through several small calls."),
    }

    def before_llm(self, msgs, state):
        if state.get("issue"):
            return msgs
        best = ""
        for m in (msgs or []):
            if not isinstance(m, dict) or m.get("role") != "user":
                continue
            c = m.get("content")
            if isinstance(c, list):
                c = "\n".join(_block_text(b) for b in c)
            if not isinstance(c, str):
                continue
            if c.startswith("[harness]"):
                continue
            if len(c) >= 200:
                best = c
                break
            if not best and len(c) > 60:
                best = c
        if best:
            state["issue"] = best[:6000]
        return msgs

    def before_tool(self, name, args, state):
        if name != "bash":
            return args
        cmd = (args or {}).get("command") or ""
        if not isinstance(cmd, str) or not cmd.strip():
            return args

        if not state.get("probed"):
            state["probed"] = 1
            args["command"] = PROBE + cmd
            return args

        bare = _DEVNULL_RE.sub(" ", cmd)
        targets = _write_targets(bare)
        real_targets = [t for t in targets if not _SCRATCH_RE.search(t)]
        is_write = bool(real_targets) or bool(
            _SED_I_RE.search(cmd) or _PERL_I_RE.search(cmd) or _PATCH_RE.search(cmd)
            or (_WRITEFN_RE.search(cmd) and "open(" in cmd))
        is_test_run = bool(_TESTRUN_RE.search(cmd))
        empty_diff = not state.get("wrote") and not state.get("has_diff")

        if _INSTALL_RE.search(cmd) and _g(state, "pipblk", 0) < 2:
            state["pipblk"] = _g(state, "pipblk", 0) + 1
            return ("no network in this container", PIP_MSG)

        if _g(state, "tblock", 0) < 1 and not _INSTALL_RE.search(cmd):
            for tgt in targets:
                if _is_test_path(tgt) and not _SCRATCH_RE.search(tgt):
                    state["tblock"] = _g(state, "tblock", 0) + 1
                    return ("test files are replaced by the grader", TEST_BLOCK_MSG)

        if "[harness] diff state" in cmd and _g(state, "iblk", 0) < 3:
            state["iblk"] = _g(state, "iblk", 0) + 1
            return ("harness status command", IMIT_MSG)

        norm = " ".join(cmd.split())
        cnt = state.setdefault("cnt", {})
        rblk = state.setdefault("repblk", {})

        if not is_write and cnt.get(norm, 0) >= 3:
            b = rblk.get(norm, 0)
            rblk[norm] = b + 1
            if b % 4 != 3:
                return ("repeated identical command", REP_MSG)

        if is_write:
            state["wrote"] = 1
            state["ro"] = 0
            state["tested"] = 0
            wlog = state.setdefault("wlog", [])
            wlog.append(cmd[:900])
            if len(wlog) > 3:
                del wlog[0]
        elif is_test_run:
            state["ro"] = 0
            state["tested"] = 1
        else:
            state["ro"] = _g(state, "ro", 0) + 1
            rc = state.setdefault("readcnt", {})
            for m in _READPATH_RE.finditer(cmd):
                p = m.group(1)
                if "/" not in p and "." not in p:
                    continue
                rc[p] = rc.get(p, 0) + 1
            if len(rc) > 120:
                state["readcnt"] = dict(sorted(rc.items(), key=lambda kv: -kv[1])[:60])
            if empty_diff and state["ro"] >= FORCE_RO:
                return ("read-only spiral, diff empty", WRITE_MSG)

        cnt[norm] = cnt.get(norm, 0) + 1
        if len(cnt) > 400:
            state["cnt"] = {norm: 1}
        return args

    def after_tool(self, name, args, obs, state):
        if name != "bash" or not isinstance(obs, str) or not obs:
            return obs
        cmd = (args or {}).get("command") or ""
        py = state.get("py") or "/opt/conda/envs/testbed/bin/python"

        if not state.get("py"):
            for m in _PROBE_PY_RE.finditer(obs):
                state["py"] = m.group(1)
                break

        if "diff --git " in obs[:3000] and "+++ " in obs[:3000]:
            state["has_diff"] = 1
        if "[end diff state]" in obs or "diff state:" in cmd:
            body = obs.split("diff state:", 1)[-1].split("[end diff state]", 1)[0]
            state["has_diff"] = 1 if [l for l in body.splitlines() if l.strip()] else 0
        if _TESTRUN_RE.search(cmd):
            state["tested"] = 1
            red = 1 if (re.search(r"\d+ failed", obs)
                        or (" short test summary" in obs and " failed" in obs)) else 0
            state["red"] = red
            state["redruns"] = _g(state, "redruns", 0) + 1 if red else 0

        hints = []
        if _NETFAIL_RE.search(obs):
            hints.append(HINT_NET.format(py=py))
        elif (_NOMOD_RE.search(obs) or _NOTFOUND_BIN_RE.search(obs)) and _runs_interpreter(cmd, py):
            hints.append(HINT_NOMOD.format(py=py))
        if "ERROR: not found" in obs and "::" in cmd:
            hints.append(HINT_NODEID)
        if obs.strip() and obs.strip()[:2500] == state.get("last_obs"):
            hints.append("[harness] identical output to your previous command - change something "
                         "before running it again.")
        state["last_obs"] = obs.strip()[:2500]

        ro = _g(state, "ro", 0)
        if not state.get("wrote") and not state.get("has_diff") and ro >= 20 and (ro - 20) % 8 == 0:
            hints.append(WRITE_HINT.format(n=ro))

        head = ""
        if len(obs) > 1500 and (" passed" in obs or " failed" in obs or "ERROR" in obs):
            keep = []
            for line in obs.splitlines():
                s = line.strip()
                if (s.startswith(("FAILED", "ERROR"))
                        or re.match(r"^=+ .*(passed|failed|error)", s)
                        or re.match(r"^\d+ (?:failed|passed|error)", s)):
                    keep.append(s[:150])
                if len(keep) >= 6:
                    break
            if keep:
                head = "[harness digest] " + " || ".join(keep) + "\n"

        if red_analyse(state):
            state["analyst"] = _g(state, "analyst", 0) + 1
            prompt = (ANALYST_HEAD + "ISSUE:\n" + (state.get("issue") or "")[:3000]
                      + "\n\nEDITS RUN SO FAR:\n" + "\n".join(state.get("wlog") or [])[:2500]
                      + "\n\nFAILING TEST OUTPUT:\n" + _fail_excerpt(obs) + ANALYST_TAIL)
            try:
                out = self.llm(prompt, max_tokens=500)
            except Exception:
                out = ""
            if isinstance(out, str) and out.strip() and not out.strip().startswith("ERROR"):
                head = ("[harness analyst] two red runs - a fresh look at your failure:\n"
                        + out.strip()[:1500] + "\n" + head)

        pre = "".join(h + "\n" for h in hints)
        if not head and not pre:
            return obs
        return head + pre + obs

    def after_llm(self, content, state):
        """Rescues on the only surface that sees the raw reply: (a) empty-diff submit ->
        reality check; (b) G7 scaffold when the episode heads for an empty diff; (c) the
        completeness review on the first non-empty submit; (d) tool-less reply -> a real bash
        check. The diff is graded at episode end either way, so no gate can lose the patch."""
        blocks = [b for b in (content or []) if isinstance(b, dict)]
        inj = _g(state, "inj", 0)
        has_tool = any(b.get("type") == "tool_use" for b in blocks)
        submits = [b for b in blocks if b.get("type") == "tool_use" and b.get("name") == "submit"]

        if submits and not state.get("wrote") and not state.get("has_diff") \
                and _g(state, "sgate", 0) < 2:
            state["sgate"] = _g(state, "sgate", 0) + 1
            return [{"type": "text", "text": GATE_EMPTY},
                    {"type": "tool_use", "id": "harness_gate_%d" % state["sgate"],
                     "name": "bash", "input": {"command": INJECT_CMD}}]

        turn = _g(state, "turn", 0)
        dead_end = (not state.get("wrote")) and (not state.get("has_diff")) \
            and _g(state, "draft", 0) < 1 and bool(state.get("issue")) \
            and (bool(submits) and _g(state, "sgate", 0) >= 1
                 or _g(state, "inj", 0) >= 3 or _g(state, "iblk", 0) >= 2 or turn >= 45)
        if dead_end and (submits or not has_tool):
            state["draft"] = 1
            return [{"type": "text", "text": DRAFT_NOTE},
                    {"type": "tool_use", "id": "harness_draft_1",
                     "name": "draft_patch", "input": {}}]

        if submits and (state.get("wrote") or state.get("has_diff")) \
                and _g(state, "rev", 0) < 1 and turn < 70:
            state["rev"] = 1
            return [{"type": "text", "text": GATE_REVIEW},
                    {"type": "tool_use", "id": "harness_review_1",
                     "name": "review_patch", "input": {}}]

        if has_tool or inj >= MAX_INJ:
            return content
        text = "\n".join(_block_text(b) for b in blocks)
        if "```" in text or "<bash" in text.lower():
            note = NOTE_PROSE
        elif state.get("has_diff") or state.get("wrote"):
            note = NOTE_PATCH
        else:
            note = NOTE_EMPTY
        state["inj"] = inj + 1
        kept = []
        for b in blocks:
            t = _block_text(b).strip()
            if t:
                kept.append({"type": "text", "text": t[:1000]})
        kept.append({"type": "text", "text": note})
        kept.append({"type": "tool_use", "id": "harness_state_%d" % (inj + 1),
                     "name": "bash", "input": {"command": INJECT_CMD}})
        return kept

    def extra_tools(self):
        return [REVIEW_TOOL, DRAFT_TOOL]

    def run_tool(self, name, args, env, state):
        if name == "draft_patch":
            if _g(state, "draftruns", 0) >= 1:
                return ("[harness] the drafter already ran - the diff is yours now: fix it and "
                        "submit, or write your own edit: " + _WRITE_HOWTO)
            state["draftruns"] = 1
            return self._draft_patch(env, state)
        if name != "review_patch":
            return "ERROR: unknown tool " + str(name)
        runs = _g(state, "revruns", 0) + 1
        state["revruns"] = runs
        if runs > 2:
            return ("[harness] review already run twice - stop asking, act: fix the bullets you "
                    "agree with, then `submit`.")
        py = state.get("py") or "/opt/conda/envs/testbed/bin/python"
        try:
            raw = env.bash("cd /testbed && git --no-pager diff --stat | tail -n 15; "
                           "echo '===== PATCH ====='; git --no-pager diff | head -c 11000")
        except Exception as e:
            return "[harness] review unavailable (" + repr(e) + "); run `git --no-pager diff` " \
                   "and check it against every sentence of the issue yourself."
        if not isinstance(raw, str) or "diff --git" not in raw:
            return ("[harness] review: your `git diff` is EMPTY - there is nothing to review and "
                    "nothing to grade. " + _WRITE_HOWTO)
        gaps = _issue_gaps(state.get("issue") or "", raw)
        if gaps:
            return ("[harness completeness check] these names appear in the ISSUE but NOWHERE in "
                    "your `git diff`:\n" + gaps
                    + "\nFor each one, decide from the issue text whether the SOURCE must contain "
                      "it; if it must, add it now. Then look for sibling paths/callers needing the "
                      "same change (run this in ONE bash call):\n" + _selfcheck_cmd(raw)
                    + "\n" + CHECKLIST_FALLBACK
                    + "\nThen re-run the WHOLE covering test file with " + py
                    + " -m pytest -q <file> and `submit` (the second submit goes through).\n\n"
                    + "Diff stat:\n" + raw[:900])
        return ("[harness] review: the diff mentions nothing the issue does not, and nothing the "
                "issue states is missing from it by name. Still tick off every sentence of the "
                "issue against `git --no-pager diff`: names, defaults, error messages, return "
                "values, empty/blank/None inputs. " + CHECKLIST_FALLBACK
                + "\nRun the WHOLE covering test file with " + py + " -m pytest -q <file>, then "
                "`submit` (the second submit goes through).\nDiff stat:\n" + raw[:1200])

    def _draft_patch(self, env, state):
        issue = state.get("issue") or ""
        rc = state.get("readcnt") or {}
        tops = [p for p, _ in sorted(rc.items(), key=lambda kv: -kv[1])[:4] if "tests/" not in p]
        if not tops:
            tops = ["README.rst", "README.md", "setup.py"]
        quoted = " ".join(p.replace("'", "") for p in tops)
        try:
            ctx = env.bash("cd /testbed && for f in " + quoted + "; do [ -f \"$f\" ] && "
                           "{ echo \"=== $f\"; sed -n '1,140p' \"$f\"; }; done 2>/dev/null | "
                           "head -c 9000")
        except Exception:
            return "[harness] draft failed to read the repo - write ANY source edit now: " + _WRITE_HOWTO
        try:
            cmd = self.llm(DRAFT_HEAD + "ISSUE:\n" + issue[:4000] + "\n\nFILES:\n"
                           + (ctx or "")[:9000], max_tokens=700)
        except Exception:
            cmd = ""
        if not isinstance(cmd, str) or not cmd.strip() or cmd.strip().startswith("ERROR"):
            return ("[harness] the drafter is unavailable - the empty-diff score is still 0, so "
                    "write the most plausible fix yourself NOW: " + _WRITE_HOWTO)
        c = cmd.strip()
        if "```" in c:
            parts = c.split("```")
            c = parts[1] if len(parts) > 1 else c
        c = c.strip()
        low = " " + c.lower()
        write_ok = ("open(" in c and "w" in c) or "sed -i" in c or "cat >" in c or "write_text" in c
        if len(c) > 4000 or not write_ok or "test_" in c or any(g in low for g in DRAFT_GUARD):
            return ("[harness] the drafter produced something unsafe/unusable - write the edit "
                    "yourself in this next command: " + _WRITE_HOWTO)
        try:
            env.bash(c)
            stat = env.bash("cd /testbed && git --no-pager diff --stat | tail -n 5")
        except Exception:
            return "[harness] draft could not be applied - write the edit yourself: " + _WRITE_HOWTO
        if not stat or "file" not in stat:
            return ("[harness] the draft did not change any tracked file - write the edit "
                    "yourself in this next command: " + _WRITE_HOWTO)
        state["wrote"] = 1
        state["has_diff"] = 1
        return ("[harness] a first draft is now in your working tree:\n" + stat
                + "\nThat was auto-generated and is probably wrong or partial. Read `git "
                  "--no-pager diff`, make it actually implement the issue, run the covering test "
                  "file, then `submit`.")

    def on_turn_end(self, turn, state):
        t = int(turn)
        state["turn"] = t
        if not state.get("wrote") and not state.get("has_diff"):
            if t in (10, 16, 24, 34, 46, 60, 75):
                return ("no source file has been modified yet (turn %d). The grader scores the git "
                        "diff, so an empty patch is an automatic 0. Apply your edit to the source "
                        "NOW in one bash call, then verify it with the project interpreter." % t)
            return None
        if state.get("wrote") and not state.get("tested") and t % 6 == 0 and t >= 12:
            return ("a source edit exists but NOTHING has run since it. Run the covering test file "
                    "with the project interpreter (`<env-python> -m pytest -x -q <file>`) before "
                    "you submit.")
        if state.get("red") and t % 8 == 0 and t >= 16:
            return ("the last test run was RED. If the failure is in behaviour your change was "
                    "supposed to produce, fix the source; if the failing test asserts the OLD "
                    "behaviour the issue asks to remove, it is expected (the grader uses its own "
                    "tests) - but the OTHER tests in that file must pass.")
        if t >= 80 and t % 10 == 0:
            return ("wrap up soon: make sure `git --no-pager diff` contains your source fix and "
                    "call `submit`.")
        return None
