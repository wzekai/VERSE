"""harness_code/hooks.py (round 6): verification must come BEFORE the patch, while it can
still be red. `spec` = one metered call reading the ISSUE TEXT ONLY -> /tmp/spec_test.py,
written and run at HEAD; before_tool blocks the first source edit once and sends the agent
there. `verify` re-runs the episode's checks and prints PROVEN / UNPROVEN / BROKEN / NOT YET;
a still-red check re-arms the submit->verify gate. Contract names come from the issue PROSE.
"""
import re
import shlex

_ENV = (
    'if [ -z "${HARNESS_PY:-}" ]; then for d in /opt/conda/envs/testbed/bin '
    '/opt/conda/envs/*/bin /testbed/.venv/bin /venv/bin; do if [ -x "$d/python" ]; then '
    'export HARNESS_PY="$d"; break; fi; done; fi; '
    'if [ -n "${HARNESS_PY:-}" ]; then export PATH="$HARNESS_PY:$PATH"; fi; '
)
_PREFIX = re.compile(r"^\s*(cd\s+\S+\s*(&&|;)\s*)?")
_OFFLINE = re.compile(r"\bpip[3]?\s+(install|download)\b|\bpython[3]?\s+-m\s+pip\b"
                      r"|conda\s+install|\bpoetry\s+install|\buv\s+pip\b", re.I)
_NOMOD = re.compile(r"No module named ['\"]?([\w.]+)")
_NOTFOUND = re.compile(r"ERROR: not found|no match in any of", re.I)
_NODEID = re.compile(r"([\w/\-\.]+\.py(?:::[\w\-\[\]\.\$:=%,]+)?)")
_WRITE = re.compile(r"sed\s+-i|cat\s*>|tee\s|git\s+apply|apply_patch|write_text"
                    r"|<<\s*.?EOF", re.I)
_RUN = re.compile(r"python[3]?\s+\S*\.py|python[3]?\s+-m\s+(pytest|unittest)|pytest"
                  r"|py\.test", re.I)
_TMPCHK = re.compile(r"/tmp/[\w/\-\.]+\.py")
_PYTEST = re.compile(r"pytest|py\.test|-m unittest")
_JUNK = r"(^|/)(build|dist|node_modules|site-packages|\.venv|venv|__pycache__|\.git)(/|$)"
_FAILID = re.compile(r"^(?:FAILED|ERROR) (\S+)", re.M)
_FAILROW = re.compile(r"\b\d+ (?:failed|errors?)\b|^FAILED|^ERROR", re.M)
_STOP = set(("self true false none null cls def import from return class the and for with "
             "this that python pytest unittest test tests file files name names value "
             "values type types string list dict int float bool str bytes args kwargs error "
             "errors line lines code codes api cli url http https if else elif try except "
             "raise get set run main src lib example examples issue bug fix fixed works "
             "work used using note block comment comments output input default defaults key "
             "keys item items data path paths method methods function functions argument "
             "arguments attribute attributes option options flag flags property properties "
             "message messages").split())
_SPEC_PROMPT = (
    "You get ONLY a bug report / feature request (no repository, no code). Write the "
    "smallest pytest file that FAILS on the un-fixed code and PASSES once the report is "
    "honoured.\n"
    "- Assert only what the report states: the literal names it uses (module, function, "
    "class, keyword argument, attribute, exception type) and its own example.\n"
    "- Import with the paths the report itself shows. Never invent a path, a fixture, a "
    "data file or the network.\n"
    "- If the report gives an expected value, assert it; otherwise assert only the SHAPE "
    "(no exception raised, attribute/key exists) - never invent a number.\n"
    "- 8-20 lines, plain pytest functions, no conftest, no parametrize.\n"
    "- Reply with the file content only: no prose, no markdown fences."
)


def _norm(cmd):
    c = _PREFIX.sub("", cmd or "").strip()
    return re.sub(r"\s+", " ", c).lower().strip(" ;&")


def _contract(issue):
    """Identifiers the issue promises, from its PROSE - its code samples are full of local
    variable names that no test will ever call."""
    prose = re.sub(r"```.*?```", " ", issue or "", flags=re.S)
    prose = re.sub(r"(?m)^(?: {4}|\t).*$", " ", prose)
    out = []
    for tok in (re.findall(r"`([^`\n]{2,40})`", prose)
                + re.findall(r"'([A-Za-z_][\w.]{2,32})'", prose)):
        for w in re.findall(r"[A-Za-z_][\w.]{1,32}", tok.strip()):
            low = w.lower()
            if (low in _STOP or low.startswith("test") or w in out or w.count(".") > 2
                    or re.search(r"\d{2,}", w)):
                continue
            out.append(w)
    return out[:12]


def _card(syms):
    if not syms:
        return ""
    return ("[CONTRACT CARD -- built by the harness from the ISSUE text, not from the repo]"
            "\nIdentifiers the issue names in prose; the hidden tests are written against "
            "THESE, not against the tests already in the repo: " + ", ".join(syms) + "."
            "\n1. Implement each LITERALLY: spelling, module, argument order, defaults, "
            "exception type, message wording. Repo tests passed BEFORE your change, so a "
            "green run proves nothing; evidence = a check that FAILS on the un-fixed code "
            "(`spec` writes one from the issue text alone)."
            "\n2. `verify` is deterministic and starts with your DIFFERENTIAL: PROVEN = a "
            "check red at HEAD is green now; UNPROVEN = nothing failed at HEAD; NOT YET = "
            "still red - fix it before submitting.")


class Hooks(BaseHooks):
    loop = {
        "obs_cap": 6144,
        "keep_full_turns": 10,
        "spam_streak": 6,
        "nudge_no_tool": ("Act with a tool: bash, `spec` (independent check from the issue "
                          "text alone), or `verify`. If the patch is verified, submit."),
        "nudge_bad_markup": ("That tool call did not parse. Re-send it as a native bash "
                             "tool call. Write big files in small pieces: several small "
                             "heredocs or printf appends."),
    }

    # ------------------------------------------------------------ prompt facts
    def system_prompt(self, assembled):
        facts = [
            "ENVIRONMENT FACTS (verified - do not re-derive them):",
            "- OFFLINE container: pip install / pip download can never work. Everything is "
            "already installed in the repo env, which the harness puts first on PATH: "
            "`cd /testbed && python -m pytest <file> -q`.",
            "- The grading tests are HIDDEN and are an UPDATED version of a test file that "
            "already exists here. `ERROR: not found: ...::test_x` means that node id is not "
            "in your repo: never grep for it.",
            "THE ORDER THAT MATTERS (a green local run predicts nothing: the last verify was "
            "green in 61/75 FAILED and 30/35 PASSED episodes; 0 of 110 episodes wrote a "
            "check before their first edit):",
            "1. `spec` - ONE call; it reads the ISSUE TEXT ONLY (no repo, no diff), writes "
            "/tmp/spec_test.py asserting what the report literally promises, and runs it. "
            "Call it BEFORE your first edit so you see it RED - that failure is what tells "
            "you what the un-fixed code lacks.",
            "2. Optionally rewrite it against the report's own example as "
            "/tmp/contract_test.py - it must fail too.",
            "3. EDIT small (passing patches median 1 file / +4 lines; a plausible patch "
            "beats an empty diff), then `verify`: it re-runs both checks + the covering "
            "tests and prints the differential verdict.",
            "4. Submit only on PROVEN, or after reading a NOT YET list. Never re-run a "
            "command that returned nothing.",
        ]
        return assembled.rstrip() + "\n\n" + "\n".join(facts)

    # ------------------------------------------------------ contract pinning
    def before_llm(self, msgs, state):
        try:
            if not msgs:
                return msgs
            first = msgs[0]
            if not (isinstance(first, dict) and isinstance(first.get("content"), str)):
                return msgs
            cur = first["content"]
            if "issue" not in state:
                state["issue"] = cur
                state["contract"] = _contract(cur)
            card = _card(state.get("contract") or [])
            if card and "[CONTRACT CARD" not in cur:
                new = [dict(m) for m in msgs]
                new[0]["content"] = cur + "\n\n" + card
                return new
        except Exception:
            return msgs
        return msgs

    # --------------------- completion gate + dead-turn resurrection
    def after_llm(self, content, state):
        try:
            if not isinstance(content, list):
                return content
            blocks = [b for b in content if isinstance(b, dict)]
            tools = [b for b in blocks if b.get("type") == "tool_use"]
            texts = [b for b in blocks if b.get("type") == "text"
                     and (b.get("text") or "").strip()]
            keep = [b for b in blocks if b.get("type") in ("text", "tool_use")]
            armed = ((state.get("writes") or state.get("spec_red"))
                     and (state.get("still_red") or not state.get("verified"))
                     and state.get("gates", 0) < 2)
            if not tools and not texts:
                state["dead"] = state.get("dead", 0) + 1
                if state["dead"] >= 2 and (state.get("writes") or state.get("specced")):
                    state["dead"] = 0
                    state["gates"] = state.get("gates", 0) + 1
                    state["verified"] = 1
                    return keep + [{"type": "tool_use",
                                    "id": "harness-rescue-" + str(len(content)),
                                    "name": "verify", "input": {}}]
                return content
            state["dead"] = 0
            if not tools:
                # a text-only reply ENDS the episode and the diff is graded as-is
                if armed and len(" ".join(b.get("text", "") for b in texts)) > 40:
                    state["gates"] = state.get("gates", 0) + 1
                    state["verified"] = 1
                    return keep + [{"type": "tool_use",
                                    "id": "harness-verify-" + str(len(content)),
                                    "name": "verify", "input": {}}]
                return content
            if armed:
                for b in tools:
                    if b.get("name") == "submit":
                        state["gates"] = state.get("gates", 0) + 1
                        state["verified"] = 1
                        b["name"] = "verify"
                        b["input"] = {}
                        break
            return content
        except Exception:
            return content

    # ------------------------------------------------------ per-command guards
    def before_tool(self, name, args, state):
        if name != "bash":
            return args
        cmd = (args or {}).get("command")
        if not isinstance(cmd, str) or not cmd.strip():
            return args
        body = re.sub(r"^\s*cd\s+\S+\s*(&&|;)\s*", "", _PREFIX.sub("", cmd).strip())
        no_tmp = re.sub(r"/tmp/\S+", " ", cmd)

        # 1. offline container (round 1: 67 attempts, 0 successes)
        if _OFFLINE.search(body) and "-e ." not in body:
            return ("offline container",
                    "[harness] BLOCKED: this container has no network, so pip install can "
                    "never succeed. Everything is installed in the repo environment: "
                    "cd /testbed && python -m pytest <file> -q. Install nothing.")

        # 2. repetition loop breaker
        key = _norm(cmd)
        hist = state.setdefault("hist", {})
        hist[key] = hist.get(key, 0) + 1
        if hist[key] >= 3:
            return ("repeated command",
                    "[harness] You have run this EXACT command " + str(hist[key]) +
                    " times; its output cannot change. Instead: (a) call `verify`, or "
                    "(b) write the fix NOW with a small python heredoc that reads the "
                    "file, replaces the old text and writes it back, then verify, submit.")

        # 3. THE ONE BLOCK. The first source edit is where evidence dies: 0/110 episodes
        #    wrote a check before editing, so every check afterwards only asserted what the
        #    agent's own patch already did. Spend one call on an independent reading.
        if (_WRITE.search(no_tmp) and not state.get("writes")
                and not state.get("specced") and not state.get("spec_blocked")):
            state["spec_blocked"] = 1
            return ("spec first",
                    "[harness] ONE call before you edit (once per episode, never again): "
                    "the `spec` tool. It reads the ISSUE TEXT ONLY - no repo, no diff - "
                    "writes /tmp/spec_test.py and runs it. After your edit, no check can "
                    "ever show you what the UN-FIXED code does, and a green suite is worth "
                    "nothing: it was green before your patch too (61 of 75 failed episodes "
                    "ended green). Call `spec` now, see which assertion fails at HEAD, "
                    "then edit and make it pass. Or write /tmp/contract_test.py from the "
                    "report's example and run it first - it must FAIL before the fix.")

        # 4. evidence bookkeeping
        if _WRITE.search(no_tmp):
            state["writes"] = state.get("writes", 0) + 1
            if state.get("open"):
                state["open"] = []
                state["verified"] = 0
        if _PYTEST.search(cmd):
            state["pytest"] = state.get("pytest", 0) + 1
        if _TMPCHK.search(cmd) and _RUN.search(cmd):
            state["repro"] = 1

        # 5. mechanically use the repo interpreter, never the conda base one
        if "HARNESS_PY" not in cmd:
            new = dict(args)
            new["command"] = _ENV + cmd
            return new
        return args

    # -------------------------------- independent contract check (1 metered call)
    def _spec_run(self, env, state):
        if state.get("specced"):
            return ("(spec ran already this episode - re-run it yourself: cd /testbed && "
                    "python -m pytest /tmp/spec_test.py -q)\n" + (state.get("spec_out") or ""))
        issue = (state.get("issue") or "").strip()
        if len(issue) < 40:
            return "[spec] the issue text is not available yet - nothing to read."
        state["specced"] = 1
        raw = self.llm(_SPEC_PROMPT + "\n\n--- REPORT ---\n" + issue[:9000], 900) or ""
        code = "\n".join(ln for ln in raw.split("\n")
                         if not re.match(r"^\s*(```|###|\$|>>>)", ln)).strip("\n")
        code = code.replace("SPECEOF", "SPECX")[:3800]
        if "import" not in code and "def " not in code:
            return ("[spec] the independent reading produced nothing runnable; ignored. "
                    "Write your own /tmp/contract_test.py from the report's example.")
        env.bash("cat > /tmp/spec_test.py <<'SPECEOF'\n" + code + "\nSPECEOF\n"
                 "echo WROTE $(wc -l < /tmp/spec_test.py)")
        out = env.bash(_ENV + "cd /testbed && timeout 300 python -m pytest "
                       "/tmp/spec_test.py -q --no-header -p no:cacheprovider "
                       "--tb=short 2>&1 | tail -25") or ""
        names = sorted(set(re.findall(r"\b(?:from|import)\s+([\w.]+)", code)
                           + re.findall(r"\b([A-Z][A-Za-z0-9_]{3,})\b", code)))[:8]
        if re.search(r"ModuleNotFoundError|ImportError|SyntaxError|collected 0 items"
                     r"|no tests ran|error collecting", out, re.I):
            return ("[spec] the independent reading could not be imported here, so it is a "
                    "NAME LIST only - what it expected the hidden test to touch: "
                    + ", ".join(names) + "\nRewrite or delete /tmp/spec_test.py; do not "
                    "chase these errors.")
        errs = re.findall(r"^E\s+.*$", out, re.M)[:5]
        if _FAILROW.search(out):
            state["spec_red"] = 1
            if not state.get("writes"):
                state["head_red"] = 1
            state["spec_out"] = (
                "[spec] RED - an INDEPENDENT reading of the report (it never saw the repo "
                "or your diff) does not hold on this code:"
                "\n    " + "\n    ".join(errs or [out.strip()[-350:]]) +
                "\nThis is evidence you could not make for yourself: names/behaviours the "
                "hidden test will call."
                "\n- If the report really asks for it: implement it (exact spelling, module, "
                "argument, exception) until `python -m pytest /tmp/spec_test.py -q` passes."
                "\n- If the check misreads the report, edit /tmp/spec_test.py so it states "
                "the report literally - and then honour THAT.")
            return state["spec_out"]
        if not state.get("writes"):
            state["spec_out"] = ("[spec] GREEN at HEAD: that reading already holds without "
                                 "any change of yours, so it cannot guide your fix - keep it "
                                 "as a name list (" + ", ".join(names) + ") and write your "
                                 "own /tmp/contract_test.py for the behaviour the report "
                                 "calls BROKEN. It must fail. Then edit.")
        else:
            state["spec_out"] = ("[spec] GREEN: with your patch, an independent reading of "
                                 "the report holds. Strongest signal available offline.")
        return state["spec_out"]

    # --------------------------------------- deterministic pre-submit loop
    def extra_tools(self):
        return [
            {"name": "spec",
             "description": ("Independent contract check: ONE model call that reads the "
                             "ISSUE TEXT ONLY (no repo, no diff), writes /tmp/spec_test.py "
                             "asserting what the report literally promises, and runs it. "
                             "Call it BEFORE your first edit. No arguments."),
             "input_schema": {"type": "object", "properties": {}, "required": []}},
            {"name": "verify",
             "description": ("Deterministic pre-submit check: differential verdict on "
                             "/tmp/spec_test.py and /tmp/contract_test.py (red at HEAD -> "
                             "green now = PROVEN), patch radar, py_compile, and the test "
                             "files covering the change. No arguments."),
             "input_schema": {"type": "object", "properties": {}, "required": []}},
        ]

    def _b(self, env, state, cmd):
        """one container command, while this tool call's 16-command budget lasts"""
        try:
            if len(env.commands) - int(state.get("cmdbase") or 0) >= 13:
                return ""
        except Exception:
            pass
        return env.bash(cmd) or ""

    def _files(self, env, state):
        raw = self._b(env, state, _ENV + "cd /testbed && git diff --name-only")
        return [ln.strip() for ln in (raw or "").split("\n") if ln.strip()]

    def _tests_for(self, env, state, files):
        raw = self._b(env, state, _ENV + "cd /testbed && git ls-files | grep -E "
                      "'(^|/)(test_[^/]*|[^/]*_test|conftest)[.]py$' | grep -vE '" +
                      _JUNK + "' | head -300")
        pool = [ln.strip() for ln in (raw or "").split("\n") if ln.strip()]
        if not pool:
            return [], ""
        stems = []
        for p in files:
            if not p.endswith(".py"):
                continue
            if p.split("/")[-1] != "__init__.py":
                stems.append(p.split("/")[-1][:-3])
            for part in re.sub(r"\.py$", "", p).replace("/", ".").split(".")[:-1][-2:]:
                if len(part) > 3 and part not in ("src", "lib"):
                    stems.append(part)
        stems = [s for s in dict.fromkeys(stems) if len(s) > 2][:8]
        picked = []
        if stems:
            raw = self._b(env, state, "cd /testbed && grep -l -E \"(" +
                          "\\|".join(stems) + ")\" " +
                          " ".join(shlex.quote(p) for p in pool[:200]) +
                          " 2>/dev/null | grep -vE '" + _JUNK + "' | head -4") or ""
            picked = [ln.strip() for ln in raw.split("\n") if ln.strip()]
        if not picked:
            for stem in stems:
                hit = [c for c in pool if stem in c]
                if hit:
                    picked = [hit[0]]
                    break
        scope = (picked[0].rsplit("/", 1)[0] if "/" in picked[0] else ".") if picked else ""
        return picked[:3], scope

    def _pt(self, env, state, targets, to=600):
        if not targets:
            return "", []
        raw = self._b(env, state, _ENV + "cd /testbed && timeout " + str(to) +
                      " python -m pytest " + " ".join(shlex.quote(t) for t in targets[:12]) +
                      " -q --no-header -p no:cacheprovider --tb=line 2>&1 | tail -35") or ""
        return raw, re.findall(_FAILID, raw)

    def _chk(self, env, state, path):
        return (self._b(env, state, _ENV + "cd /testbed && if [ -f " + path + " ]; then "
                        "timeout 300 python -m pytest " + path + " -q --no-header -p "
                        "no:cacheprovider --tb=line 2>&1 | tail -12; fi") or "").strip()

    def run_tool(self, name, args, env, state):
        try:
            try:
                state["cmdbase"] = len(env.commands)
            except Exception:
                state["cmdbase"] = 0
            if name == "spec":
                return self._spec_run(env, state)
            if name != "verify":
                return "unknown tool: " + str(name)
            files = self._files(env, state)
            if not files:
                return ("[verify] git diff is EMPTY - nothing to verify, and an empty patch "
                        "grades 0. Take the file you have already read and write your best "
                        "fix NOW, then submit.")
            state["check"] = state.get("check", 0) + 1
            state["still_red"] = 0
            out = []
            # 1. THE DIFFERENTIAL: re-run this episode's own checks
            sp = self._chk(env, state, "/tmp/spec_test.py") if state.get("specced") else ""
            ow = self._chk(env, state, "/tmp/contract_test.py")
            now = [p for p, r in (("/tmp/spec_test.py", sp), ("/tmp/contract_test.py", ow))
                   if _FAILROW.search(r)]
            if now:
                state["still_red"] = 1
                state["verified"] = 0
            head = state.get("head_red")
            if head and not now:
                out.append("[differential] PROVEN: a check that FAILED on the un-fixed code "
                           "is green now - your patch demonstrably does something. What is "
                           "left is whether it also breaks something.")
            elif now and head:
                out.append("[differential] NOT YET: " + ", ".join(now) + " failed at HEAD "
                           "(that is why it exists) and STILL FAILS - your patch does not "
                           "deliver the report's own contract. Do not submit yet.")
            elif now:
                out.append("[differential] BROKEN: " + ", ".join(now) + " is red with your "
                           "patch (that file was green at HEAD). Make it green, or rewrite "
                           "the check so it states the report literally.")
            elif not head:
                out.append("[differential] UNPROVEN: nothing this episode ran ever failed "
                           "on the UN-FIXED code, so no run of yours - however green - "
                           "shows your patch changes behaviour. Call `spec` and answer its "
                           "failing assertion, or write /tmp/contract_test.py from the "
                           "report's example. Green existing tests are the NORMAL state of "
                           "a broken patch: the last verify was green in 61 of 75 fails.")
            if sp:
                out.append("=== INDEPENDENT READING (/tmp/spec_test.py: model call, issue "
                           "text only) ===\n" + (sp[-1100:] or "(no output)"))
            if ow:
                out.append("=== YOUR CHECK (/tmp/contract_test.py) ===\n" + (ow[-800:] or ""))
            else:
                out.append("=== YOUR CHECK ===\nMISSING: call `spec`, or write "
                           "/tmp/contract_test.py from the report's own example (it must "
                           "fail before the fix).")
            # 2. patch shape + syntax
            st = self._b(env, state, _ENV + "cd /testbed && git --no-pager diff --stat "
                         "| tail -15") or ""
            out.append(st)
            ins = re.findall(r"(\d+) insertion", st)
            if (int(ins[0]) if ins else 0) > 25 or len(files) > 2:
                out.append("[radar] " + str(len(files)) + " file(s), +" + (ins[0] if ins
                           else "0") + " lines; passing patches median 1 file / +4. Every "
                           "line beyond the report's contract is one the hidden test can "
                           "break.")
            pys = [f for f in files if f.endswith(".py")]
            if pys:
                syn = self._b(env, state, _ENV + "cd /testbed && python -m py_compile " +
                              " ".join(shlex.quote(p) for p in pys[:20]) +
                              " 2>&1 | tail -12") or ""
                if syn.strip():
                    out.append("=== SYNTAX: your patch cannot be parsed ===\n" + syn[:1000])
            # 3. tests covering the change
            picked, scope = self._tests_for(env, state, files)
            targets = picked or ([scope] if scope else [])
            bad = []
            if targets:
                raw, bad = self._pt(env, state, targets)
                out.append("=== TESTS THAT COVER YOUR CHANGE (" +
                           ", ".join(targets[:4]) + ") ===\n" + (raw[-1200:] or "(none)"))
                if bad:
                    out.append("Those tracked tests were green before your change: a red one "
                               "now is a REGRESSION you caused - narrow the edit until it is "
                               "green. A failure named for behaviour the report asks for is "
                               "instead a gap: implement it.")
            else:
                out.append("=== TESTS ===\nNo tracked test file covers the modules you "
                           "changed - the graded tests are hidden; the issue text is your "
                           "only spec.")
            if not bad and not now:
                out.append("[verify] nothing red. Before submitting: does your patch "
                           "implement EVERY identifier in the CONTRACT CARD, in the module "
                           "the report names, including the invalid/edge case it shows? "
                           "That is all the hidden test can fail on. Then submit.")
            return "\n\n".join(x for x in out if x)[:6000]
        except Exception as e:
            return "[harness] " + str(name) + " failed: " + repr(e)

    # ------------------------------------------------ observation annotations
    def after_tool(self, name, args, obs, state):
        if not isinstance(obs, str):
            return obs
        cmd = ((args or {}).get("command") if isinstance(args, dict) else "") or ""
        if not obs.strip():
            if _WRITE.search(cmd):
                return ("[ok] no output - your edit may well have landed; confirm with "
                        "cd /testbed && git diff --stat.")
            return ("[no output] Nothing matched: that symbol/string is not in these files. "
                    "Never re-run it. Take a name from the CONTRACT CARD, or start writing "
                    "the fix in the file you have already read.")
        notes = []
        # the differential: a check that fails BEFORE any edit is the only real evidence
        if (_RUN.search(cmd) and not _WRITE.search(cmd)
                and not state.get("writes") and (_TMPCHK.search(cmd) or "::" in cmd)):
            if _FAILROW.search(obs) or "Traceback" in obs:
                state["head_red"] = 1
                notes.append("[harness] that FAILED on the un-fixed code - keep it: it is "
                             "the only real evidence this episode can produce. Edit until "
                             "it passes.")
            elif _TMPCHK.search(cmd):
                notes.append("[harness] that PASSED with no patch applied, so it asserts "
                             "nothing about your fix. Make it assert the NEW behaviour the "
                             "report promises - it has to fail first.")
        nm = _NOMOD.search(obs)
        if nm or "name resolution" in obs:
            notes.append("[harness] " + (nm.group(1) + " is not importable with that "
                         "interpreter" if nm else "network failure") + "; the repo env is "
                         "/opt/conda/envs/testbed/bin (already first on PATH) and the "
                         "container is OFFLINE - never pip install or download.")
        if _NOTFOUND.search(obs) and ".py" in obs:
            notes.append("[harness] those node ids are HIDDEN tests, not in your repo (" +
                         ", ".join(_NODEID.findall(obs)[:3]) + "). Never search for them: "
                         "implement the CONTRACT CARD names literally.")
        if re.search(r"\b\d+ (failed|passed)\b", obs):
            f = re.search(r"(\d+) (?:failed|errors?)", obs)
            p = re.search(r"(\d+) passed", obs)
            bits = ([f.group(1) + " failing"] if f else []) + \
                   ([p.group(1) + " passing"] if p else [])
            txt = "[harness summary] " + ", ".join(bits)
            fl = re.findall(_FAILID, obs)[:6]
            if fl:
                txt += " | first failures: " + ", ".join(fl)
            elif not f:
                txt += (" (green before your edit too - proves nothing; the verdict that "
                        "matters is `verify`'s DIFFERENTIAL)")
            notes.append(txt)
        if not notes:
            return obs
        return "\n".join(notes) + "\n\n" + obs

    # ------------------------------------------------- cross-turn pressure
    def on_turn_end(self, turn, state):
        writes = state.get("writes", 0)
        if not writes and not state.get("specced") and turn in (5, 9, 13):
            return ("Call `spec` now: one tool call - it reads the ISSUE TEXT ONLY and "
                    "writes+runs an independent check that fails on the un-fixed code. Read "
                    "which assertion fails, THEN edit.")
        if writes == 0 and turn in (10, 16, 22, 30, 45, 70):
            return ("NO SOURCE EDIT YET (turn " + str(turn) + "). Stop reading files: take "
                    "the file you have already read and land your best-guess fix NOW, then "
                    "git diff --stat. An empty diff grades 0; a plausible patch can pass.")
        if writes and not state.get("head_red") and turn in (14, 24, 40):
            return ("You edited, but NOTHING you ran ever failed on the un-fixed code, so "
                    "nothing you see can tell you whether the fix works. `spec` (issue text "
                    "only), or a /tmp/contract_test.py of the report's example: make it "
                    "fail, then make it pass.")
        if writes and state.get("still_red") and turn % 4 == 0:
            return ("Your own check is still RED: the patch does not deliver what the "
                    "report promises. Fix that assertion before you submit.")
        return None
