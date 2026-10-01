"""hooks.py -- executable harness layer (round 6).

KEPT byte-preserved from the net-positive head (r1 +4/-2 = 17->19, r2 +1/-0 = 19->20, r4 +1/-0
= 19->20): the empty-diff rescue, the repeat-run breaker, the offline blocker, the runner hint,
the collection-import hint, the progress coach and loop keep_full_turns=16. Marker string
"[harness]" and every existing tag ("harness-empty-diff-check") are unchanged.

TWO CHANGES THIS ROUND, both measured on the r5 sweep (110 episodes, 35 passes, [harness] in 58):

1. ADDED -- "harness-submit-check", a second POINT OF NO RETURN. `submit` ENDS the episode and the
   graded patch IS the tree, and executor.py handles the submit tool_use (line ~675) BEFORE
   before_tool, so a submit issued while nothing has been written grades an EMPTY patch with no
   mechanism firing anywhere. after_llm now rewrites that one block IN PLACE (same list, same
   block id) into a real read-only `git status` check and after_tool words the observation from
   the actual status. It can only fire when state["writes"] == 0: measured, 35/35 passing
   episodes of the graded sweep ran at least one command _is_write() flags, so the state is
   unreachable on a passing trajectory -- the expected cost of firing there is ~0 turns. Capped
   at 2 per episode; the DIRTY wording never says "stop editing".

2. NARROWED -- the runner-unavailable hint. It matched "command not found" in ANY observation and
   fired 49 times across the sweep, 7 of them on observations from commands that were not test
   runs at all (a `grep -rn`, a `git diff --stat`, a `python -c` repro whose traceback happened
   to contain ModuleNotFoundError). It is now gated on the command being runner-shaped, so it
   stops annotating unrelated observations in progressing episodes; text and tag are unchanged.
"""

import re

_NET = ("curl ", "curl -", "wget ", "pip install", "pip3 install", "python -m pip",
        "uv pip", "conda install", "conda create", "mamba install", "poetry install",
        "npm install", "npm ci", "yarn add", "apt-get install", "apt install",
        "git clone", "git fetch", "git pull", "nslookup ", "ping -c")

_RE_TEST = re.compile("pytest|python -m unittest|nosetests|tox")

# A real, cheap, read-only command. The sentinel suffix marks it as harness-authored.
_RESCUE_CMD = ("cd /testbed && git status --porcelain | head -40; git diff --stat | tail -n 3 "
               " # harness-empty-diff-check")
_RESCUE_TAG = "harness-empty-diff-check"

# Same shape for the submit path: `submit` takes no arguments and is handled by the loop before
# before_tool, so the only place it can be intercepted is the reply block itself.
_SUBMIT_CMD = ("cd /testbed && git status --porcelain | head -40; git diff --stat | tail -n 3 "
               " # harness-submit-check")
_SUBMIT_TAG = "harness-submit-check"

_SUBMIT_EMPTY = ("[harness] You called `submit` but the tree is UNTOUCHED -- the status above shows "
                 "no modified file, so the patch that gets graded is EMPTY and this task scores 0. "
                 "Check %d of 2. This is not a prompt to keep exploring: write the fix NOW "
                 "(`sed -i ...` or a python here-doc) into the source file that owns the behaviour "
                 "the issue names, then call `submit` again. A partial or guessed patch can still "
                 "pass; an empty one cannot.")

_SUBMIT_DIRTY = ("[harness] Your edits ARE on disk, so `submit` is legitimate -- it was just issued "
                 "before any write command had been recorded here. Call `submit` again now; nothing "
                 "else is required of you.")

_RESCUE_EMPTY = ("[harness] The tree is UNTOUCHED: `git status` above shows no modified file, so "
                 "this task scores 0 no matter how good your analysis was. This is rescue %d of 3 "
                 "and the episode ends on your next no-tool turn. Decide NOW, in this turn:\n"
                 "(a) write your best-guess fix into the source file that owns the behaviour "
                 "(sed -i ... or a python here-doc). A partial or guessed patch can still pass; an "
                 "empty one cannot.\n"
                 "(b) if you still cannot see the shape of the fix, run ONE targeted `grep -n` on "
                 "the symbol the issue names and edit what it points at -- not another full read.\n"
                 "Do not summarise. Do not re-read files. Emit an edit in this turn.")

_RESCUE_DIRTY = ("[harness] The status above shows your edits ARE on disk. Nothing more is "
                 "required of the tree: run the tests of the module you touched (or `git diff` to "
                 "re-read your own change), then call `submit`. Do not end the episode with prose "
                 "only after you have verified the change you already made.")

_RULES = """

NON-NEGOTIABLE RULES (these decide your score):
- Your score IS the git diff you leave in /testbed. An empty diff scores 0 no matter how much
  you explored; a plausible guess always beats none. Never end while the diff is empty.
- Draft early: by your ~10th command have a FIRST draft edit written into the source, then
  explore only to refine it. Reading files never scores; editing does.
- Where you can, reproduce the bug with a 3-line `python -c` snippet: it shows what to change
  and proves the change afterwards.
- Never re-run a command you already ran unless the tree changed since: its output will be
  byte-identical. Use the observation you already have.
- NO NETWORK. Never pip/uv/conda install, curl, wget or git clone -- they cannot work here.
  Dependencies are preinstalled in the conda env `testbed`: `cd /testbed && python -m pytest
  <path> -x -q` (or /opt/conda/envs/testbed/bin/python -m pytest ...).
- Never deliver a new test file as the fix: graders run their own tests. Edit the source that
  owns the behaviour and make it COMPLETE -- every case the issue names, related call sites,
  public exports.
- Before ending: `git diff --stat` shows your edit and the tests of every module you touched
  have been run at least once.
"""

_HINT = ("[harness] That runner is unavailable as typed. The environment is the conda env "
         "`testbed`: `cd /testbed && /opt/conda/envs/testbed/bin/python -m pytest <file> -x -q`. "
         "Nothing can be installed (the container is offline).")

_MOD_HINT = ("[harness] The import failed at COLLECTION time, so no test ran. Two options, in "
             "order: (1) the repo's own env may sit at a different interpreter -- retry the same "
             "command as `cd /testbed && /opt/conda/envs/testbed/bin/python -m pytest <file> -x -q`, "
             "or `python -c \"import <pkg>\"` from /testbed to check which copy is picked up. "
             "(2) if that package is a genuinely absent third-party dependency (the container is "
             "offline and nothing can be installed), stop fighting that runner: verify your change "
             "with a focused `python -c` snippet against the edited file and move on. Never edit "
             "the repo's dependency pins or test config to dodge it.")

# The hint above was written for a missing/mis-pointed TEST RUNNER; gate the "command not found"
# branch on the command actually being runner-shaped.
_RE_RUNNER = re.compile(r"pytest|unittest|nosetests|tox|python")


def _norm(cmd):
    return " ".join(cmd.split())


def _is_write(cmd):
    """Plausibly changes the working tree? Over-flagging is safe: it only lets one more
    identical command through before the cache answers again, and it is what keeps the
    submit-check and the empty-diff rescue OFF every working trajectory."""
    c = cmd.replace("2>&1", " ").replace(">&2", " ").replace("2>/dev/null", " ")
    c = c.replace(">/dev/null", " ").replace("&>", " ")
    if ">" in c or "<<" in c:
        return True
    for m in ("sed -i", "sed --in-place", "perl -pi", "perl -i", "tee ", "patch ", "git apply",
              "git checkout", "git restore", "git reset", "cp ", "mv ", "rm ", "touch ",
              "mkdir ", "chmod ", "ln -s", "open(", "write(", ",'w'", ',"w"'):
        if m in cmd:
            return True
    return False


class Hooks(BaseHooks):
    loop = {"keep_full_turns": 16}

    def system_prompt(self, assembled):
        return (assembled or "") + _RULES

    # --------------------------------------- points of no return, in the reply block
    def after_llm(self, content, state):
        if not isinstance(content, list):
            return content
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            # (1) the episode is ending by SUBMIT of a tree that was never written to. Mutate the
            # block IN PLACE (same list object, same tool_use id) into the read-only tree check;
            # after_tool words the observation from the real status. Unreachable on a passing
            # trajectory -- writes==0 means no edit command ever ran, so the graded patch is empty.
            if block.get("name") == "submit":
                if state.get("writes", 0) or state.get("submit_checks", 0) >= 2:
                    return content
                state["submit_checks"] = state.get("submit_checks", 0) + 1
                block["name"] = "bash"
                block["input"] = {"command": _SUBMIT_CMD}
                return content
            return content
        # (2) no tool call at all = the episode ENDS here and the tree is graded as it stands.
        if state.get("writes", 0) or state.get("rescues", 0) >= 3:
            return content
        state["rescues"] = state.get("rescues", 0) + 1
        n = state["rescues"]
        # MUTATE IN PLACE and return the SAME list object. Returning a new list makes the runtime
        # re-validate every block, and its validator knows only text|tool_use -- this executor
        # emits `thinking` blocks, so a rebuilt list raises "unknown block type 'thinking'" and
        # the whole hook is dropped for the episode (measured in round 1: errors on 8% of episodes).
        content.append({"type": "text",
                        "text": "[harness] Empty-diff exit intercepted (rescue %d of 3): "
                                "checking the tree." % n})
        content.append({"type": "tool_use", "id": "harness_rescue_%d" % n,
                        "name": "bash", "input": {"command": _RESCUE_CMD}})
        return content

    # ------------------------------------------------- tool guards
    def before_tool(self, name, args, state):
        if name != "bash":
            return args
        cmd = args.get("command")
        if not isinstance(cmd, str) or not cmd.strip():
            return args
        low = " " + " ".join(cmd.split()) + " "
        if any(m in low for m in _NET):
            return ("offline", "[harness] Blocked: this container is offline, so that command can "
                    "never succeed and every second spent on it is a lost turn. All dependencies "
                    "are already installed. Run the tests with `cd /testbed && python -m pytest "
                    "<file> -x -q`. Now get back to fixing the source.")
        if _is_write(cmd):
            state["gen"] = state.get("gen", 0) + 1
            state["writes"] = state.get("writes", 0) + 1
            if "w0" not in state:
                state["w0"] = state.get("turn", 0)
            return args
        key = _norm(cmd)
        seen = state.setdefault("seen", {})
        gen = state.get("gen", 0)
        rec = seen.get(key)
        if isinstance(rec, list) and rec[1] == gen:
            rec[0] += 1
            if rec[0] >= 3:
                cached = rec[2] or "(output not retained -- use what you already learned)"
                return ("repeat-unchanged",
                        "[harness] Blocked: you have run this EXACT command %d times and the tree "
                        "has not changed since, so the output cannot differ. Cached output:\n%s\n"
                        "STOP re-reading. Pick one action now: write your edit (sed -i / python "
                        "heredoc), run a test, or run a genuinely DIFFERENT command that answers "
                        "a question you still have." % (rec[0], cached[:900]))
        elif isinstance(rec, list):
            rec[0], rec[1] = 1, gen
        elif len(seen) < 250:
            seen[key] = [1, gen, ""]
        return args

    def after_tool(self, name, args, obs, state):
        if name != "bash" or not isinstance(obs, str):
            return obs
        cmd = args.get("command") if isinstance(args, dict) else ""
        if isinstance(cmd, str) and cmd.strip():
            if _RESCUE_TAG in cmd:
                # The rescue's own observation: the directive is worded from the real status.
                state["rescue_seen"] = state.get("rescue_seen", 0) + 1
                n = state.get("rescues", 1)
                if obs.strip():
                    return obs + "\n\n" + _RESCUE_DIRTY
                return obs + "\n\n" + (_RESCUE_EMPTY % n)
            if _SUBMIT_TAG in cmd:
                # The submit-check's own observation (see after_llm). Both branches keep the
                # episode moving: a dirty tree is told to submit again, an empty one to edit now.
                state["submit_seen"] = state.get("submit_seen", 0) + 1
                n = state.get("submit_checks", 1)
                if obs.strip():
                    return obs + "\n\n" + _SUBMIT_DIRTY
                return obs + "\n\n" + (_SUBMIT_EMPTY % n)
            key = _norm(cmd)
            seen = state.setdefault("seen", {})
            rec = seen.get(key)
            if isinstance(rec, list):
                rec[2] = obs[:1200]
            elif len(seen) < 250:
                seen[key] = [1, state.get("gen", 0), obs[:1200]]
            if isinstance(rec, list) and rec[0] == 2 and rec[1] == state.get("gen", 0):
                obs = obs + ("\n[harness] That was the 2nd identical run of this command with an "
                             "unchanged tree; the next one will be blocked. Use what you have.")
            if _RE_TEST.search(cmd):
                state["tests"] = state.get("tests", 0) + 1
                state["test_fail"] = (" failed" in obs or " error" in obs or "FAILED" in obs
                                      or "ERROR" in obs or "no tests ran" in obs)
                if "modulenotfounderror" in obs.lower() or "importerror while loading conftest" \
                        in obs.lower():
                    return obs + "\n\n" + _MOD_HINT
        o = obs.lower()
        if "no module named 'pytest'" in o or "no module named pytest" in o:
            return obs + "\n" + _HINT
        if "command not found" in o and _RE_RUNNER.search(cmd):
            return obs + "\n" + _HINT
        return obs

    # ------------------------------------------------- coach
    def on_turn_end(self, turn, state):
        state["turn"] = turn
        writes = state.get("writes", 0)
        if state.get("coached", 0) >= 6:
            return None
        if turn - state.get("last_coach", -9) < 6:
            return None
        if writes == 0:
            if turn >= 11:
                state["last_coach"] = turn
                state["coached"] = state.get("coached", 0) + 1
                if turn < 26:
                    return ("You have used %d turns and modified NO file yet. Name in one sentence "
                            "the file that owns the behaviour, then write a DRAFT edit into it now "
                            "(sed -i / python heredoc) and refine afterwards -- more reading is "
                            "worth less than a first patch." % (turn + 1))
                return ("EMPTY DIFF = SCORE 0. %d turns in and no file has been edited. Write your "
                        "best-guess patch into the source NOW (a wrong patch is worth strictly "
                        "more than none), then `git diff` to confirm it landed." % (turn + 1))
            return None
        if state.get("tests", 0) == 0:
            if turn >= state.get("w0", 0) + 3:
                state["last_coach"] = turn
                state["coached"] = state.get("coached", 0) + 1
                return ("Your edit is on disk but nothing that exercises it has been RUN. Re-run "
                        "the reproduction snippet and the touched module's tests (`cd /testbed && "
                        "python -m pytest <file> -x -q`); a fix that was never executed is usually "
                        "wrong in its last 10 percent.")
            return None
        if state.get("test_fail") and turn >= 20:
            state["last_coach"] = turn
            state["coached"] = state.get("coached", 0) + 1
            return ("Your last test run still had failures. Read the failing assertion literally "
                    "and fix that behaviour (not the test), then re-run it plus the neighbouring "
                    "tests of the modules you touched.")
        if turn >= 30:
            state["last_coach"] = turn
            state["coached"] = state.get("coached", 0) + 1
            return ("The tests you ran pass. Before ending: (1) `git diff` and re-read the issue "
                    "once -- did you cover EVERY case it names, the related call sites and the "
                    "public exports? (2) run the test file of each module you touched. Then "
                    "`submit`.")
        return None
