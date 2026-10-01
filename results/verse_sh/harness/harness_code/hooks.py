"""hooks.py -- executable harness layer.

Failure modes this is built from (78 failing traces vs 32 passing ones):

M1 DEGENERATE READ LOOPS. Failing traces re-issued the *same* command 3..87
   times in a row (mean streak 5.1 vs 1.7 on passes; 15% of all failing
   commands were verbatim duplicates) and died with an empty patch. before_tool
   refuses a 3rd identical read-only call and hands the model back its own
   previous output plus the next action. Replayed offline over the recorded
   corpus: fires on 8 failing traces (85/74/56/52/26/10/4/3 blocks), 0 errors.

M2 ENDING WITH AN EMPTY PATCH. 15/78 failures never wrote a single file
   ("VERIFIER: no patch submitted"), a guaranteed 0 the model could not see.
   after_llm turns that dead turn into one real status action (at most twice
   per episode, and never once source has been edited).

M3 ENVIRONMENT BLINDNESS. The repo's interpreter is not plain `python`: half of
   all episodes burned turns on "No module named pytest" and on `pip install`
   in a container with no network (0 of 45 pip installs ever succeeded).

All hooks are bookkeeping + string work and degrade to identity on any error.
"""

import re
import shlex

_READ_TOKENS = frozenset((
    "cat", "sed", "grep", "rg", "ag", "find", "ls", "head", "tail", "wc", "nl",
    "awk", "sort", "uniq", "tree", "stat", "file", "which", "whereis", "tr",
    "cut", "column", "less", "more", "diff", "pydoc", "echo", "pwd", "basename",
    "dirname", "realpath", "du", "date", "python", "python3", "python2",
))
_TEST_MARKERS = ("pytest", "py.test", "-m unittest", "unittest discover", "tox",
                 "nosetests", "make test", "coverage run")
_WRAPPERS = frozenset(("cd", "sudo", "time", "env", "nohup", "command", "then",
                       "do", "if", "for", "export", "set", "timeout"))
_NET_RE = re.compile(r"\b(pip|pip3|uv|conda|poetry|curl|wget|npm)\b")
_FDW_RE = re.compile(r"\d*>&+\d+")
_HEREDOC_RE = re.compile(r"<<-?\s*['\"]?[A-Za-z_]\w*")


def _norm(cmd):
    return re.sub(r"\s+", " ", cmd or "").strip()


def _strip_scratch(cmd):
    """Blank out writes that cannot touch the repo (2>&1, >/dev/null, > /tmp/x,
    tee /tmp/x) so they are not mistaken for source edits."""
    c = _FDW_RE.sub(" ", cmd or "")
    c = re.sub(r"\btee\s+(-a\s+)?(/[^\s;|']*|\S*\.log)\b",
               lambda m: " " if "/tmp/" in m.group(0) or "/dev/" in m.group(0)
               else m.group(0), c)
    c = re.sub(r"\d*>>?\s*(/tmp/[^\s;|']+|/dev/null|/dev/stdout|\S+\.log)\b",
               " ", c)
    return c


def _looks_write(cmd):
    c = _strip_scratch(cmd or "")
    low = " " + _norm(c).lower() + " "
    if (re.search(r"\bsed\s+[^|;]*-i\b", low) or "perl -pi" in low
            or "perl -0pi" in low or "git apply" in low or " patch -p" in low
            or "write_text" in low or "write_bytes" in low
            or re.search(r"open\([^()]*,\s*['\"][wa]", low)
            or "tee " in low or "truncate -s" in low):
        return True
    if re.search(r"\d*>>?\s*\S", c):          # any surviving redirect
        return True
    return False


def _looks_test(cmd):
    low = _norm(cmd).lower()
    return any(m in low for m in _TEST_MARKERS)


def _looks_pip_install(cmd):
    low = _norm(cmd).lower()
    return bool(re.search(r"\b(pip|pip3|uv|conda|poetry|mamba)\b[^|;&]*"
                          r"\b(install|add|download)\b", low))


def _segments(cmd):
    """Quote-aware split into segments. None when the command cannot be
    tokenised safely (heredoc / unbalanced quotes) -> caller must not touch it."""
    if _HEREDOC_RE.search(cmd):
        return None
    try:
        toks = shlex.split(cmd)
    except Exception:
        return None
    if not toks:
        return None
    segs, cur = [], []
    for t in toks:
        if t in ("&&", "||", ";", "|", "&"):
            segs.append(cur)
            cur = []
        else:
            cur.append(t)
    segs.append(cur)
    return [s for s in segs if s]


def _cmd_word(seg):
    """First real command word of a segment (skipping cd/env/timeout-style
    wrappers and their arguments); None if the segment is not a plain command."""
    i = 0
    while i < len(seg):
        t = seg[i].lower()
        if t in _WRAPPERS:
            i += 1
            if i < len(seg) and (t in ("timeout", "time")
                                 or "=" in seg[i] or seg[i][:1] in ("-", "/")
                                 or re.match(r"^[\d.]+[sm]?$", seg[i])):
                i += 1              # wrapper argument (path / duration / VAR=val)
            continue
        return t
    return None


def _blockable(cmd):
    """True for pure-inspection commands: re-running one cannot change the
    world, so the answer is already known and the repeat is pure waste."""
    c = _norm(cmd)
    if not c or _looks_write(c) or _looks_test(c) or _looks_pip_install(c):
        return False
    if _NET_RE.search(c):
        return False
    segs = _segments(c)
    if not segs:
        return False
    low = c.lower()
    for seg in segs:
        if len(seg) <= 2 and seg[0].lower() == "cd":
            continue                     # directory change only
        t = _cmd_word(seg)
        if t is None:
            return False
        if t == "git":
            return False                 # git subcommands are too varied
        ok = t in _READ_TOKENS
        if ok and t.startswith("python"):
            ok = not any(w in low for w in ("open(", "shutil", "subprocess",
                                            "os.system", "os.remove", ".write(",
                                            "import pytest", "sys.argv"))
        if not ok:
            return False
    return True


class Hooks(BaseHooks):

    loop = {
        "nudge_no_tool": (
            "Do not stop. Call `bash` and keep working: write the actual source "
            "edit, verify it with the repo's own interpreter (an env under "
            "/opt/conda/envs/*/bin/python -- plain `python` has no pytest), then "
            "call `submit`. An episode whose final `git diff` is empty scores 0."),
    }

    # ------------------------------------------------------------------ prompt
    def system_prompt(self, assembled):
        return (assembled.rstrip() + "\n\n"
            "HOW TO WORK IN THIS ENVIRONMENT\n"
            "1. ACT EARLY. Make your first source edit within ~10 commands and "
            "keep exploring afterwards to refine it. An edit that is written and "
            "then corrected beats a perfect plan: an episode whose final `git "
            "diff` is empty scores zero, no matter how much you understood.\n"
            "2. NEVER RUN THE SAME COMMAND TWICE. A repeated `cat`/`grep` returns "
            "the same text and gets you nothing. Need a fact you already saw -- "
            "use it. Need a new fact -- ask a different question (other file, "
            "other pattern, `-A 20` for context).\n"
            "3. PYTHON. The repo's packages live in a conda env, NOT in plain "
            "`python`. Run tests with /opt/conda/envs/testbed/bin/python -m "
            "pytest <path> -q (if that path is missing, `ls /opt/conda/envs` "
            "once and use the env you find). Plain `python -m pytest` only "
            'prints "No module named pytest".\n'
            "4. NO NETWORK. `pip install` / `uv add` can never work here -- do "
            "not try it, not even once; every dependency is already installed.\n"
            "5. EDIT RECIPE (reliable for multi-line changes):\n"
            "   cd /testbed && python3 - <<'PY'\n"
            "   p = 'src/pkg/file.py'\n"
            "   s = open(p).read()\n"
            "   s = s.replace(OLD, NEW, 1)\n"
            "   open(p, 'w').write(s)\n"
            "   PY\n"
            "   then `git diff` to confirm the edit landed.\n"
            "6. FINISH. Before `submit`: run the test file covering your change "
            "and read `git diff` once. If a test you expected still fails, fix "
            "the source -- never delete, skip or weaken tests to go green.\n")

    # -------------------------------------------------- command bookkeeping
    def before_tool(self, name, args, state):
        if name != "bash" or not isinstance(args, dict):
            return args
        try:
            cmd = args.get("command") or ""
            key = _norm(cmd)
            if not key:
                return args
            if _looks_pip_install(cmd):
                state["pipblocked"] = state.get("pipblocked", 0) + 1
                return ("no-network",
                    "[harness] BLOCKED: this container has NO network, so `pip "
                    "install` can never succeed (it only burns minutes on retries). "
                    "Every dependency is already installed -- use the repo env "
                    "interpreter: /opt/conda/envs/testbed/bin/python -m pytest "
                    "<path> -q  (or `ls /opt/conda/envs` to find it). Continue "
                    "with the actual fix.")
            seen = state.setdefault("seen", {})
            n = seen.get(key, 0)
            if n >= 2 and _blockable(key):
                state["blocked"] = state.get("blocked", 0) + 1
                prev = (state.get("cache", {}).get(key) or "(output not captured)")[:800]
                msg = ("[harness] BLOCKED: you have issued this exact command 3 "
                       "times and its output cannot change. Its output was:\n"
                       "---\n" + prev + "\n---\n")
                if state.get("edits", 0) == 0:
                    msg += ("You have modified NO file yet, so this episode scores "
                            "0 as it stands. Your next command must WRITE an edit:\n"
                            "cd /testbed && python3 - <<'PY'\n"
                            "p = '<the file you have just read>'\n"
                            "s = open(p).read()\n"
                            "s = s.replace(OLD, NEW, 1)\n"
                            "open(p, 'w').write(s)\n"
                            "PY\nThen run the tests for it and check `git diff`.")
                else:
                    msg += ("Your edits are already on disk. Stop re-reading: run "
                            "the tests covering the changed file, then call "
                            "`submit`.")
                return ("repeat-of-identical-read", msg)
            seen[key] = n + 1
            if len(seen) > 500:
                for k in list(seen)[:250]:
                    seen.pop(k, None)
            if _looks_write(cmd):
                state["seen"] = {}          # after an edit, re-reads are legitimate
                state["edits"] = state.get("edits", 0) + 1
                state["last_edit_turn"] = state.get("turn", 0)
            elif _looks_test(cmd):
                state["tests"] = state.get("tests", 0) + 1
            return args
        except Exception:
            return args

    def after_tool(self, name, args, obs, state):
        if name != "bash":
            return obs
        try:
            out = obs if isinstance(obs, str) else ""
            key = _norm(args.get("command", "")) if isinstance(args, dict) else ""
            if key and out:
                state.setdefault("cache", {})[key] = out[:1500]
            if not out.strip():
                return ("(no output -- that command answered nothing. Wrong path "
                        "or the code is elsewhere. Try: grep -rn \"<keyword from "
                        "the issue>\" /testbed --include=\"*.py\")")
            if "No module named pytest" in out or "No module named 'pytest'" in out:
                out += ("\n[harness] plain `python` here has no pytest -- use the "
                        "repo env: /opt/conda/envs/testbed/bin/python -m pytest "
                        "<path> -q (find envs with `ls /opt/conda/envs`).")
            elif "NewConnectionError" in out or "Retrying (Retry(total=" in out:
                out += ("\n[harness] NO network in this container: `pip install` "
                        "can never succeed -- stop retrying it. Dependencies are "
                        "already installed in /opt/conda/envs/*/bin/python.")
            elif "ModuleNotFoundError" in out and "/opt/conda/envs" not in out:
                out += ("\n[harness] that import failed in the interpreter you "
                        "used; the repo's packages live in a conda env -- rerun "
                        "with /opt/conda/envs/testbed/bin/python ...")
            return out
        except Exception:
            return obs

    # --------------------------------------------------- dead-turn rescue (M2)
    def after_llm(self, content, state):
        try:
            blocks = content if isinstance(content, list) else []
            for b in blocks:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    return content
            if state.get("edits", 0) > 0:
                return content             # a patch exists -> ending is gradable
            if state.get("injects", 0) >= 2:
                return content
            state["injects"] = state.get("injects", 0) + 1
            blocks.append({"type": "text", "text":
                "[harness] You ended the turn without calling a tool while "
                "/testbed still has NO edits -- an empty patch scores 0 for "
                "certain. Running one status command for you; on your NEXT turn "
                "write the fix into the source file that must change. If you are "
                "not certain which line is wrong, make the most plausible edit "
                "anyway: a real attempt outscores an empty diff every time."})
            blocks.append({"type": "tool_use", "name": "bash", "input": {
                "command": "cd /testbed && echo '=== patch status (empty == 0) "
                           "===' && git diff --stat | tail -5 && git status "
                           "--short | head -20"}})
            return blocks
        except Exception:
            return content

    # ---------------------------------------------------------- progress notes
    def on_turn_end(self, turn, state):
        try:
            state["turn"] = turn
            edits = state.get("edits", 0)
            tests = state.get("tests", 0)
            fired = state.setdefault("fired", {})
            if turn - state.get("last_note", -10) < 4:
                return None

            def go(tag, text):
                fired[tag] = 1
                state["last_note"] = turn
                return text

            if edits == 0:
                if turn >= 25 and not fired.get("a3"):
                    return go("a3",
                        "BUDGET WARNING: %d commands in and not one file has been "
                        "written -- that scores 0. Write your best-guess fix NOW "
                        "(python3 here-doc: read the file, .replace(), write it "
                        "back), then test it. Further exploration cannot pay for "
                        "itself any more." % turn)
                if turn >= 14 and not fired.get("a2"):
                    return go("a2",
                        "CHECKPOINT: `git diff` is still empty after %d commands. "
                        "Pick the function the issue names and edit it now; keep "
                        "reading only between edits." % turn)
                if turn >= 6 and not fired.get("a1"):
                    return go("a1",
                        "NOTE: %d commands of exploration, no file written yet. "
                        "Make your first source edit within ~5 commands, then "
                        "verify it." % turn)
            elif tests == 0 and turn >= state.get("last_edit_turn", 0) + 5:
                if not fired.get("b1"):
                    return go("b1",
                        "You have edited source but never RUN a test. Verify with "
                        "/opt/conda/envs/testbed/bin/python -m pytest <test file> "
                        "-q (locate it: `ls tests/`, or grep the changed symbol in "
                        "the test dir). Then call `submit`.")
            elif tests and edits and turn >= 45 and not fired.get("c1"):
                return go("c1",
                    "%d turns used and your edit is on disk. Run the covering "
                    "tests once, read `git diff` once, then call `submit` -- do "
                    "not go back to reading files." % turn)
            return None
        except Exception:
            return None
