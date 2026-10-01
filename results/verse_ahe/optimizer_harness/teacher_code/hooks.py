"""teacher_code/hooks.py -- Mission Control v5 for my teaching loop.

Re-derived from the run dir this meta-episode (rounds.jsonl + report_cards + the three
round_04 intervener audits):
* Scoreboard val /50: R1 17->19 (+4/-2), R2 19->20 (+1/-0), R3 20->19 (+1/-2, 'not
  gated'), R4 19->20 (+1/-0, EFFECTIVE, base_round=2 prune of R3's annotate-only
  injectors). The head is now the ~14.5 KB R2 layer; its only marker string is
  '[harness]'. The prune R3 made possible is DONE.
* WHY MY TOOLS WERE WORTHLESS IN R4 (measured): all three arms' `firecount` printed "no
  round_*/trajectories anywhere near /run_dir" and all three `verify` prints said
  "runtime: NOT FOUND" -- the FREE contract dry-run, my flagship, NEVER RAN. Cause: a
  teaching episode's /run_dir IS the ROUND dir (flat: trajectories/, traj_train/,
  workspace/harness_code/, executor_source/hooks_runtime.py, outcomes.json,
  history.json) while my meta dir has round_NN/ subdirs; v4 hard-coded round_*/... . v5
  discovers either layout, reads pass/fail from each trajectory's own eval_passed header,
  and prints the real listing instead of a dead end.
* Cost: self_a 22 / self_c 23 / self_b 32 bash calls; self_b hit the 30-commands-per-
  episode bash budget and was cut off mid-analysis. Submits landed at turns 25/36/26.
  Pool rose to 11/24 because verify bundled 3 probes per call -- keep that bundling.
* All three arms proposed the SAME prune, because my head-facts were stale: they are
  rewritten below so the lenses diverge next round.
"""
import re

_ALLOWED = {"re", "json", "math", "string", "textwrap", "difflib", "collections",
            "itertools", "functools", "heapq", "bisect", "statistics", "copy",
            "fnmatch", "shlex", "time", "pathlib"}
_ROOTS = ("harness_code/", "memory.md", "middleware/", "skills/", "subagents/",
          "system_prompt.md", "tool_notes/", "workflow.md")
_TAG = re.compile(r"\[harness[a-z0-9-]*\]")
_SEEN = {}

# Layout-agnostic search: an episode sees the round dir as /run_dir, my meta-episode sees
# round_NN/ subdirs. Every lookup must work in BOTH and must say what it saw if it fails.
_FIND = r'''
import glob, os, re
def _rank(p):
    m = re.search(r"_(\d+)$", os.path.basename(os.path.dirname(p) or ""))
    return int(m.group(1)) if m else -1
def discover(pats, want_dir=True, min_files=1):
    out = {}
    for pat in pats:
        for p in glob.glob(pat):
            if want_dir:
                if not os.path.isdir(p) or len(glob.glob(os.path.join(p, "*.md"))) < min_files:
                    continue
            elif not os.path.isfile(p):
                continue
            try:
                out[os.path.realpath(p)] = p
            except Exception:
                out[p] = p
    return sorted(out.values(), key=lambda x: (_rank(x), len(glob.glob(os.path.join(x, "*.md")))
                                               if os.path.isdir(x) else 0))
def latest(pats, want_dir=True, min_files=1):
    h = discover(pats, want_dir, min_files)
    return h[-1] if h else ""
def listing():
    bits = ["cwd: " + os.getcwd()]
    for d in (".", "/run_dir", "../"):
        try:
            bits.append("ls %s: %s" % (d, ", ".join(sorted(os.listdir(d))[:40])))
        except Exception:
            pass
    return "\n".join(bits)
'''

_SWEEP_PATS = ("trajectories", "traj_*", "*/trajectories", "*/traj_*",
               "/run_dir/trajectories", "/run_dir/traj_*",
               "/run_dir/round_*/trajectories", "/run_dir/round_*/traj_*",
               "../trajectories", "../round_*/trajectories")
_RT_PATS = ("executor_source/hooks_runtime.py", "*/executor_source/hooks_runtime.py",
            "/run_dir/executor_source/hooks_runtime.py",
            "/run_dir/round_*/executor_source/hooks_runtime.py",
            "../executor_source/hooks_runtime.py", "*/*/hooks_runtime.py", "*/hooks_runtime.py")
_HEAD_PATS = ("workspace/harness_code/hooks.py", "*/workspace/harness_code/hooks.py",
              "/run_dir/workspace/harness_code/hooks.py",
              "/run_dir/*/workspace/harness_code/hooks.py")


def _strip_strings(content):
    q = chr(34) * 3
    body = "".join(content.split(q)[::2])
    body = re.sub(r"'[^'\n]*'", "S", body)
    body = re.sub(q + "[^" + q + "]*" + q, "S", body)
    return "\n".join(l.split("#")[0] for l in body.split("\n"))


def _store(state):
    return state if isinstance(state, dict) else _SEEN


def _bump(store, key):
    try:
        v = int(store.get(key, 0) or 0) + 1
    except Exception:
        v = 1
    store[key] = v
    return v


def _sh(env, cmd):
    try:
        r = env.bash(cmd)
    except Exception as exc:
        return "[tool] env.bash raised: %r" % (exc,)
    return r if isinstance(r, str) else str(r)


def _edits(args):
    raw = args.get("edits")
    return [e for e in raw if isinstance(e, dict)] if isinstance(raw, list) else []


def _shape_problems(args):
    if not isinstance(args.get("edits"), list):
        return ["'edits' must be a LIST of {path, content} objects, never one JSON string -- "
                "one arm lost a whole round to that at turn 19."]
    if not _edits(args):
        return ["'edits' is empty: pass at least one {path, content}."]
    p = []
    for i, e in enumerate(_edits(args)):
        path = e.get("path")
        if not isinstance(path, str) or not path.strip():
            p.append("edit[%d]: 'path' must be a real string path." % i)
            continue
        if not any(path.startswith(r) for r in _ROOTS):
            p.append("edit[%d] (%s): outside the writable harness roots %s -- a teacher_ws/ or "
                     "stray path bounces the WHOLE proposal. Durable notes go in memory.md."
                     % (i, path, ", ".join(_ROOTS)))
            continue
        if not e.get("delete") and not isinstance(e.get("content"), str):
            p.append("edit[%d] (%s): no string 'content' -- every edit must carry the FULL file "
                     "text or the change is silently dropped." % (i, path))
    return p


def _block_of(content, start):
    body = content[start:start + 3000]
    cut = body.find("\n    def ")
    return body[:cut] if cut > 0 else body


_BANCE = re.compile(r"(?<![\w.])(open|eval|exec|compile|getattr|setattr|globals|locals)"
                    r"\s*\(")
_REBUILD = re.compile(r"return\s*(\[|sorted\(|list\()")


def _py_problems(content, head_src=""):
    """Free contract checks for a drafted harness hooks.py (no pool)."""
    p, n = [], []
    if len(content) > 30000:
        p.append("content is %d chars: over the executor's source cap -- the audit bounces it."
                 % len(content))
    code = _strip_strings(content)
    for m in re.finditer(r"^\s*(?:from|import)\s+([a-zA-Z_][\w.]*)", code, re.M):
        if m.group(1).split(".")[0] not in _ALLOWED:
            p.append("import '%s' is not on the allowlist (pure-computation modules only)."
                     % m.group(1))
    for m in _BANCE.finditer(code):
        p.append("banned construct '%s(' -- the audit rejects it." % m.group(1))
    for m in re.finditer(r"^\s*def\s+(__\w+_*)", code, re.M):
        if m.group(1) != "__init__":
            p.append("defining %s is rejected by the audit (only __init__ is allowed)."
                     % m.group(1))
    i = content.find("def after_llm")
    if i >= 0:
        if _REBUILD.search(_block_of(content, i)):
            p.append("after_llm returns a REBUILT reply list. The runtime re-validates any NEW "
                     "list and knows only text|tool_use, but this executor emits 'thinking' "
                     "blocks -> \"unknown block type 'thinking'\", the hook degrades to identity "
                     "and does nothing all sweep. Append IN PLACE, return the same object.")
        n.append("An in-place after_llm returns the SAME object, so it can never appear in "
                 "hooks_stats['changed'] (that counter needs a non-identity return). Prove "
                 "firing by grepping the sweep for your own marker.")
    j = content.find("def after_tool")
    if j >= 0 and "isinstance" not in _block_of(content, j):
        p.append("after_tool has no isinstance(obs, str) guard -- non-string observations happen "
                 "(a TypeError on None was logged). Start with 'if not isinstance(obs, str): "
                 "return obs'.")
    if not _TAG.findall(content):
        n.append("No '[harness...]' marker in this layer: tag every injected observation or the "
                 "next round cannot count it and will silently delete it.")
    if head_src:
        lost = sorted(set(_TAG.findall(head_src)) - set(_TAG.findall(content)))
        if lost:
            p.append("your draft DROPS head marker tags: %s. If deleting that mechanism IS the "
                     "proposal, say so in the description; otherwise keep the strings "
                     "byte-identical." % ", ".join(lost[:6]))
    return p, n


_FIRE = r'''
import collections, glob, os, pathlib, re, sys
''' + _FIND + r'''
TAG = re.compile(r"\[harness[a-z0-9-]*\]")
d = latest(''' + repr(_SWEEP_PATS) + r''')
if not d:
    print("FIRECOUNT: no sweep dir found. This is what the env actually looks like:")
    print(listing()); sys.exit(0)
files = sorted(glob.glob(os.path.join(d, "*.md")))
passed, failed, hits, unk = set(), set(), collections.defaultdict(set), 0
for f in files:
    tid = os.path.basename(f)[:-3]
    try:
        txt = pathlib.Path(f).read_text(errors="ignore")
    except Exception:
        continue
    m = re.search(r"^eval_passed:\s*(\w+)", txt, re.M)
    if m:
        (passed if m.group(1).lower() in ("true", "1") else failed).add(tid)
    else:
        unk += 1
    for t in set(TAG.findall(txt)):
        hits[t].add(tid)
if not passed and not failed:
    op = latest(["outcomes.json", "*/outcomes.json", "/run_dir/outcomes.json",
                 "/run_dir/*/outcomes.json"], want_dir=False)
    if op:
        o = json.loads(pathlib.Path(op).read_text())
        passed = set(i for i in o if o.get(i) == 1.0)
        failed = set(i for i in o if o.get(i) != 1.0)
print("SWEEP %s | %d episodes | pass %d fail %d (unknown %d)"
      % (d, len(files), len(passed), len(failed), unk))
head = latest(''' + repr(_HEAD_PATS) + r''')
hs = pathlib.Path(head).read_text(errors="ignore") if head else ""
htags = set(TAG.findall(hs))
print("HEAD %s | %d bytes | tags in source: %s" % (head or "not found", len(hs), sorted(htags)))
print("FIRING: marker | episodes | of those PASSED (a marker firing in many PASSING episodes")
print("is a regression channel -> narrow its trigger or delete it)")
for t, ids in sorted(hits.items(), key=lambda kv: -len(kv[1])):
    print("  %-24s %3d/%3d | %3d passed" % (t, len(ids), len(files),
                                            len([i for i in ids if i in passed])))
if not hits:
    print("  (no [harness*] marker anywhere in this sweep -- the layer never fires)")
print("HEAD MARKERS THAT NEVER FIRED (dead trigger, or deleted after this sweep):",
      sorted(t for t in htags if t not in hits))
print("FIRING BUT NOT IN THE HEAD (tag drift, or this sweep predates the inherited head):",
      sorted(t for t in hits if t not in htags))
print("NOTE: a .md keeps only bash fences + observations, so injected TEXT can be invisible --")
print("count the mechanism's own artefact: grep -rl '<marker substring>' %s | wc -l" % d)
print("NOTE: this sweep graded the head you INHERITED; prove a NEW marker with a probe.")
'''

_VERIFY = r'''
import pathlib, sys
import glob, os, re
''' + _FIND + r'''
DRAFT = "@DRAFT@"
print(listing())
RT = latest(''' + repr(_RT_PATS) + r''')
print("runtime:", RT or "NOT FOUND -- dry-run by hand: find . -name hooks_runtime.py, import "
                      "it, HookedRuntime(<draft>), after_llm(thinking block), after_tool(None)")
if not RT:
    sys.exit(0)
sys.path.insert(0, os.path.dirname(RT))
try:
    import hooks_runtime as HR
except Exception as e:
    print("import hooks_runtime FAILED:", repr(e)); sys.exit(0)
src = pathlib.Path(DRAFT).read_text(errors="ignore")
print("draft bytes:", len(src))
cls = HR.__dict__.get("HookedRuntime")
if cls is None:
    print("no HookedRuntime; runtime names:",
          [x for x in sorted(dir(HR)) if not x.startswith("_")]); sys.exit(0)
try:
    rt = cls(src)
except Exception as e:
    print("CONSTRUCT FAILED (that is an audit bounce):", repr(e)); sys.exit(0)
st = {}
def errs(tag):
    e = list(rt.__dict__.get("errors") or [])
    print("   errors after", tag, ":", e[:4] if e else "none")
for nm, payload in (("thinking", [{"type": "thinking", "thinking": "x"},
                                  {"type": "text", "text": "hi"}]),
                    ("no-tool", [{"type": "text", "text": "done"}]),
                    ("tool_use", [{"type": "tool_use", "name": "bash", "id": "t1",
                                   "input": {"command": "ls"}}])):
    try:
        out = rt.after_llm(payload, st)
        print("after_llm", nm, "->", type(out).__name__,
              "same_obj" if out is payload else "REBUILT (danger: thinking blocks)")
        errs("after_llm/" + nm)
    except Exception as e:
        print("after_llm", nm, "RAISED", repr(e))
for nm, o in (("str", "ok failed 1 error"), ("None", None), ("dict", {"a": 1})):
    try:
        rt.after_tool("bash", {"command": "python -m pytest -q"}, o, st)
        print("after_tool", nm, "-> ok"); errs("after_tool/" + nm)
    except Exception as e:
        print("after_tool", nm, "RAISED", repr(e))
try:
    rt.before_tool("bash", {"command": "echo hi"}, st); print("before_tool -> ok")
except Exception as e:
    print("before_tool RAISED", repr(e))
try:
    print("system_prompt ->", len(str(rt.system_prompt("BASE"))), "chars")
except Exception as e:
    print("system_prompt RAISED", repr(e))
print("DRY-RUN DONE -- any RAISED line means that mechanism does nothing all sweep")
'''


class Hooks(BaseHooks):
    """Optional methods only; `state` is the per-episode dict the runtime hands every hook."""

    def system_prompt(self, assembled):
        try:
            brief = (
                "## MISSION CONTROL (these hooks enforce the starred rules)\n"
                "* HEAD STATE (journal val /50: 17->19->20->19->20, noise +-2, 'kept' only means "
                "'not gated'). R4 was SUBTRACTIVE with base_round=2 and deleted R3's two "
                "annotate-only injectors, so the head is the ~14.5 KB R2 layer: empty-diff "
                "rescue, offline-command blocker, identical-command block, runner-unavailable "
                "hint, collection-import hint, turn coach. The obvious prune is ALREADY DONE -- "
                "prune again only if firecount names a live harmful or dead mechanism, else ADD "
                "a point-of-no-return mechanism. A round that changes nothing scores zero.\n"
                "* STAR FIRE ONLY WHERE THE EPISODE IS ALREADY LOST: no-tool_use reply (the "
                "episode ENDS and the tree is graded), empty diff at exit, byte-identical "
                "repeat, empty observation. That is where R1's +4 and R2's +1 came from; "
                "changing what a PROGRESSING model reads cost R3 two rerun-confirmed "
                "regressions for one fix. If you must, cap it per episode, ADD text instead of "
                "reordering or shortening evidence, and probe currently-PASSING tasks.\n"
                "* STAR MEASURE FIRST, FREE: `firecount` prints per marker the episodes it fired "
                "in and how many of those PASSED, head markers that never fired, and tag drift. "
                "It finds the sweep in either dir layout; if it prints a directory listing, grep "
                "from that and MOVE ON -- bash is capped at 30 commands per episode and an arm "
                "that runs out is cut off mid-analysis.\n"
                "* STAR VERIFY IN ONE CALL: verify({draft:'<path>', tids:[1 short failing, "
                "2 currently-PASSING]}) dry-runs your draft against the round's real "
                "hooks_runtime for FREE (thinking block, no-tool reply, after_tool(None) -- the "
                "bugs that make a layer do nothing all sweep) and then runs the paid probes. "
                "Only PASSING probes can see a regression.\n"
                "* Do not spend turns naming tasks you will fix: predictions realized 0 times in "
                "three scored rounds and the screen (12 hard failures; winner then netted -1) "
                "cannot see regressions. Design so you cannot lose a pass.\n"
                "* STAR EDITS MAY ONLY TOUCH: harness_code/, memory.md, middleware/, skills/, "
                "subagents/, system_prompt.md, tool_notes/, workflow.md. A teacher_ws/ path "
                "bounces the WHOLE proposal. 'edits' is a LIST of {path, content} with FULL "
                "contents; durable notes go in memory.md.\n"
                "* SUBMIT BY TURN 14-18. Last round the arms submitted at 25/36/26 and one ran "
                "out of bash. Unsubmitted = zero; keep 2 turns for a resubmit.\n\n")
            return brief + (assembled or "")
        except Exception:
            return assembled

    def extra_tools(self):
        return [
            {"name": "firecount",
             "description": ("FREE (no execution pool). Finds the newest sweep in EITHER layout "
                             "(the round dir mounted as /run_dir, or round_NN/ subdirs), takes "
                             "pass/fail from each trajectory's own eval_passed header, and "
                             "prints for every '[harness*]' marker the episodes it fired in and "
                             "how many of those PASSED, plus head markers that never fired and "
                             "tag drift. If it finds nothing it prints the real directory "
                             "listing. Call before choosing a mechanism. No arguments."),
             "input_schema": {"type": "object", "properties": {}, "required": []}},
            {"name": "verify",
             "description": ("One call = full verification of a drafted harness hooks.py. Step 1 "
                             "(FREE): locates the round's real hooks_runtime in any layout and "
                             "runs your draft through after_llm (thinking / no-tool / tool_use), "
                             "after_tool (str / None / dict), before_tool and system_prompt, "
                             "printing every error and whether each hook returned the same "
                             "object. Step 2 (PAID, pool): runs each tid as an episode with the "
                             "draft applied as harness_code/hooks.py. Pass 1 short failing id "
                             "AND 2 currently-PASSING ids: only passing probes see "
                             "regressions."),
             "input_schema": {"type": "object",
                              "properties": {
                                  "draft": {"type": "string",
                                            "description": "path to your drafted hooks.py "
                                                           "(e.g. /scratch/draft/hooks.py)"},
                                  "tids": {"type": "array", "items": {"type": "string"},
                                           "description": "task ids to probe with the draft "
                                                          "applied (paid)"}},
                              "required": ["draft"]}},
        ]

    def run_tool(self, name, args, env, state):
        try:
            args = args if isinstance(args, dict) else {}
            if name == "firecount":
                return _sh(env, "python3 - <<'PYEOF'\n" + _FIRE + "PYEOF\n")[:9000]
            if name == "verify":
                draft = str(args.get("draft") or "").strip().replace("'", "").replace('"', "")
                if not draft:
                    return "verify: give 'draft' = path to your drafted hooks.py."
                tids = args.get("tids") or []
                if isinstance(tids, str):
                    tids = [tids]
                tids = [str(t).strip() for t in list(tids)[:3] if str(t).strip()]
                out = ["== FREE contract dry-run of %s ==" % draft]
                out.append(_sh(env, "python3 - <<'PYEOF'\n"
                                    + _VERIFY.replace("@DRAFT@", draft) + "PYEOF\n"))
                if not tids:
                    out.append("== NO PAID PROBE RUN == call again with "
                               "tids=[1 short failing, 2 currently-PASSING].")
                    return "\n".join(out)[:9000]
                src = _sh(env, "cat " + draft)
                if len(src) < 200 or src.startswith("[tool] env.bash raised"):
                    out.append("== probes skipped: cannot read the draft (%d chars) ==" % len(src))
                    return "\n".join(out)[:9000]
                out.append("== PAID probes with the draft applied to harness_code/hooks.py ==")
                for t in tids:
                    try:
                        r = env.run_episode(t, [{"path": "harness_code/hooks.py",
                                                 "content": src}])
                    except Exception as exc:
                        r = "run_episode raised %r" % (exc,)
                    out.append("-- %s : %s" % (t, re.sub(r"\s+", " ", str(r or ""))[:800]))
                return "\n".join(out)[:12000]
            return "unknown tool: %s" % name
        except Exception as exc:
            return "[tool %s failed: %r -- fall back to manual bash/run_episode]" % (name, exc)

    def before_tool(self, name, args, state):
        try:
            if not isinstance(args, dict):
                return args
            store = _store(state)
            if name in ("verify", "firecount"):
                if name == "verify":
                    store["verified"] = 1
                    if args.get("tids"):
                        _bump(store, "paid")
                return args
            if name == "bash":
                if "hooks_runtime" in str(args.get("command") or ""):
                    store["verified"] = 1
                _bump(store, "bash")
                return args
            if name in ("fix_probe", "run_episode", "ablate", "substitute", "replay"):
                n = _bump(store, "paid")
                many = max([len(v) for v in args.values() if isinstance(v, list)] or [0])
                if n >= 8 and int(store.get("probe_gates", 0) or 0) < 1:
                    _bump(store, "probe_gates")
                    extra = (" Also %d targets in one call: one SHORT episode answers the same "
                             "question (long suites die on a 3600s wall-clock)." % many) \
                        if many > 3 else ""
                    return ("probe-budget",
                            "[harness] %s call #%d. 'fixed 0/N' is the MEASURED base rate on "
                            "hard failures, so stop re-probing failures: spend what is left on "
                            "currently-PASSING tasks with your edits applied -- regressions are "
                            "what cost net score.%s Repeat this exact call if you truly need "
                            "it." % (name, n, extra))
                return args
            if name == "submit_proposal":
                problems = _shape_problems(args)
                notes = []
                for e in _edits(args):
                    path, content = e.get("path"), e.get("content")
                    if (isinstance(path, str) and path.endswith(".py")
                            and isinstance(content, str) and "hook" in path):
                        pp, nn = _py_problems(content, str(store.get("head_src") or ""))
                        problems += ["%s: %s" % (path, x) for x in pp]
                        notes += ["%s: %s" % (path, x) for x in nn]
                if problems and int(store.get("gates", 0) or 0) < 2:
                    _bump(store, "gates")
                    tail = ("\nAlso: " + "\nAlso: ".join(notes[:3])) if notes else ""
                    return ("contract-gate",
                            "[harness] Submission NOT sent. Fix and submit again (this gate "
                            "blocks at most twice per episode, then always lets it through):\n- "
                            + "\n- ".join(problems[:8]) + tail)
                turn = int(store.get("turn", 0) or 0)
                if (not int(store.get("verified", 0) or 0) and turn <= 22
                        and int(store.get("verify_gate", 0) or 0) < 1):
                    _bump(store, "verify_gate")
                    return ("verify-gate",
                            "[harness] Submission NOT sent: this layer was never dry-run against "
                            "the real runtime. One call, then resubmit: verify({draft:'<your "
                            "hooks.py>'}) -- free, and it catches the bugs that make a layer do "
                            "nothing all sweep (a rebuilt after_llm list rejected on 'thinking' "
                            "blocks; after_tool crashing on a non-string observation). Fires "
                            "once per episode, then always lets the submit through.")
                return args
            return args
        except Exception:
            return args

    def after_tool(self, name, args, obs, state):
        try:
            if not isinstance(obs, str):
                return obs
            store = _store(state)
            low = obs.lower()
            if name in ("fix_probe", "run_episode", "verify"):
                if "fixed 0/" in low or "fail" in low or "inconclusive" in low:
                    obs += ("\n[harness] Expected -- the measured base rate on hard failures, "
                            "not evidence your mechanism is wrong. Do NOT re-probe it; spend the "
                            "next call on a currently-PASSING task, the only probe that can see "
                            "a regression.")
                if "not found" in low and "runtime" in low:
                    store["verified"] = 0
                    obs += ("\n[harness] The dry-run DID NOT RUN (runtime not located), so you "
                            "are NOT verified. Do it by hand in one bash call: `find . -name "
                            "hooks_runtime.py`, import it, build HookedRuntime(<your draft "
                            "source>), then call after_llm with a thinking block and after_tool "
                            "with None.")
                if ("unknown block type" in low or "bad return" in low or "raised" in low
                        or "traceback" in low or "typeerror" in low):
                    obs += ("\n[harness] Your hook CODE is failing, so that mechanism does "
                            "nothing all sweep: return the SAME list object you were handed "
                            "(append in place, never rebuild a reply list -- 'thinking' blocks "
                            "fail re-validation) and guard after_tool with "
                            "if not isinstance(obs, str): return obs.")
            elif name == "bash":
                n = int(store.get("bash", 0) or 0)
                if "budget" in low and ("exhaust" in low or "command" in low):
                    obs += ("\n[harness] The bash budget is spent: stop analysing. Write the "
                            "change-set from the evidence you already have, verify it, and "
                            "SUBMIT -- an unsubmitted episode scores zero.")
                elif n >= 18 and not store.get("drafted") and not int(store.get("paid", 0) or 0):
                    store["drafted"] = 1
                    obs += ("\n[harness] %d bash calls, no draft and no probe. Candidates last "
                            "round burned 22-32 calls and submitted at turns 25-36. Write the "
                            "change-set to a scratch file now, then verify(...) and submit." % n)
            elif name == "submit_proposal" and "reject" in low:
                obs += ("\n[harness] Rejected. Usual causes: (a) a path outside the harness "
                        "roots (harness_code/, memory.md, middleware/, skills/, subagents/, "
                        "system_prompt.md, tool_notes/, workflow.md) -- teacher_ws/ is NOT "
                        "writable from here; (b) a missing path or missing FULL 'content'; "
                        "(c) 'edits' given as one JSON string instead of a list. Fix and "
                        "resubmit the WHOLE change-set now.")
            return obs
        except Exception:
            return obs

    def on_turn_end(self, turn, state):
        try:
            t = int(turn)
        except Exception:
            return None
        store = _store(state)
        store["turn"] = t
        if t <= 1:
            return ("Read the journal, do not trust memory: history.json / outcomes.json in this "
                    "dir, or rounds.jsonl + report_cards/ one level up (val before/after, "
                    "verdict, kept, flip_confirm, base_round). 'kept' only means 'not gated' -- "
                    "R3 lost net (20->19) and was still kept, which is why R4 pruned it via "
                    "base_round=2. Then run the FREE `firecount` tool ONCE and keep its output.")
        if t == 4:
            return ("Decide the change-set from firing evidence, not from a story: which marker "
                    "fires in episodes that PASS (regression channel -> narrow or delete), which "
                    "fires NOWHERE (dead trigger -> fix the predicate or drop it), and which "
                    "episode shape is the biggest ALIVE omission. Do not re-propose last round's "
                    "prune -- it already shipped.")
        if t == 8 and not store.get("drafted"):
            store["drafted"] = 1
            return ("Stop widening the analysis (bash is capped at 30 commands per episode): "
                    "write the change-set to a scratch file now. Keep every existing marker "
                    "string you are not deliberately deleting, and tag anything new.")
        if t == 11 and not int(store.get("verified", 0) or 0):
            return ("Nothing verified yet. verify({draft:'<path>', tids:[1 short failing, 2 "
                    "currently-PASSING]}) does the free dry-run and the probes in one call, then "
                    "submit. Regressions, not misses, are what cost net score.")
        if t >= 14:
            return ("SUBMIT NOW if you have not: unsubmitted = zero, and a verified-enough "
                    "change-set beats a polished unpublished one. Harness-root paths only, FULL "
                    "file contents, 'edits' as a LIST; keep 2 turns for a resubmit.")
        return None
