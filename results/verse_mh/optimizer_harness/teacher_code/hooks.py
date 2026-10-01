"""Governor hooks for MY teaching loop (authored after round 1).

Measured failure patterns of my own teaching episodes (round 1 evidence):
  * ~110 bash corpus-mining calls per candidate episode vs 4-6 fix_probes: the
    scarce resources (turns, shared execution pool) were spent on exploration
    that a single reusable triage script does in one call.
  * The accepted change-set's transforming hooks fixed 3 tasks but REGRESSED 2
    previously-passing ones: probes only tested predicted-fix directions, never
    a "stays green" canary from the currently-passing set.
  * 4 of 9 observed flips did not reproduce: single-run flip evidence is ~coin
    flip; conclusions should weight confirmed (twice-run) flips.
  * Every episode still submitted (good) - keep the anti-deadlock releases.

Design: every hook is fail-open (identity on any exception), blocks are capped
and self-releasing, and the only additive tool runs a pure-computation triage
script over the run directory.
"""

import re

BASH_SOFT = 40          # bash mining calls allowed before guidance kicks in
BASH_HARD = 70          # absolute cap once drafting/probing is underway
POOL_CAP = 4            # fix_probe + run_episode per episode (pool shared across episodes)

# commands that are drafting/tooling rather than corpus mining: always allowed
_DRAFTY = re.compile(
    r"/scratch|cat\s*>|\btee\s|<<?'?-?\s*(EOF|PY)|python[0-9]*\s+-<|cp\s|mkdir|git\s+diff|sed\s+-i",
    re.I)

_TRIAGE = r'''
import glob, re, sys, collections
import os
cands = []
for pat in ("round_*/trajectories", "/run_dir/round_*/trajectories"):
    cands += glob.glob(pat)
d = sorted(cands)[-1] if cands else ""
if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
    d = sys.argv[1].rstrip("/")
if not d:
    print("no trajectories dir found"); raise SystemExit
flag = collections.defaultdict(list)
rows = []
for f in sorted(glob.glob(d + "/*.md")):
    t = open(f, errors="ignore").read()
    m = re.search(r"task_id: (\S+)", t)
    tid = m.group(1) if m else f.split("/")[-1][:-3]
    ok = bool(re.search(r"eval_passed: True", t[:500]))
    cmds = re.findall(r"```bash\n(.*?)```", t, re.S)
    ran = any(re.search(r"pytest|-m unittest", c) for c in cmds)
    green = bool(re.search(r"\d+ passed", t)) and not re.search(r"\d+ failed", t)
    base = bool(re.search(r"No module named .?pytest", t))
    pip = any(re.search(r"\b(pip|uv|conda)\b.*install", c) for c in cmds)
    edit = any(re.search(r"sed -i|write_text|git apply|patch -p|\.write\(", c) for c in cmds)
    rows.append((tid, ok))
    if not ok:
        for k, v in [("never_ran_tests", not ran), ("no_edit_seen", not edit),
                     ("base_python_no_pytest", base), ("pip_try_offline", pip),
                     ("green_but_graded_fail", green), ("mangled_nodeids", "ERROR: not found" in t),
                     ("net_attempt", "NewConnectionError" in t)]:
            if v: flag[k].append(tid)
nfail = sum(1 for r in rows if not r[1])
print("triage over %s: %d tasks, %d pass, %d fail" % (d, len(rows), len(rows) - nfail, nfail))
for k, v in sorted(flag.items(), key=lambda kv: -len(kv[1])):
    print("%-24s %3d  e.g. %s" % (k, len(v), v[:5]))
passing = [r[0] for r in rows if r[1]]
print("passing set (%d): pick regression canaries from these" % len(passing))
print(", ".join(passing[:15]))
'''


class Hooks(BaseHooks):

    # ------------------------------------------------------------ extra tool
    def extra_tools(self):
        return [{
            "name": "corpus_triage",
            "description": ("Run the standard failure-mode triage over the latest round's "
                            "trajectories in ONE call: pass/fail counts, failure-class flags "
                            "(never_ran_tests, no_edit_seen, base_python_no_pytest, "
                            "pip_try_offline, green_but_graded_fail, mangled_nodeids, "
                            "net_attempt) with example task ids, and the passing-task list to "
                            "pick regression canaries from. Replaces dozens of exploratory "
                            "bash calls. Optional arg: dir (path to a trajectories dir)."),
            "input_schema": {"type": "object",
                             "properties": {"dir": {"type": "string",
                                                    "description": "trajectories dir (default: latest round)"}}},
        }]

    def run_tool(self, name, args, env, state):
        if name != "corpus_triage":
            return "unknown tool %s" % name
        try:
            d = ((args or {}).get("dir") or "").strip().replace("'", "")
            argv = " - %s" % d if d else " -"
            return env.bash("python3%s <<'TRIAGEPY'\n%s\nTRIAGEPY\n" % (argv, _TRIAGE))
        except Exception as e:
            return "corpus_triage failed: %r (fall back to manual bash triage)" % (e,)

    # ------------------------------------------------------------ prompt tail
    def system_prompt(self, assembled):
        try:
            return (assembled or "") + (
                "\n\nBUDGET DISCIPLINE (measured round 1: ~110 mining bash calls per episode "
                "vs 4 probes - backwards). (1) Use the corpus_triage tool (or the one triage "
                "script) FIRST; then targeted greps only. Commit to your top 2 root causes by "
                "turn ~25 and draft edits in /scratch. (2) Every probe batch must include at "
                "least one REGRESSION CANARY: a task the CURRENT head already passes - "
                "round-1 transforming hooks fixed 3 and regressed 2 because no canary was "
                "ever probed. (3) A single-run flip is ~50% noise: count only twice-confirmed "
                "flips; prefer killing a whole failure class over chasing task luck. "
                "(4) Reserve the last ~8 turns: final probe of the exact bytes you submit, "
                "then submit. Unsubmitted episode = zero.")
        except Exception:
            return assembled

    # ------------------------------------------------------------ tool gate
    def before_tool(self, name, args, state):
        try:
            if name in ("fix_probe", "run_episode"):
                used = state.get("pool", 0)
                if used >= POOL_CAP:
                    return ("execution pool reserved",
                            "BLOCKED: this episode already used its %d shared-execution slots "
                            "(other parallel episodes need them). Revise your draft using "
                            "evidence you already have, then submit." % POOL_CAP)
                state["pool"] = used + 1
                return args

            if name == "bash":
                cmd = ""
                try:
                    cmd = (args or {}).get("command") or ""
                except Exception:
                    return args
                if _DRAFTY.search(cmd):
                    state["draft"] = True
                    return args
                n = state.get("bash", 0) + 1
                state["bash"] = n
                cap = BASH_HARD if (state.get("pool") or state.get("draft")) else BASH_SOFT
                if n > cap:
                    blocked = state.get("blocked", 0) + 1
                    state["blocked"] = blocked
                    if blocked <= 2:   # anti-deadlock: 3rd attempt passes through
                        return ("mining budget spent",
                                "BLOCKED (%d/%d): %d mining bash calls already. Use corpus_triage, "
                                "or write your edits to /scratch (allowed), or fix_probe. If you "
                                "already know the answer, act on it; if you must confirm exactly "
                                "one more thing, retry this single command - the next retry passes."
                                % (blocked, 2, n))
                return args

            if name == "submit_proposal":
                edits = []
                try:
                    edits = (args or {}).get("edits") or []
                except Exception:
                    return args
                if edits and not state.get("pool") and not state.get("warned"):
                    state["warned"] = True
                    return ("probe the draft first",
                            "BLOCKED (once): zero executed evidence yet this episode. Draft "
                            "these same bytes in /scratch and fix_probe 2 predicted-fix targets "
                            "+ 1 passing canary; a probe reporting your hook call counts beats "
                            "any amount of reading. Then resubmit (this block fires only once).")
                return args
        except Exception:
            return args
        return args

    # ------------------------------------------------------------ pacing
    def on_turn_end(self, turn, state):
        try:
            t = int(turn)
        except Exception:
            return None
        try:
            if t == 20 and not state.get("draft"):
                return ("Milestone t20: stop broad mining. Write your top-2 root causes and "
                        "draft edit files to /scratch now (corpus_triage covers the stats).")
            if t == 32 and not state.get("pool"):
                return ("Milestone t32: no executed evidence yet. fix_probe your current draft "
                        "on 2 fix-targets + 1 canary NOW; pool left this episode: %d."
                        % max(0, POOL_CAP - state.get("pool", 0)))
            if t == 45:
                return ("Milestone t45: final probe (same bytes as submission) then SUBMIT "
                        "within a few turns. Unsubmitted = 0 for the whole episode.")
            if t == 52:
                return ("SUBMIT NOW with your best current change-set - budget is almost out. "
                        "A mediocre submitted change-set beats a perfect unsubmitted one.")
        except Exception:
            return None
        return None
