# teacher_code/hooks.py v2 - the teacher's own executable layer, mounted on MY teaching loop.
# Round-1 disease: 200+ offline bash calls per episode; the probe window died on wall-clock
# with 17 of 24 execution slots destroyed unused. v1 (round 2) fixed the *cold start*: every
# candidate probed within ~6 minutes and every candidate submitted once.
# Round-2 residual disease, from the audited tool timelines: after the FIRST experiment the
# best candidate kept a probe cycle of a few minutes while the two losers went 50-75 MINUTES
# (15+ turns, often zero bash calls - just huge offline draft-writing turns) with the pool
# sitting full. So v2 meters the whole episode, not just the start:
#   - pre-first-experiment bash gate (kept from v1);
#   - inter-experiment gate: bash blocked while the pool has been silent too long;
#   - post-submit wrap-up gate (post-submit offline analysis buys nothing);
#   - on_turn_end ledger with an urgency ramp;
#   - triage v2: trajectory digests (markdown or JSON transcripts) + a `state` mode that
#     replays rounds.jsonl (verdicts, val scores, flips, hook stats) in one call.
# Everything defensive: any hook that raises degrades to identity for that call.

import re
import shlex
import json

MAX_BASH_BEFORE_FIRST_EXPERIMENT = 10   # v1 gate, unchanged
MAX_BASH_BETWEEN_EXPERIMENTS = 16       # offline bash while the pool goes unused
MAX_TURNS_BETWEEN_EXPERIMENTS = 8       # round 2: stalls were generation turns, not bash
NOTE_TURNS_BETWEEN_EXPERIMENTS = 5
MAX_BASH_AFTER_SUBMIT = 12

POOL = ("fix_probe", "run_episode", "ablate", "substitute", "replay")

CONTRACT = """

## Operating contract (mounted by your own hooks - mechanically enforced below)
- Experiments beat analysis. The execution pool expires on WALL-CLOCK, not on count:
  unused probe slots are destroyed at the end of the round, and the window usually
  dies long before the turn budget does.
- Draft edits and run a real fix_probe BEFORE your 10th bash call. Use the `triage`
  tool (modes: summary/fail/grep/state) instead of ad-hoc scanning or re-reading
  rounds.jsonl by hand.
- KEEP THE CYCLE. Round 2 evidence: the candidate that won probed every few minutes;
  the two that stalled went 50-75 minutes between experiments drafting huge files
  offline and burned their verification budget on prose. Never let more than ~5 turns
  pass without a pool experiment. Work in cycles: predict -> probe -> read flips ->
  adjust -> predict. A mechanism you have not seen FLIP a task is a hypothesis.
- SHIP SMALL. Keep head guards byte-identical and submit the single file you change;
  a byte-for-byte-untouched core makes screening flips attributable to YOUR mechanism.
  Draft the file in /scratch (heredocs/sed/write_notes), emit its full content ONCE,
  inside submit_proposal.
- Ship the smallest change-set the evidence supports, and label every claim in
  root_causes as VERIFIED (executed probe/ablate/replay) or HYPOTHESIS (offline).
- Always call submit_proposal. An unsubmitted episode scores zero; a submitted
  hypothesis beats an unsubmitted verification.
"""

TRIAGE = r'''import re, os, glob, sys, json
mode = sys.argv[1] if len(sys.argv) > 1 else "summary"
pat = sys.argv[2] if len(sys.argv) > 2 else ""

def load_md(TR):
    rows = []
    for f in sorted(glob.glob(os.path.join(TR, "*.md"))):
        try:
            txt = open(f, errors="replace").read()
        except Exception:
            continue
        tid = os.path.basename(f)[:-3]
        head = txt[:500]
        passed = "eval_passed: True" in head
        vm = re.search(r"verifier_verdict:\s*(.*)", head)
        le = re.search(r"last_error:\s*(.*)", head)
        cmds = [c.strip() for c in re.findall(r"```bash\n(.*?)```", txt, re.S)]
        rows.append(dict(t=tid, p=passed, v=(vm.group(1)[:70] if vm else ""),
                         e=(le.group(1)[:70] if le else ""), cmds=cmds, txt=txt))
    return rows

def load_json(TR):
    rows = []
    for f in sorted(glob.glob(os.path.join(TR, "*.json"))):
        try:
            d = json.load(open(f, errors="replace"))
        except Exception:
            continue
        msgs = d.get("transcript") or []
        chunks = []
        for m in msgs:
            c = m.get("content")
            if isinstance(c, list):
                c = "\n".join(json.dumps(x) if not isinstance(x, str) else x for x in c)
            if isinstance(c, str):
                chunks.append(c)
        txt = "\n".join(chunks)
        cmds = [x.strip() for x in re.findall(r"```bash\n(.*?)```", txt, re.S)]
        phi = d.get("phi")
        rows.append(dict(t=d.get("instance_id") or os.path.basename(f)[:-5],
                         p=(phi == 1 or phi is True),
                         v=str(d.get("verdict") or "")[:70], e="", cmds=cmds, txt=txt))
    return rows

TR = ""
rows = []
for cand in ["trajectories"] + sorted(glob.glob("round_*/trajectories")):
    if glob.glob(os.path.join(cand, "*.md")):
        TR, rows = cand, load_md(cand)
        break
if not rows:
    for cand in ["traj_train"] + sorted(glob.glob("round_*/traj_train")) + \
                sorted(glob.glob("*/trajectories")) + sorted(glob.glob("round_*/trajectories")) + ["trajectories"]:
        if glob.glob(os.path.join(cand, "*.json")):
            TR, rows = cand, load_json(cand)
            break
if mode == "state":
    print("== state: rounds.jsonl history ==")
    try:
        for line in open("rounds.jsonl"):
            d = json.loads(line)
            v = d.get("verdict")
            tr = d.get("train_score") or d.get("train_outcomes")
            if isinstance(tr, dict):
                tr = "%d/%d" % (sum(tr.values()), len(tr))
            print("r%-2s kept=%s train=%s val %s -> %s verdict=%s edit_bytes=%s files=%s" % (
                d.get("round"), d.get("kept"), d.get("train_score"), d.get("val_score_before"),
                d.get("val_score_after"), (v or {}).get("label") if isinstance(v, dict) else v,
                d.get("edit_bytes"), ",".join(d.get("files") or [])[:60]))
            if isinstance(v, dict):
                print("    fixed=%s regressed=%s" % (v.get("fixed"), v.get("regressed")))
            s = d.get("search") or {}
            if s:
                print("    winner=%s screen=%s" % (s.get("winner_lens"),
                      [(x.get("lens"), x.get("score"), "fix%s" % (x.get("fixed") or []),
                        "reg%s" % (x.get("regressed") or [])) for x in s.get("screen", [])]))
            fc = d.get("flip_confirm") or []
            if fc:
                print("    flips confirmed %d/%d; unconfirmed: %s" % (
                    sum(1 for x in fc if x.get("confirmed")), len(fc),
                    ",".join(str(x.get("tid")) for x in fc if not x.get("confirmed"))[:100]))
            hs = d.get("hooks_stats") or {}
            if hs:
                print("    hooks: episodes=%s changed=%s llm_calls=%s" % (hs.get("episodes"), hs.get("changed"), hs.get("llm_calls")))
    except Exception as exc:
        print("no rounds.jsonl here:", repr(exc))
    sys.exit(0)

if not rows:
    print("NO_TRAJECTORIES. top-level entries:")
    for d in sorted(glob.glob("*"))[:40]:
        print("  ", d)
    sys.exit(0)

WRITE = re.compile(r"<<-?\s*\w|\bsed\b[^|;&]*-i|\bperl\s+-[ip]e|\btee\b\s|>>?\s*/\S|apply_patch|git\s+apply|\bpatch\b|\.write\(|write_text|\brm\b|\bmv\b|\bcp\b|\btouch\b|\bmkdir\b")
TEST = re.compile(r"\bpytest\b|-m\s+unittest|\btox\b|\bnosetests\b")
for r in rows:
    cnt = {}; mxrep = 0; ro = 0; maxro = 0; fw = None
    for i, c in enumerate(r["cmds"]):
        k = " ".join(c.split())
        cnt[k] = cnt.get(k, 0) + 1
        if cnt[k] > mxrep:
            mxrep = cnt[k]
        w = bool(WRITE.search(c))
        if w or TEST.search(c):
            ro = 0
            if w and fw is None:
                fw = i
        else:
            ro += 1
            if ro > maxro:
                maxro = ro
    r.update(n=len(r["cmds"]), rep=mxrep, ro=maxro,
             fw=(fw if fw is not None else -1),
             env=r["txt"].count("No module named"),
             stash=len(re.findall(r"git\s+stash", r["txt"])))
try:
    json.dump([{k: v for k, v in r.items() if k not in ("cmds", "txt")} for r in rows],
              open("/scratch/triage.json", "w"))
except Exception:
    pass
P = [r for r in rows if r["p"]]
F = [r for r in rows if not r["p"]]
def med(v):
    v = sorted(v)
    return v[len(v) // 2] if v else -1
def line(lbl, g):
    if not g:
        return
    print("%-6s n=%3d cmds_med=%4d rep>=4:%2d ro_med=%3d ro_max=%3d no_write:%2d write@med=%3d nopatch:%2d stash:%2d env_err:%4d"
          % (lbl, len(g), med([r["n"] for r in g]), sum(1 for r in g if r["rep"] >= 4),
             med([r["ro"] for r in g]), max(r["ro"] for r in g),
             sum(1 for r in g if r["fw"] < 0), med([r["fw"] for r in g if r["fw"] >= 0]),
             sum(1 for r in g if "no patch" in r["v"].lower()),
             sum(1 for r in g if r["stash"]), sum(1 for r in g if r["env"] > 0)))
print("dir=%s tasks=%d pass=%d fail=%d" % (TR, len(rows), len(P), len(F)))
line("PASS", P)
line("FAIL", F)
print("fail_nopatch=%d  fail_rep4=%d  fail_nowrite=%d  fail_with_patch=%d" % (
    sum(1 for r in F if "no patch" in r["v"].lower()),
    sum(1 for r in F if r["rep"] >= 4), sum(1 for r in F if r["fw"] < 0),
    sum(1 for r in F if r["fw"] >= 0)))
if mode == "grep" and pat:
    try:
        rx = re.compile(pat)
    except Exception as exc:
        print("bad regex:", exc)
        sys.exit(0)
    hits = 0
    for r in rows:
        for c in set(r["cmds"]):
            if rx.search(c):
                print(r["t"], "|", " ".join(c.split())[:120])
                hits += 1
                break
    print("grep_hits=%d" % hits)
else:
    src = F if mode == "fail" else F
    src = sorted(src, key=lambda r: (-r["rep"], -r["ro"]))[:25]
    for r in src:
        print("  %-46s cmds=%4d rep=%3d ro=%4d fw=%4d nopatch=%s | %s"
              % (r["t"][:46], r["n"], r["rep"], r["ro"], r["fw"],
                 "no patch" in r["v"].lower(), r["v"][:36]))
'''


class Hooks(BaseHooks):
    def __init__(self):
        self.bash = 0
        self.exp = 0
        self.left = -1
        self.turn = 0
        self.bash_since_exp = 0
        self.turns_since_exp = 0
        self.bash_since_submit = 0
        self.submits = 0
        self.blocked = 0

    def system_prompt(self, assembled):
        try:
            return assembled + CONTRACT
        except Exception:
            return assembled

    def before_tool(self, name, args, state):
        try:
            if not isinstance(args, dict):
                return args
            if name in POOL:
                self.exp += 1
                self.bash_since_exp = 0
                self.turns_since_exp = 0
                return args
            if name == "submit_proposal":
                self.submits += 1
                self.bash_since_submit = 0
                return args
            if name == "bash":
                self.bash += 1
                self.bash_since_exp += 1
                self.bash_since_submit += 1
                if self.submits:
                    if self.bash_since_submit > MAX_BASH_AFTER_SUBMIT:
                        self.blocked += 1
                        return ("BLOCKED (teacher budget): you SUBMITTED %d time(s) and since "
                                "then spent %d bash calls offline. Post-submit analysis buys "
                                "nothing unless it changes the change-set: run a pool experiment "
                                "that would falsify it, or stop here." % (self.submits, self.bash_since_submit))
                    return args
                if self.exp == 0 and self.bash > MAX_BASH_BEFORE_FIRST_EXPERIMENT:
                    self.blocked += 1
                    return ("BLOCKED (teacher budget): %d bash calls spent with ZERO executed "
                            "experiments this episode. The probe window expires on wall-clock "
                            "while you simulate offline, and rounds end with slots destroyed. "
                            "Draft your change-set in /scratch NOW and fix_probe it on 2-3 "
                            "predicted-fix tasks (triage tool for one-shot corpus evidence)." % self.bash)
                if (self.exp > 0 and (self.bash_since_exp >= MAX_BASH_BETWEEN_EXPERIMENTS
                                      or self.turns_since_exp >= MAX_TURNS_BETWEEN_EXPERIMENTS)):
                    self.blocked += 1
                    return ("BLOCKED (teacher pacing): %d turns / %d bash calls since your last "
                            "pool experiment. Round-2 candidates lost the round exactly here: "
                            "hour-long offline drafting turns while the probe pool sat full. "
                            "Either run the fix_probe your current draft predicts (2-3 tasks), "
                            "or compose the submit_proposal now with what you have." %
                            (self.turns_since_exp, self.bash_since_exp))
                return args
            return args
        except Exception:
            return args

    def after_tool(self, name, args, obs, state):
        try:
            if name in POOL:
                m = re.search(r"(?:budget left|probes left|pool left)[:=]\s*(\d+)", obs or "")
                if m:
                    self.left = int(m.group(1))
                return ("[teacher ledger: experiments=%d pool_left=%s blocked=%d] "
                        % (self.exp, self.left, self.blocked)) + (obs or "")
            if name == "submit_proposal":
                return "[submitted #%d] " % self.submits + (obs or "")
            return obs
        except Exception:
            return obs

    def on_turn_end(self, turn, state):
        try:
            self.turn = turn or 0
            self.turns_since_exp += 1
            if self.submits:
                if self.turn % 8 == 0:
                    return ("ledger: submits=%d experiments=%d - you may STOP with a passing "
                            "entry. Keep going only for pool experiments that could change the "
                            "submitted change-set; if one does, re-probe and re-submit with the "
                            "FULL file content." % (self.submits, self.exp))
                return None
            if self.exp == 0:
                return ("ledger: bash=%d experiments=0 pool_left=%s - NO EXPERIMENT YET. The "
                        "unspent pool is destroyed at the end of the round. Draft minimal edits "
                        "and fix_probe them now." % (self.bash, self.left))
            if self.turns_since_exp >= NOTE_TURNS_BETWEEN_EXPERIMENTS:
                return ("pacing: %d turns since your last pool experiment (bash_since_exp=%d; "
                        "bash is blocked at %d turns). Probes cost minutes; offline drafting "
                        "marathons cost the round. Run the fix_probe your draft predicts, or "
                        "compose your submit." % (self.turns_since_exp, self.bash_since_exp,
                                                  MAX_TURNS_BETWEEN_EXPERIMENTS))
            if self.turn % 6 == 0:
                return ("ledger: bash=%d experiments=%d submits=0 pool_left=%s - claims still "
                        "HYPOTHESIS are worthless to the next round: verify or downgrade them, "
                        "then call submit_proposal before the budget ends."
                        % (self.bash, self.exp, self.left))
            return None
        except Exception:
            return None

    def extra_tools(self):
        try:
            return [{
                "name": "triage",
                "description": ("One-shot digest over the round's evidence. mode=summary: "
                                "pass/fail tables with exact-command-repeat counts, read-only "
                                "streaks (median AND max, so guard thresholds sit above the "
                                "passing maximum), first-write index, no-patch counts, stash "
                                "usage, env-error counts, failing-task shortlist. mode=fail: "
                                "worst failing tasks only. mode=grep: which tasks ran a command "
                                "matching the regex (field 'pattern'). mode=state: compact "
                                "replay of rounds.jsonl - verdicts, val scores before/after, "
                                "fixed/regressed tasks, screening winner, flip confirmation, "
                                "hook stats. Costs one sandbox call; use it instead of ad-hoc "
                                "trajectory scanning or hand-reading the journal."),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "mode": {"type": "string", "enum": ["summary", "fail", "grep", "state"]},
                        "pattern": {"type": "string", "description": "regex over bash commands, mode=grep"},
                    },
                    "required": ["mode"],
                },
            }]
        except Exception:
            return []

    def run_tool(self, name, args, env, state):
        try:
            if name != "triage":
                return "ERROR: unknown tool " + str(name)
            mode = (args or {}).get("mode") or "summary"
            if mode not in ("summary", "fail", "grep", "state"):
                mode = "summary"
            pat = (args or {}).get("pattern") or ""
            cmd = ("python3 - " + mode + " " + shlex.quote(pat)
                   + " <<'PYEOFTRIEAGE'\n" + TRIAGE + "\nPYEOFTRIEAGE")
            return env.bash(cmd)
        except Exception as exc:
            return "triage failed: " + repr(exc)
