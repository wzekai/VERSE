"""Evidence sources: how each baseline family presents trajectories to the optimizer.

Each source implements EvidenceSource.gather(ctx) -> (report_text, claims). The report goes
into the optimizer's prompt; the claims (verse.claims.Claim) are the structured records behind
it. Verification is not decided here. When ctx.probes is set, each push source verifies its
top claims, and with the pull sources the optimizer runs experiments itself through tools.
Otherwise claims stay hypotheses. One renderer labels each statement either way.

Sources and the baseline implementations they follow:
  push_tails            simple baseline: the tail of each failed transcript (not used by the
                        released configurations)
  push_digest_layered   AHE Agent Debugger: overview.md with one root-cause line per task, plus
                        detail/{task}.md with the full analysis and the trajectory path
  push_clustered        Self-Harness: regex verifier buckets, a per-trace LLM causal record,
                        exact clustering on (terminal_cause, criticality, agent_mechanism),
                        then a brief
  pull_explore          Meta-Harness: no pre-built report; trajectory files are written to the
                        run directory and the coding-agent optimizer gets their paths
  pull_digest           HarnessX: pull_explore's files plus an on-demand
                        `digest_trajectories` tool (the HarnessX digester is a sub-worker the
                        MetaAgent spawns, not a pipeline stage)
  push_digest_flat      HarnessX approximated as a push report (not used by the released
                        configurations)
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict

from verse.claims import Claim, render_claims

from verse.evolution.protocols import EvidenceSource
from verse.evolution.registry import register_component

_ERR_NAMED_RE = re.compile(r"\b\w+(?:Error|Exception|Warning)\b|Traceback")
_ERR_LINE_RE = re.compile(r"error|failed|exception|traceback|assert", re.I)


# Shared helpers
def _last_error_line(transcript: list) -> str:
    named = generic = ""
    for m in reversed(transcript):
        if m.get("role") != "user":
            continue
        for ln in reversed((m.get("content") or "").splitlines()):
            if _ERR_NAMED_RE.search(ln):
                named = ln.strip()[:160]
                break
            if not generic and _ERR_LINE_RE.search(ln):
                generic = ln.strip()[:160]
        if named:
            break
    return named or generic


def _tail(transcript: list, chars: int = 1500) -> str:
    text = "\n".join(f"[{m['role']}] {m['content']}" for m in transcript[1:])
    return f"...{text[-chars:]}"


def _verify_top_claims(ctx, claims: list, max_probes: int = 2) -> None:
    """Verification for push sources: perturb (ablate) only claims that name a step.

    Most of the probe budget belongs to the optimizer's own forward probes (fix_probe,
    substitute), so evidence-side probes are capped at max_probes; replaying a deterministic
    failure only reproduces it. Probes run concurrently (independent containers, no LLM
    calls). Mutates claims in place."""
    if ctx.probes is None:
        return
    from concurrent.futures import ThreadPoolExecutor
    jobs = []
    for c in claims:
        if len(jobs) >= max(1, int(max_probes)):
            break
        step = c.tags.get("step")
        if step is not None and c.task_id:
            jobs.append((c, int(step)))

    def _one(job):
        c, step = job
        return c, ctx.probes.ablate(ctx, c.task_id, step)

    if not jobs:
        return
    with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as pool:
        for c, rec in pool.map(_one, jobs):
            if rec.setup != "(not run)":
                # a not-run record would read as a performed experiment in the
                # report, so it is dropped
                c.evidence.append(rec)


def _verifier_verdict_line(transcript: list) -> str:
    """The episode's ground-truth verdict line (SWE `VERIFIER ...`, TB `TEST FAILURES:` /
    `TEST RESULT:`), which the executors append to the transcript. It is not a step (not a
    bash turn), so it is surfaced explicitly; otherwise readers would see it only through the
    generic last-error regex, which an earlier traceback in the episode can shadow."""
    for m in reversed(transcript or []):
        c = m.get("content", "") if isinstance(m, dict) else ""
        if isinstance(c, str) and c.startswith(("VERIFIER", "TEST FAILURES:", "TEST RESULT:")):
            return c[:300]
    return ""


def render_trajectory_md(tid: str, transcript: list, phi: float) -> str:
    """HarnessX-style trajectory markdown (frontmatter plus numbered steps), the file that
    pull_explore and the digester read. Step indices here are the indices the probes accept."""
    from verse.runtime.b1_evidence import transcript_steps
    steps = transcript_steps(transcript)
    n_cmds = sum(1 for s in steps if s["kind"] == "bash")
    verdict = _verifier_verdict_line(transcript)
    front = [f"---", f"task_id: {tid}", f"eval_passed: {phi >= 0.5}",
             f"n_steps: {len(steps)}", f"n_commands: {n_cmds}",
             f"last_error: {_last_error_line(transcript) or '(none)'}"] \
        + ([f"verifier_verdict: {verdict}"] if verdict else [])
    # code_hooks configurations: list hook errors here too, so fail-open errors are visible
    # wherever the optimizer reads, not only in round-level stats and raw trajectory JSON
    try:
        from verse.evolution.hooks_runtime import extract_hooks_summary
        hs = extract_hooks_summary(transcript)
        if hs and hs.get("n_errors"):
            errs = "; ".join(f"{e.get('hook')}: {e.get('error')}"
                             for e in (hs.get("errors") or [])[:3])
            front.append(f"harness_code_errors: {hs['n_errors']} ({errs[:400]})")
    except Exception:
        pass
    front += [f"---", ""]
    body = [f"# Trajectory: {tid}", ""]
    # Grader-log tail (the executors append it on failure), shown as its own section before
    # the steps. AHE gives its debugger the same input (the verifier output the agent never
    # saw, last 60 lines), and the Self-Harness buckets scan verifier stdout. Rendered for
    # every configuration.
    for m in reversed(transcript or []):
        c = m.get("content", "") if isinstance(m, dict) else ""
        if isinstance(c, str) and c.startswith("GRADER LOG"):
            body += ["## Verifier output (ground truth — the agent never saw this)",
                     "```", c.split("\n", 1)[1] if "\n" in c else "(empty)", "```", ""]
            break
    for k, s in enumerate(steps):
        if s["kind"] == "bash":
            body.append(f"### step {k}\n```bash\n{s.get('command', '')}\n```")
            oi = s.get("o_idx")
            if oi is not None and oi < len(transcript):
                full = transcript[oi].get("content") or ""
                # keep head and tail: error verdicts (pytest summaries, tracebacks) appear
                # at the end of the output, so a head-only cap would hide them
                if len(full) > 1100:
                    obs = (full[:600] + f"\n<<DISPLAY-CAP: {len(full)} chars; head+tail "
                           "shown — the run itself saw the full output>>\n" + full[-500:])
                else:
                    obs = full
                body.append(f"observation:\n```\n{obs}\n```")
        else:
            body.append(f"### step {k}  [{s['kind']}]")
    return "\n".join(front + body)


def materialize_run_dir(ctx) -> str:
    """Write this round's trajectories (ctx.scratch_dir/trajectories) and outcome table: the
    files every evidence source, and the pull optimizer, works over.

    Passing trajectories are written for every configuration (the HarnessX digester also
    reads passing traces); eval_passed in the frontmatter tells them apart."""
    troot = os.path.join(ctx.scratch_dir, "trajectories")
    os.makedirs(troot, exist_ok=True)
    outcomes = ctx.train_outcomes or {}
    both = dict(ctx.fail_transcripts or {})
    both.update(getattr(ctx, "success_transcripts", None) or {})
    for tid, transcript in both.items():
        p = os.path.join(troot, f"{tid}.md")
        if not os.path.exists(p):
            open(p, "w").write(render_trajectory_md(tid, transcript, outcomes.get(tid, 0.0)))
    with open(os.path.join(ctx.scratch_dir, "outcomes.json"), "w") as f:
        json.dump(outcomes, f, indent=1)
    return troot


def _success_contrast(ctx, max_pairs: int = 6) -> str:
    """Index of same-repository pass/fail pairs, appended to every configuration's report.
    Contrasting passing with failing trajectories shows behavior that works, which failures
    alone do not reveal."""
    succ = getattr(ctx, "success_transcripts", None) or {}
    if not succ:
        return ""
    def _repo(tid):
        return tid.rsplit("-", 1)[0]
    fails_by_repo = {}
    for tid in (ctx.fail_transcripts or {}):
        fails_by_repo.setdefault(_repo(tid), []).append(tid)
    pairs = []
    for stid in sorted(succ):
        for ftid in fails_by_repo.get(_repo(stid), []):
            pairs.append((stid, ftid))
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break
    lines = [f"\n\n## Success contrast ({len(succ)} passing trajectories also in trajectories/)",
             "Passing trajectories show what WORKING behavior looks like under this harness — "
             "mine them for winning patterns worth distilling into skills."]
    if pairs:
        lines.append("Same-repo pass/fail pairs (highest-signal diffs):")
        lines += [f"- PASS trajectories/{s}.md  vs  FAIL trajectories/{f}.md" for s, f in pairs]
    else:
        lines.append("(no same-repo pass/fail pair this round; read any "
                     "eval_passed: True trajectory for contrast)")
    return "\n".join(lines)


# 1. push_tails
@register_component("evidence", "push_tails")
class PushTails(EvidenceSource):
    def __init__(self, max_tasks: int = 3, tail_chars: int = 3000):
        self.max_tasks = int(max_tasks)
        self.tail_chars = int(tail_chars)

    def gather(self, ctx):
        materialize_run_dir(ctx)
        claims, chunks = [], []
        for tid, tr in list(ctx.fail_transcripts.items())[: self.max_tasks]:
            err = _last_error_line(tr)
            claims.append(Claim(statement=f"failure tail suggests: {err or 'unclear'}",
                                task_id=tid))
            chunks.append(f"### FAILED {tid}\n{_tail(tr, self.tail_chars)}")
        _verify_top_claims(ctx, claims)
        report = (render_claims(claims, "Root-cause claims") + "\n\n" + "\n\n".join(chunks)
                  + _success_contrast(ctx))
        return report, claims


# 2. push_digest_layered (AHE)
@register_component("evidence", "push_digest_layered")
class PushDigestLayered(EvidenceSource):
    """AHE Agent Debugger: an LLM analysis of each failed task (FAILURE POINT / ROOT CAUSE /
    WHAT SHOULD HAVE BEEN DONE / GENERAL LESSON, <=300 words), rendered as overview
    one-liners plus detail files with drill-down paths. The analysis uses the optimizer's
    own model (AHE routes it through a separate QA LLM)."""

    def __init__(self, max_tasks: int = 12, qa_max_tokens: int = 1200):
        self.max_tasks = int(max_tasks)
        self.qa_max_tokens = int(qa_max_tokens)

    _QA = ("You are a trace debugger. Analyze this failed coding-agent trajectory and answer "
           "in under 300 words with exactly these sections:\n1. FAILURE POINT\n2. ROOT CAUSE\n"
           "3. WHAT SHOULD HAVE BEEN DONE\n4. GENERAL LESSON\n"
           "The verifier output below is ground truth the agent never saw — find the TRUE root "
           "cause, not the first visible error.\nIf your ROOT CAUSE names a specific step, write "
           "it as 'step K' using the step numbers shown.")

    def gather(self, ctx):
        from verse.runtime.executor import _invoke_bedrock, _TEMP_DEPRECATED
        troot = materialize_run_dir(ctx)
        model = (ctx.executor or {}).get("intervener_model") or \
            (ctx.executor or {}).get("model", "qwen38-27b")
        regions = (ctx.executor or {}).get("regions", "us-west-2,us-east-1,us-east-2")
        detail_dir = os.path.join(ctx.scratch_dir, "analysis", "detail")
        os.makedirs(detail_dir, exist_ok=True)

        claims, over_lines = [], []
        fails = list(ctx.fail_transcripts.items())[: self.max_tasks]
        for tid, tr in fails:
            _full = open(os.path.join(troot, f"{tid}.md")).read()
            traj_md = _full[:24000] + (
                f"\n<<DISPLAY-CAP: trajectory file is {len(_full)} chars; first 24000 shown>>"
                if len(_full) > 24000 else "")
            body = {"max_tokens": self.qa_max_tokens,
                    "messages": [{"role": "user", "content": f"{self._QA}\n\n{traj_md}"}]}
            if not any(t in model for t in _TEMP_DEPRECATED):
                body["temperature"] = 0.0
            try:
                qa = _invoke_bedrock(model, regions, body)
            except Exception as e:
                qa = f"ROOT CAUSE: (debugger unavailable: {type(e).__name__})"
            m = re.search(r"ROOT CAUSE[:\s]*(.+?)(?=\n\s*\d\.|\n##|\Z)", qa, re.S)
            root = (m.group(1).strip() if m else qa[:300]).replace("\n", " ")[:400]
            sm = re.search(r"step (\d+)", root)
            c = Claim(statement=root, task_id=tid,
                      tags={"step": int(sm.group(1))} if sm else {})
            claims.append(c)
            open(os.path.join(detail_dir, f"{tid}.md"), "w").write(
                f"# {tid}\n\n## QA Analysis\n{qa}\n\n## Trace Path\ntrajectories/{tid}.md\n")
            over_lines.append(f"- **{tid}**: {root[:200]}")

        _verify_top_claims(ctx, claims)
        overview = (f"# Debugger Analysis Overview — round {ctx.round_idx}\n"
                    f"Analyzed {len(fails)} failed tasks "
                    f"({len(ctx.fail_transcripts)} total failures this round).\n\n"
                    + "\n".join(over_lines)
                    + "\n\nPer-task detail: analysis/detail/{task}.md ; raw: trajectories/{task}.md")
        open(os.path.join(ctx.scratch_dir, "analysis", "overview.md"), "w").write(overview)
        report = (overview + "\n\n" + render_claims(claims, "Root-cause claims")
                  + _success_contrast(ctx))
        return report, claims


# 3. push_clustered (Self-Harness)
@register_component("evidence", "push_clustered")
class PushClustered(EvidenceSource):
    """Self-Harness: a rule-based terminal bucket, a one-shot LLM causal record per trace
    (strict JSON: terminal_cause / criticality / agent_mechanism / incorrect_step_ids),
    exact string clustering on the signature triple, and a brief with per-cluster
    representatives."""

    def __init__(self, max_tasks: int = 20, max_clusters: int = 6):
        self.max_tasks = int(max_tasks)
        self.max_clusters = int(max_clusters)

    _BUCKETS = [  # (bucket, regex over the tail), priority order of Self-Harness trace.py
        ("missing_required_artifact", r"filenotfounderror|no such file|does not exist|not found"),
        ("missing_dependency", r"modulenotfounderror|importerror|no module named"),
        ("agent_timeout", r"timed out|timeout"),
        ("verifier_runtime_error", r"runtimeerror|valueerror|typeerror"),
        ("verifier_assertion", r"assertionerror|expected|mismatch"),
    ]

    _LLM = ("Analyze this failed agent trace. Return ONLY JSON: "
            '{"terminal_cause": "<snake_case>", "criticality": '
            '"root_cause|contributor|non_terminal_friction|unknown", '
            '"agent_mechanism": "<snake_case>", "incorrect_step_ids": [<int>...], '
            '"reasoning": "<short>"}. terminal_cause must name the causal terminal signature, '
            "not just restate the error class.")

    def _bucket(self, transcript: list) -> str:
        text = _tail(transcript, 3000).lower()
        for name, rx in self._BUCKETS:
            if re.search(rx, text):
                return name
        return "reward_zero"

    def gather(self, ctx):
        from verse.runtime.executor import (_invoke_bedrock, _json_candidates,
                                                  _TEMP_DEPRECATED)
        materialize_run_dir(ctx)
        model = (ctx.executor or {}).get("intervener_model") or \
            (ctx.executor or {}).get("model", "qwen38-27b")
        regions = (ctx.executor or {}).get("regions", "us-west-2,us-east-1,us-east-2")
        records = []
        for tid, tr in list(ctx.fail_transcripts.items())[: self.max_tasks]:
            bucket = self._bucket(tr)
            body = {"max_tokens": 800,
                    "messages": [{"role": "user", "content":
                                  f"{self._LLM}\n\nterminal bucket (rule-based): {bucket}\n\n"
                                  f"{_tail(tr, 6000)}"}]}
            if not any(t in model for t in _TEMP_DEPRECATED):
                body["temperature"] = 0.0
            rec = {"terminal_cause": bucket, "criticality": "unknown",
                   "agent_mechanism": "unknown", "incorrect_step_ids": [], "reasoning": ""}
            try:
                reply = _invoke_bedrock(model, regions, body)
                for cand in _json_candidates(reply):
                    try:
                        d = json.loads(cand)
                        if isinstance(d, dict) and "terminal_cause" in d:
                            rec.update({k: d[k] for k in rec if k in d})
                            break
                    except Exception:
                        continue
            except Exception:
                pass
            rec["task_id"], rec["bucket"] = tid, bucket
            records.append(rec)

        clusters = defaultdict(list)          # exact triple key, as in Self-Harness
        for r in records:
            clusters[(r["terminal_cause"], r["criticality"], r["agent_mechanism"])].append(r)
        rank = {"root_cause": 0, "contributor": 1, "unknown": 2, "non_terminal_friction": 3}
        ordered = sorted(clusters.items(),
                         key=lambda kv: (rank.get(kv[0][1], 2), -len(kv[1])))[: self.max_clusters]

        claims, sections = [], []
        for (cause, crit, mech), members in ordered:
            rep = members[0]
            step = (rep.get("incorrect_step_ids") or [None])[0]
            c = Claim(statement=f"cluster {cause}/{crit}/{mech} ({len(members)} tasks): "
                                f"{rep.get('reasoning', '')[:300]}",
                      task_id=rep["task_id"],
                      tags={"cluster_size": len(members), "signature": [cause, crit, mech],
                            **({"step": step} if step is not None else {})})
            claims.append(c)
            sections.append(
                f"### Cluster: {cause} / {crit} / {mech}  ({len(members)} tasks)\n"
                + "\n".join(f"- {m['task_id']}: {m.get('reasoning', '')[:160]}"
                            for m in members[:5]))
        _verify_top_claims(ctx, claims)
        report = ("# Verifier-Causal Diagnosis Brief\n"
                  f"{len(records)} failed traces -> {len(clusters)} clusters "
                  f"(signature = terminal_cause / criticality / agent_mechanism)\n\n"
                  + render_claims(claims, "Cluster root-cause claims")
                  + "\n\n" + "\n\n".join(sections) + _success_contrast(ctx))
        return report, claims


# 4. pull_explore (Meta-Harness)
@register_component("evidence", "pull_explore")
class PullExplore(EvidenceSource):
    """Meta-Harness: no pre-digested report. Writes the trajectory files and returns
    pointers plus a failure index; the coding-agent optimizer explores them itself (and,
    when probes are enabled, runs experiments). Claims come back attached to the proposal,
    not from here."""

    def __init__(self, index_lines: int = 80):
        self.index_lines = int(index_lines)

    def gather(self, ctx):
        materialize_run_dir(ctx)
        outcomes = ctx.train_outcomes or {}
        fails = [t for t, v in outcomes.items() if v < 0.5]
        lines = [f"- trajectories/{tid}.md — {_last_error_line(ctx.fail_transcripts.get(tid, [])) or 'no error line'}"
                 for tid in fails[: self.index_lines] if tid in ctx.fail_transcripts]
        report = (
            f"## Run directory\n"
            f"- trajectories/  — one .md per task, PASSING and FAILED (frontmatter says "
            f"eval_passed; numbered steps + observations)\n"
            f"- outcomes.json  — every task's pass/fail this round\n"
            f"- history.json   — your previous proposals and their verdicts\n\n"
            f"## Failure index ({len(fails)} failed / {len(outcomes)} total)\n"
            + "\n".join(lines)
            + "\n\nExplore with list_dir/read_file/grep. Look for patterns shared across many "
              "failures before deep-reading."
            + _success_contrast(ctx))
        return report, []


# 5. push_digest_flat (HarnessX digester as a push report)
@register_component("evidence", "push_digest_flat")
class PushDigestFlat(EvidenceSource):
    """HarnessX trajectory digester run as a push report (prompt contract of
    workers/trajectory_digester.py): one cross-task digest with five fixed sections (by exit
    reason / by eval outcome / tool health / capability gaps / candidate hypotheses), <=1500
    words, quoting evidence, never inventing task ids. One LLM call over the batch (AHE makes
    one call per task)."""

    def __init__(self, max_tasks: int = 25, max_tokens: int = 2000):
        self.max_tasks = int(max_tasks)
        self.max_tokens = int(max_tokens)

    _PROMPT = ("You are a trajectory digester. Input: excerpts of failed agent trajectories. "
               "Output a single markdown digest, terse and evidence-quoting, under ~1500 words, "
               "with exactly these sections:\n"
               "## By termination\n## By failure symptom\n## Tool health\n"
               "## Named capability gaps\n## Candidate hypotheses\n"
               "Each hypothesis bullet: a lever the harness engineer might pull, formatted "
               "'<snake_case_name> — <one line> (e.g. <task_id>)' where <task_id> is ONE input "
               "task exhibiting the mechanism — required, so the hypothesis can be tested by "
               "experiment; add 'step K of <task_id>' when a specific step is implicated. "
               "Use only task ids present in the input; never invent them.")

    def gather(self, ctx):
        from verse.runtime.executor import _invoke_bedrock, _TEMP_DEPRECATED
        materialize_run_dir(ctx)
        model = (ctx.executor or {}).get("intervener_model") or \
            (ctx.executor or {}).get("model", "qwen38-27b")
        regions = (ctx.executor or {}).get("regions", "us-west-2,us-east-1,us-east-2")
        chunks = []
        for tid, tr in list(ctx.fail_transcripts.items())[: self.max_tasks]:
            chunks.append(f"### {tid}\nlast_error: {_last_error_line(tr)}\n{_tail(tr, 1200)}")
        body = {"max_tokens": self.max_tokens,
                "messages": [{"role": "user",
                              "content": self._PROMPT + "\n\n" + "\n\n".join(chunks)}]}
        if not any(t in model for t in _TEMP_DEPRECATED):
            body["temperature"] = 0.0
        try:
            digest = _invoke_bedrock(model, regions, body)
        except Exception as e:
            digest = f"(digester unavailable: {type(e).__name__})"
        claims = []
        in_hyp = False
        for ln in digest.splitlines():
            if ln.strip().startswith("## "):
                in_hyp = "hypotheses" in ln.lower()
                continue
            if in_hyp and ln.strip().startswith(("-", "*")):
                stmt = ln.strip().lstrip("-* ")[:400]
                m = re.search(r"step (\d+) of (\S+?)[\s,.)]", stmt + " ")
                tags = {"step": int(m.group(1))} if m else {}
                tid = m.group(2) if m else ""
                if not tid:
                    # exemplar task id: any known task id named in the bullet (the prompt
                    # requires one); without it the claim cannot be verified
                    for cand in re.findall(r"[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-\d+", stmt):
                        if cand in ctx.insts_by_tid:
                            tid = cand
                            break
                claims.append(Claim(statement=stmt, task_id=tid, tags=tags))
        _verify_top_claims(ctx, claims)
        report = (digest + "\n\n" + render_claims(claims, "Hypothesis claims")
                  + _success_contrast(ctx))
        return report, claims


# 6. pull_digest (HarnessX)
@register_component("evidence", "pull_digest")
class PullDigest(PullExplore):
    """HarnessX: the digester is not a pipeline stage but a leaf sub-worker that the
    MetaAgent spawns on demand (spawn_reflect_worker) when there are too many trajectories to
    read. The evidence interface is therefore pull (the optimizer reads trajectory files, as
    in pull_explore) plus an on-demand `digest_trajectories` tool, added to the optimizer
    episode through intervener_tools().

    Notes on the reimplementation:
      - The digest keeps the HarnessX 5-section worker prompt contract (by exit reason / by
        eval outcome / tool health / capability gaps / candidate artifact hypotheses,
        <=1500 words, quoting evidence, never inventing task ids).
      - The HarnessX WorkerSpec digester is a read-only 20-step agent; here it is one
        bounded LLM call over rendered excerpts (the same simplification as
        push_digest_flat). Calls per episode are capped (HarnessX spawns it when needed,
        not every turn).
      - The tool description keeps the grounding rule: the digest is an index, not
        evidence, and final claims must cite trajectories the optimizer actually read."""

    def __init__(self, index_lines: int = 80, max_tasks: int = 25,
                 max_tokens: int = 2000, max_calls: int = 4):
        super().__init__(index_lines=index_lines)
        self.max_tasks = int(max_tasks)
        self.max_tokens = int(max_tokens)
        self.max_calls = int(max_calls)

    _PROMPT = ("You are a trajectory digester (a read-only sub-worker). Input: excerpts of "
               "agent trajectories, passing and failed. Output a single markdown digest, "
               "terse and evidence-quoting, under ~1500 words, with exactly these sections:\n"
               "## By exit reason\n## By eval outcome\n## Tool health\n"
               "## Named capability gaps\n## Candidate artifact hypotheses\n"
               "Failed tasks' 'why' must come from behavior visible in the trajectory. Each "
               "hypothesis bullet: '<snake_case_name> — <one line> (e.g. <task_id>)' naming "
               "ONE input task exhibiting the mechanism; add 'step K of <task_id>' when a "
               "specific step is implicated. Use only task ids present in the input; never "
               "invent them. Every claim must be traceable to a trajectory you actually "
               "read; when in doubt, omit.")

    def gather(self, ctx):
        report, claims = super().gather(ctx)
        report += (
            "\n\nA `digest_trajectories` tool is mounted: it runs a digester sub-worker "
            "over many trajectories at once (5-section summary). Use it when the failure "
            "set is too large to read one by one; treat its output as an INDEX — ground "
            "your final claims in trajectories you actually read.")
        return report, claims

    def intervener_tools(self, ctx):
        """Return one tool (spec, fn). The driver calls this once per optimizer episode, so
        each episode has its own call counter (best-of-N candidates run concurrently)."""
        spec = {
            "name": "digest_trajectories",
            "description": (
                "Spawn a digester sub-worker over trajectories: returns one 5-section "
                "markdown digest (by exit reason / by eval outcome / tool health / "
                "capability gaps / candidate hypotheses). Pass task_ids to digest a "
                "subset, or omit for all failed tasks (capped) plus a few passing ones. "
                "The digest is an INDEX, not evidence — cite trajectories you read, "
                f"not the digest. Max {self.max_calls} calls per episode."),
            "input_schema": {"type": "object", "properties": {
                "task_ids": {"type": "array", "items": {"type": "string"},
                             "description": "optional subset (passing or failed)"}},
                "required": []}}
        state = {"calls": 0}

        def _run(args: dict) -> str:
            from verse.runtime.executor import _invoke_bedrock, _TEMP_DEPRECATED
            state["calls"] += 1
            if state["calls"] > self.max_calls:
                return (f"ERROR: digest budget ({self.max_calls} calls/episode) exhausted "
                        "— read trajectories directly with read_file/grep")
            both = dict(ctx.fail_transcripts or {})
            passing = dict(getattr(ctx, "success_transcripts", None) or {})
            asked = [str(t) for t in (args.get("task_ids") or []) if str(t).strip()]
            if asked:
                sel = [(t, both.get(t) or passing.get(t)) for t in asked]
                unknown = [t for t, tr in sel if tr is None]
                sel = [(t, tr) for t, tr in sel if tr is not None][: self.max_tasks]
                note = (f"(no trajectory for: {', '.join(unknown[:6])})\n\n"
                        if unknown else "")
            else:
                sel = list(both.items())[: self.max_tasks]
                sel += list(passing.items())[: max(2, self.max_tasks // 8)]
                note = ""
            if not sel:
                return "ERROR: no trajectories matched"
            chunks = []
            for tid, tr in sel:
                passed = tid in passing and tid not in both
                chunks.append(f"### {tid} (eval_passed: {passed})\n"
                              f"last_error: {_last_error_line(tr)}\n{_tail(tr, 1200)}")
            model = (ctx.executor or {}).get("intervener_model") or \
                (ctx.executor or {}).get("model", "qwen38-27b")
            regions = (ctx.executor or {}).get("regions", "us-west-2,us-east-1,us-east-2")
            body = {"max_tokens": self.max_tokens,
                    "messages": [{"role": "user",
                                  "content": self._PROMPT + "\n\n" + "\n\n".join(chunks)}]}
            if not any(t in model for t in _TEMP_DEPRECATED):
                body["temperature"] = 0.0
            try:
                digest = _invoke_bedrock(model, regions, body)
            except Exception as e:
                return f"ERROR: digester unavailable: {type(e).__name__}"
            return (note + digest
                    + f"\n\n(digest calls used: {state['calls']}/{self.max_calls})")

        return [{"spec": spec, "fn": _run}]
