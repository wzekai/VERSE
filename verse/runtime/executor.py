"""Executor episode loop (native tool calling) and LLM transport.

Contains the LLM transport with retry and backoff (a local vLLM server speaking the Messages
API, or a hosted model through the Bedrock Converse API), the native tool-calling SWE episode
(run_task), the text-only fallback loop (_run_task_text), and JSON extraction from model
replies (_json_candidates).
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request

from verse.runtime import swe_judge as _judge
from verse.runtime.swe_env import make_session, parse_action, truncate_observation

BASE_PROMPT = (
    "You are a software engineering agent working inside a git repository checked out at /testbed.\n"
    "You have a `bash` tool (runs shell commands; state persists across calls) and a `submit` tool.\n"
    "Workflow: (1) explore the repo to locate the code the issue describes (grep/find/cat); "
    "(2) edit the source files to fix it (sed/python/here-docs — do NOT just write a test); "
    "(3) run the relevant tests to verify your fix; (4) call `submit` when the fix is complete.\n"
    "Make focused, minimal edits to the actual source. Always call the bash tool to act — do not "
    "describe commands in prose."
)

def _chat(url: str, model: str, messages: list, max_tokens: int = 4096, timeout: int = 300,
          temperature: float = 0.0, n: int = 1) -> str | list:
    """One chat-completions request (used by the text-only loop). n=1 returns a str; n>1
    returns a list of n candidates from one vLLM request that share the prompt prefix (the
    server computes the prefix KV cache once)."""
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens,
                       "temperature": temperature, "n": n}).encode()
    req = urllib.request.Request(f"{url.rstrip('/')}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        choices = json.load(r)["choices"]
    if n == 1:
        return choices[0]["message"]["content"] or ""
    return [c["message"]["content"] or "" for c in choices]


_BR_CLIENTS: dict = {}
_BR_LOCK = threading.Lock()

# Model-id substrings for which requests omit `temperature`, so the server applies the model's
# default sampling. The optimizer (qwen38-flash-next) runs in thinking mode, where Qwen documents
# greedy decoding as unsafe (endless repetition); without temperature the server uses the
# checkpoint's generation_config (T=1.0, top_p 0.95, top_k 20). Executors are not listed and
# keep temperature 0.
_TEMP_DEPRECATED = ("qwen38-flash-next",)


# Local vLLM servers. vLLM also serves the Messages API at /v1/messages and returns content
# blocks, tool_use blocks and stop_reason (end_turn / tool_use / max_tokens) in the shape the
# tool loop reads, so no translation is needed: callers build Messages-format bodies and this
# transport sends them as-is (plus `model`). The launcher sets which model ids are served
# locally, and where:
#   T2E_VLLM_MODELS="qwen38-flash-next=http://127.0.0.1:8100,qwen35-9b=http://127.0.0.1:8101"
#   several replicas of one model: "qwen38-27b=http://127.0.0.1:8101|http://127.0.0.1:8102|..."
# Models not listed go to the Bedrock Converse API; the region pool is ignored for listed
# models.
def _vllm_base(model_id: str, body: dict | None = None, salt: int = 0) -> str:
    """Base URL of a locally served model ("" when the model is not listed).

    A served name may list several identical replicas, "name=url1|url2|...".
    The replica is chosen by a stable hash of the conversation's system prompt
    and first message, so every turn of one episode goes to the same replica and
    its prefix cache stays warm. `salt` (the retry attempt) rotates the choice so
    a dead replica does not absorb all retries. The request body is not changed.
    """
    for item in os.environ.get("T2E_VLLM_MODELS", "").split(","):
        name, _, urls = item.strip().partition("=")
        if name and urls and name == model_id:
            pool = [u.strip().rstrip("/") for u in urls.split("|") if u.strip()]
            if len(pool) == 1 or not body:
                return pool[salt % len(pool)]
            msgs = body.get("messages") or []
            key = json.dumps([body.get("system", ""), msgs[0] if msgs else None],
                             sort_keys=True, default=str)
            h = int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16)
            return pool[(h + salt) % len(pool)]
    return ""


def _invoke_vllm(base: str, model_id: str, body: dict) -> dict:
    """One Messages-format request to a local vLLM server. Returns the parsed response
    (content blocks and stop_reason). Raises urllib.error.HTTPError or URLError for the
    caller's retry loop (429, 5xx, or connection refused while the server restarts)."""
    b = dict(body)
    b["model"] = model_id
    req = urllib.request.Request(f"{base}/v1/messages", data=json.dumps(b).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    # long optimizer generations (32k output tokens, thinking mode) can run for minutes
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            parsed = json.loads(r.read())
    except (ConnectionError, http.client.HTTPException) as e:
        # a reset or disconnect mid-response is a transport fault: raise it as URLError so the
        # retry loop retries it
        raise urllib.error.URLError(e)
    u = parsed.get("usage") or {}
    _tok_add(u.get("input_tokens"), u.get("output_tokens"))
    return parsed


def _botocore_session():
    """The credential chain shared by every Bedrock call in this process.

    T2E_BEDROCK_ASSUME_ROLE (optional): assume the given role (temporary STS credentials,
    auto-refreshed via botocore) for Bedrock calls.
    T2E_BEDROCK_EXTERNAL_ID (optional): ExternalId for the AssumeRole call."""
    from botocore.session import Session as BcSession
    bc = BcSession()
    role = os.environ.get("T2E_BEDROCK_ASSUME_ROLE", "").strip()
    if role:
        from botocore.credentials import (
            AssumeRoleCredentialFetcher, DeferredRefreshableCredentials)
        extra = {"RoleSessionName": "t2e-xacct-bedrock", "DurationSeconds": 3600}
        ext_id = os.environ.get("T2E_BEDROCK_EXTERNAL_ID", "").strip()
        if ext_id:
            extra["ExternalId"] = ext_id
        fetcher = AssumeRoleCredentialFetcher(
            client_creator=bc.create_client,
            source_credentials=bc.get_credentials(),
            role_arn=role,
            extra_args=extra)
        bc._credentials = DeferredRefreshableCredentials(
            method="assume-role", refresh_using=fetcher.fetch_credentials)
    return bc


def _bedrock_client(region: str):
    """One botocore client per region, shared across threads (botocore clients are
    thread-safe)."""
    with _BR_LOCK:
        if region not in _BR_CLIENTS:
            import boto3
            from botocore.config import Config
            cfg = Config(retries={"max_attempts": 1}, read_timeout=600,
                         max_pool_connections=32)
            sess = boto3.Session(botocore_session=_botocore_session())
            _BR_CLIENTS[region] = sess.client(
                "bedrock-runtime", region_name=region, config=cfg)
        return _BR_CLIENTS[region]


def _invoke_bedrock(model_id: str, region: str, body: dict, *, max_retries: int = 10) -> str:
    """One model call with exponential backoff and full jitter on throttling and transient
    errors; returns the reply text. `region` may be a comma-separated pool
    ("us-west-2,us-east-1"): each call starts at a random pool member and each retry moves to
    the next, which spreads quota across regions and avoids a saturated one. Ignored for models
    served by vLLM."""
    body = _invoke_bedrock_raw(model_id, region, body, max_retries=max_retries)
    # text blocks only; the tool-calling loop reads the raw response via _invoke_bedrock_raw
    parts = [b.get("text", "") for b in body.get("content", []) if b.get("type") == "text"]
    return "\n".join(p for p in parts if p) if parts else ""


def _to_converse(body: dict) -> tuple:
    """Convert a Messages-format body to (messages, system, inferenceConfig, toolConfig) for
    converse(). Callers always build Messages-format bodies, so any hosted model (e.g. GLM,
    Qwen, DeepSeek, Mistral) runs the same tool loop; the translation lives only here."""
    sys_txt = body.get("system") or ""
    msgs = []
    for m in body.get("messages", []):
        c = m.get("content")
        blocks = []
        if isinstance(c, str):
            blocks = [{"text": c.strip() or "(no output)"}]
        else:
            for b in c or []:
                t = b.get("type")
                if t == "text":
                    blocks.append({"text": (b.get("text") or "").strip() or "(no output)"})
                elif t == "tool_use":
                    # Some models (e.g. glm-5) occasionally emit tool names the API rejects
                    # (over 64 chars or illegal characters). Once such a turn is in the history,
                    # every later converse() call fails with a 400. Sanitize the name when
                    # sending history back: the tool call already happened, so the name only
                    # has to satisfy the API.
                    _nm = re.sub(r"[^a-zA-Z0-9_-]", "_", (b.get("name") or "tool"))[:64]
                    blocks.append({"toolUse": {"toolUseId": b.get("id"), "name": _nm,
                                               "input": b.get("input") or {}}})
                elif t == "tool_result":
                    rc = b.get("content")
                    rtxt = rc if isinstance(rc, str) else "\n".join(
                        x.get("text", "") for x in (rc or []) if isinstance(x, dict))
                    blocks.append({"toolResult": {"toolUseId": b.get("tool_use_id"),
                                                  "content": [{"text": (rtxt or "").strip() or "(no output)"}]}})
        if blocks:
            msgs.append({"role": m.get("role"), "content": blocks})
    inf = {"maxTokens": int(body.get("max_tokens", 4096))}
    if "temperature" in body:
        inf["temperature"] = float(body["temperature"])
    tool_cfg = None
    if body.get("tools"):
        tool_cfg = {"tools": [{"toolSpec": {"name": t["name"], "description": t.get("description", ""),
                                            "inputSchema": {"json": t["input_schema"]}}}
                              for t in body["tools"]]}
        tc = body.get("tool_choice")
        if isinstance(tc, dict) and tc.get("type") == "tool" and tc.get("name"):
            # Messages-format {"type": "tool", "name": ...} maps to converse toolChoice (used to
            # force the optimizer's final-turn submit). If a model's backend rejects toolChoice,
            # the call raises and the caller falls back to an instruction in the prompt.
            tool_cfg["toolChoice"] = {"tool": {"name": tc["name"]}}
    return msgs, ([{"text": sys_txt}] if sys_txt else None), inf, tool_cfg


def _from_converse(resp: dict) -> dict:
    """Convert a converse() response to the Messages-format shape (content blocks and
    stop_reason)."""
    out = (resp.get("output") or {}).get("message") or {}
    content = []
    for b in out.get("content") or []:
        if "text" in b:
            content.append({"type": "text", "text": b["text"]})
        elif "toolUse" in b:
            tu = b["toolUse"]
            content.append({"type": "tool_use", "id": tu.get("toolUseId"),
                            "name": tu.get("name"), "input": tu.get("input") or {}})
    stop = resp.get("stopReason") or ""
    return {"content": content, "stop_reason": {"end_turn": "end_turn", "tool_use": "tool_use",
                                                "max_tokens": "max_tokens"}.get(stop, stop)}


# Token accounting. Responses carry usage counts; they are summed per thread (each episode runs
# on its own thread in sweep()) so sweeps can log token usage. Accounting only.
_TOKENS = threading.local()


def _tok_reset():
    _TOKENS.calls, _TOKENS.inp, _TOKENS.out = 0, 0, 0


def _tok_read() -> dict:
    return {"calls": getattr(_TOKENS, "calls", 0), "input": getattr(_TOKENS, "inp", 0),
            "output": getattr(_TOKENS, "out", 0)}


def _tok_add(inp, out):
    if not hasattr(_TOKENS, "calls"):
        _tok_reset()
    _TOKENS.calls += 1
    _TOKENS.inp += int(inp or 0)
    _TOKENS.out += int(out or 0)


def _invoke_bedrock_raw(model_id: str, region: str, body: dict, *, max_retries: int = 10) -> dict:
    """Like _invoke_bedrock but returns the full parsed response (content blocks and
    stop_reason), so the tool-calling loop can read tool_use blocks. Same region pool and retry
    logic. Models listed in T2E_VLLM_MODELS go to their local vLLM server."""
    regions = [r.strip() for r in (region or "us-west-2").split(",") if r.strip()]
    start = random.randrange(len(regions))
    delay = 2.0
    for attempt in range(max_retries + 1):
        rgn = regions[(start + attempt) % len(regions)]
        try:
            vllm_base = _vllm_base(model_id, body, attempt)
            if vllm_base:
                return _invoke_vllm(vllm_base, model_id, body)
            rt = _bedrock_client(rgn)
            msgs, system, inf, tool_cfg = _to_converse(body)
            kw = {"modelId": model_id, "messages": msgs, "inferenceConfig": inf}
            if system:
                kw["system"] = system
            if tool_cfg:
                kw["toolConfig"] = tool_cfg
            raw = rt.converse(**kw)
            u = raw.get("usage") or {}
            _tok_add(u.get("inputTokens"), u.get("outputTokens"))
            return _from_converse(raw)
        except Exception as e:
            # some exceptions carry response=None; check the type so a retriable timeout does not
            # turn into an AttributeError
            resp = getattr(e, "response", None)
            code = (resp.get("Error") or {}).get("Code", "") if isinstance(resp, dict) else ""
            # the vLLM transport surfaces HTTP status instead of botocore error codes:
            # 429 throttle, 5xx transient, both retriable
            http_status = getattr(e, "code", 0) if type(e).__name__ == "HTTPError" else 0
            retriable = code in ("ThrottlingException", "TooManyRequestsException",
                                 "ServiceUnavailableException", "ModelNotReadyException",
                                 "InternalServerException") or "Throttl" in str(e) \
                or http_status == 429 or 500 <= http_status < 600 \
                or "timed out" in str(e).lower() or "timeout" in type(e).__name__.lower() \
                or "ConnectionClosed" in type(e).__name__ \
                or "EndpointConnection" in type(e).__name__ \
                or "URLError" in type(e).__name__ \
                or "model identifier is invalid" in str(e)
            # "model identifier is invalid" means the model is not available in this region.
            # Each retry moves to the next region in the pool; a truly wrong model id still
            # fails once retries are exhausted. ConnectionClosedError means the TCP stream
            # dropped mid-response (e.g. an idle timeout on a long generation); it is a
            # transient transport fault, and not retrying it would lose the episode.
            if not retriable or attempt == max_retries:
                raise
            sleep_s = random.uniform(0, delay)               # full jitter
            print(f"[llm] {code or type(e).__name__} -> retry {attempt+1}/{max_retries} "
                  f"in {sleep_s:.1f}s", flush=True)
            time.sleep(sleep_s)
            delay = min(delay * 2, 60.0)
    raise RuntimeError("unreachable")


# Executor tool schema. The executor acts through native tool calls (Messages API tool_use
# blocks); its text is never parsed for actions. A prompt/parser format mismatch (e.g. a model
# writing <bash> tags to a parser that expects ```bash fences, so no command runs) cannot happen
# when the action is a typed tool call.
_BASH_TOOL = {
    "name": "bash",
    "description": ("Run a bash command inside the target repository at /testbed. Use this to explore "
                    "the repo (ls, grep, cat, find), edit source files (via sed/python/here-docs), and "
                    "run tests. State persists across calls (same shell/container). You may call this "
                    "multiple times in one turn for independent commands."),
    "input_schema": {"type": "object",
                     "properties": {"command": {"type": "string", "description": "the shell command"}},
                     "required": ["command"]},
}
_SUBMIT_TOOL = {
    "name": "submit",
    "description": ("Submit your solution. Call this ONLY after you have edited the source files to fix "
                    "the issue and verified the fix. Your accumulated edits (git diff of /testbed) "
                    "become the final patch. Takes no arguments."),
    "input_schema": {"type": "object", "properties": {}},
}
_EXECUTOR_TOOLS = [_BASH_TOOL, _SUBMIT_TOOL]
_KEEP_FULL_TURNS = 12          # keep the most recent N turns' observations at full length
_SHRUNK_STUB = "[older observation elided to fit context — re-run the command if you need it again]"

# Tool-call markup emitted as text (weak executors under context pressure): DSML tags, raw
# function_calls XML, <tool_call>-style markup. Such a turn is a failed tool call, not a
# deliberate finish, but it can arrive with stop_reason=end_turn, where an "end_turn means done"
# rule would end the episode mid-work. Detect the markup and nudge, whatever the stop_reason.
_FAKE_TOOLCALL_RE = re.compile(
    r"DSML|<\s*function_calls|<\s*tool_call|<\s*invoke\b|antml:|"
    r"<\s*bash\b|```json\s*\{\s*\"(?:name|tool)\"", re.I)


def _looks_like_failed_toolcall(texts: list) -> bool:
    return any(t and _FAKE_TOOLCALL_RE.search(t) for t in texts)


def _shrink_old_observations(msgs: list, keep_recent_msgs: int) -> None:
    """Replace tool_result content in messages older than the last keep_recent_msgs with a short
    stub, so a long episode's history fits the context. The structure (tool_use/tool_result
    pairing, block count) is kept, because the API rejects a tool_use without its tool_result."""
    cutoff = max(0, len(msgs) - keep_recent_msgs)
    for m in msgs[:cutoff]:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for blk in content:
            if isinstance(blk, dict) and blk.get("type") == "tool_result":
                c = blk.get("content")
                if isinstance(c, str) and len(c) > len(_SHRUNK_STUB):
                    blk["content"] = _SHRUNK_STUB


def run_task(inst: dict, system_prompt: str, url: str, model: str, *,
             max_turns: int = 250, bedrock_region: str = "",
             steps_out: list | None = None, hooks_src: str | None = None) -> tuple[float, list]:
    """One executor episode under the given harness prompt. Returns (phi, transcript).

    url='bedrock' runs the native tool-calling loop (routed to vLLM or Bedrock by
    _invoke_bedrock_raw); any other url runs the text-only loop against that endpoint. If
    steps_out is given, a per-turn record (text, commands, observations) is appended for
    logging. hooks_src is the source of harness_code/hooks.py for code_hooks configurations and
    adds the five-hook executable layer (hooks_runtime); None runs the loop without hooks."""
    image = _judge._image_name(inst)
    session = make_session(image=image)
    problem = (inst.get("problem_statement") or "")[:8000]
    # code_hooks layer (None for markdown-only harnesses). A hooks.py that fails to load is a
    # broken harness, not an infra failure: the episode runs without hooks and the load error is
    # recorded in the transcript, so the episode is still scored, and the verification tool and
    # the next optimizer round can see why the code layer had no effect.
    hooks_rt, hooks_load_error = None, None
    if hooks_src:
        from verse.evolution.hooks_runtime import (
            HookedRuntime, HookLoadError, BashCapability, summary_transcript_entry)
        try:
            hooks_rt = HookedRuntime(hooks_src)
        except HookLoadError as e:
            hooks_load_error = str(e)[:600]
    if hooks_rt is not None:
        system_prompt = hooks_rt.system_prompt(system_prompt)
    # readable transcript ({role, content: str}) used for evidence, logging and continuation
    transcript = [{"role": "system", "content": system_prompt},
                  {"role": "user", "content": f"Issue:\n{problem}"}]
    if hooks_load_error:
        transcript.append({"role": "system",
                           "content": f"HARNESS_CODE_LOAD_ERROR: {hooks_load_error}"})
    patch = ""
    if url != "bedrock":
        # text-only executor loop against a chat-completions endpoint
        return _run_task_text(inst, image, session, system_prompt, transcript, url, model,
                              max_turns=max_turns, steps_out=steps_out)
    region = bedrock_region or "us-west-2"
    # Per-episode wall-clock cap: a few instances (slow containers combined with per-command
    # timeouts) can run for hours and hold a sweep thread. At the deadline the episode stops as
    # on max_turns exhaustion, and the accumulated edits are graded.
    deadline = time.monotonic() + float(os.environ.get("T2E_EPISODE_WALL_S", "2700"))
    # Loop settings and hook LLM budget (code_hooks only): validated loop settings (max_turns
    # may only shrink; see hooks_runtime.LOOP_KNOBS) and an LLM capability for hooks whose calls
    # share the turn budget (agent turns + hook LLM calls <= max_turns), so a harness cannot buy
    # extra compute.
    obs_cap, keep_full, spam_cap = 4096, _KEEP_FULL_TURNS, 6
    nudge_no_tool = ("You did not call a tool. Call `bash` to run a command, or `submit` "
                     "when the fix is complete and verified.")
    nudge_bad_markup = ("Your tool call did not parse — emit a NATIVE tool call (the bash/"
                        "submit tools), not XML or markup in your text.")
    tools = _EXECUTOR_TOOLS
    cap = None
    if hooks_rt is not None:
        lc = hooks_rt.loop_config(max_turns)
        max_turns = lc.get("max_turns", max_turns)
        obs_cap = lc.get("obs_cap", obs_cap)
        keep_full = lc.get("keep_full_turns", keep_full)
        spam_cap = lc.get("spam_streak", spam_cap)
        nudge_no_tool = lc.get("nudge_no_tool", nudge_no_tool)
        nudge_bad_markup = lc.get("nudge_bad_markup", nudge_bad_markup)
        def _hook_llm(prompt, mt):
            b = {"max_tokens": mt,
                 "messages": [{"role": "user", "content": prompt}]}
            if not any(t in model for t in _TEMP_DEPRECATED):
                b["temperature"] = 0.0
            r = _invoke_bedrock_raw(model, region, b)
            return "\n".join(x.get("text", "") for x in r.get("content", [])
                             if x.get("type") == "text")
        hooks_rt.attach_llm(_hook_llm, max_calls=max_turns // 2)
        extra_specs = hooks_rt.extra_tool_specs()
        if extra_specs:
            tools = _EXECUTOR_TOOLS + extra_specs
        cap = BashCapability(session.execute, deadline)
    extra_names = {t["name"] for t in tools} - {"bash", "submit"}
    # native-tool-calling message history (content = list of blocks)
    msgs = [{"role": "user", "content": f"Issue:\n{problem}"}]
    no_tool_streak = 0        # consecutive turns without a parsed tool call (spam guard)
    try:
        if not session.start():
            return float("nan"), transcript          # infra, not a verdict
        for turn in range(max_turns):
            # hook LLM calls share the turn budget: stop once agent turns plus hook LLM calls
            # reach the cap
            if (hooks_rt is not None and hooks_rt.llm_cap is not None
                    and turn + hooks_rt.llm_cap.calls >= max_turns):
                transcript.append({"role": "user",
                                   "content": "(turn budget consumed: agent turns + "
                                              "hook llm calls hit the cap)"})
                patch = session.get_patch()
                break
            if time.monotonic() > deadline:
                transcript.append({"role": "user", "content": "(episode wall-clock cap hit)"})
                patch = session.get_patch()
                break
            # Sliding-window context guard: over a long episode the tool results would exceed the
            # model's context. Observations older than the last keep_full turns are replaced by a
            # stub (observation masking, as in mini-SWE-agent). Only tool_result content strings
            # shrink; the blocks stay, because the API requires every tool_use to keep its paired
            # tool_result.
            if turn and turn % 8 == 0:
                _shrink_old_observations(msgs, keep_recent_msgs=keep_full * 2)
            body_msgs = hooks_rt.before_llm(msgs, turn) if hooks_rt is not None else msgs
            body = {"max_tokens": 4096,
                    "system": system_prompt, "tools": tools, "messages": body_msgs}
            # temperature 0 for the executor; models listed in _TEMP_DEPRECATED use the server's
            # default sampling instead
            if not any(t in model for t in _TEMP_DEPRECATED):
                body["temperature"] = 0.0
            resp = _invoke_bedrock_raw(model, region, body)
            content = resp.get("content", [])
            if hooks_rt is not None:
                # after_llm lets harness code convert tool-call markup the model emitted as
                # text (DSML/XML) into real tool_use blocks
                content = hooks_rt.after_llm(content, turn)
            msgs.append({"role": "assistant", "content": content})
            texts = [b.get("text", "") for b in content if b.get("type") == "text"]
            tool_uses = [b for b in content if b.get("type") == "tool_use"]
            rec = {"turn": turn, "text": "\n".join(t for t in texts if t),
                   "commands": [], "observations": []}
            # readable assistant line for the transcript
            asst_lines = [t for t in texts if t.strip()]

            if not tool_uses:
                # no tool call: a deliberate end_turn finishes with the repo diff, else nudge
                transcript.append({"role": "assistant",
                                   "content": "\n".join(asst_lines) or "(no action)"})
                # tool-call markup in text is a failed call, not a finish: nudge, whatever the
                # stop_reason
                fake_call = _looks_like_failed_toolcall(texts)
                if resp.get("stop_reason") == "end_turn" and turn > 0 and not fake_call:
                    patch = session.get_patch()
                    if steps_out is not None:
                        rec["kind"] = "end_no_tool"; steps_out.append(rec)
                    break
                no_tool_streak += 1
                if no_tool_streak >= spam_cap:
                    # the model keeps producing unparseable turns: stop (nudges could use up the
                    # whole episode) and grade the work done so far
                    transcript.append({"role": "user",
                                       "content": "(episode ended: repeated unparseable turns)"})
                    patch = session.get_patch()
                    if steps_out is not None:
                        rec["kind"] = "spam_abort"; steps_out.append(rec)
                    break
                nudge = nudge_bad_markup if fake_call else nudge_no_tool
                msgs.append({"role": "user", "content": nudge})
                transcript.append({"role": "user", "content": nudge})
                if steps_out is not None:
                    rec["kind"] = "nudge"; steps_out.append(rec)
                continue

            # execute every tool_use this turn (the model may batch independent commands); one
            # tool_result per tool_use_id (the API requires a 1:1 match)
            tool_results = []
            submitted = False
            real_action = False       # any submit / extra tool / non-empty bash this turn
            for tu in tool_uses:
                name, tid = tu.get("name"), tu.get("id")
                if name == "submit":
                    submitted = True
                    real_action = True
                    tool_results.append({"type": "tool_result", "tool_use_id": tid,
                                         "content": "Submitted."})
                    continue
                if name in extra_names and hooks_rt is not None:
                    # hook-defined tool: implemented by harness code, acting only through the
                    # capability (same container as bash). Its commands go into the transcript
                    # as ```bash fences so replay, perturbation and trace minimization see them.
                    real_action = True
                    n0 = len(cap.commands)
                    out = hooks_rt.run_tool(name, tu.get("input") or {}, cap, turn)
                    out_t = truncate_observation(out, obs_cap)
                    tool_results.append({"type": "tool_result", "tool_use_id": tid,
                                         "content": out_t})
                    for c in cap.commands[n0:]:
                        rec["commands"].append(c)
                        asst_lines.append(f"```bash\n{c}\n```")
                    rec["observations"].append(f"[tool {name}] " + out_t[:1000])
                    continue
                t_args = tu.get("input") or {}
                if hooks_rt is not None:
                    t_args, block, synth = hooks_rt.before_tool(name or "bash", t_args, turn)
                    if block:
                        # synthetic result (as in HarnessX's synthetic_result): the hook
                        # answers the call itself; the model sees the synthetic observation
                        # and the container never runs the command
                        real_action = True
                        obs_t = truncate_observation(synth, obs_cap) if synth \
                            else f"[harness] command blocked: {block}"
                        tool_results.append({"type": "tool_result", "tool_use_id": tid,
                                             "content": obs_t})
                        rec["observations"].append(obs_t[:1000])
                        asst_lines.append(
                            f"(synthetic: {t_args.get('command', '')[:80]})" if synth
                            else f"(blocked: {t_args.get('command', '')[:80]})")
                        continue
                cmd = t_args.get("command", "")
                if cmd.strip():
                    real_action = True
                    obs = session.execute(cmd)
                else:
                    # Empty-command guard (as in the TB loop): a bash tool_use with a blank
                    # command does nothing, and weak models repeat it when a large here-doc
                    # fails to serialize. Return an actionable error; the streak is counted
                    # below.
                    obs = ("ERROR: the bash tool_use arrived with an EMPTY `command` "
                           "argument (this happens when a large file body is attempted "
                           "in one call). Retry with a real command; write big files in "
                           "PIECES: printf '...' > file, then printf '...' >> file, or "
                           "several small heredocs.")
                if hooks_rt is not None:
                    obs = hooks_rt.after_tool(name or "bash", t_args, obs, turn)
                obs_t = truncate_observation(obs, obs_cap)
                tool_results.append({"type": "tool_result", "tool_use_id": tid, "content": obs_t})
                rec["commands"].append(cmd)
                rec["observations"].append(obs_t[:1000])
                # Render as a ```bash fence, not "$ cmd": b1_evidence.transcript_steps re-parses
                # the readable transcript with parse_action, which only recognizes fenced blocks
                # and <bash> tags. "$ cmd" lines would parse as no-ops and replay-based evidence
                # would lose these commands.
                asst_lines.append(f"```bash\n{cmd}\n```")
            # empty-command turns count toward the spam guard (a tool_use with a blank command
            # is not a real action); real actions reset it
            if real_action:
                no_tool_streak = 0
            else:
                no_tool_streak += 1
                if no_tool_streak >= spam_cap:
                    transcript.append({"role": "assistant", "content": "\n".join(asst_lines)})
                    transcript.append({"role": "user",
                                       "content": "(episode ended: repeated empty commands)"})
                    # grade the work done so far, as on every other early stop
                    patch = session.get_patch()
                    if steps_out is not None:
                        rec["kind"] = "spam_abort"; steps_out.append(rec)
                    break
            transcript.append({"role": "assistant", "content": "\n".join(asst_lines)})
            if rec["observations"]:
                transcript.append({"role": "user",
                                   "content": "\n---\n".join(rec["observations"])})
            msgs.append({"role": "user", "content": tool_results})
            if hooks_rt is not None and not submitted:
                note = hooks_rt.on_turn_end(turn)
                if note:
                    # add as an extra text block on the same user turn (tool_results is a list
                    # of blocks), not as a separate user message, which would break the
                    # tool_use/tool_result pairing the API enforces
                    tool_results.append({"type": "text",
                                         "text": f"[harness note] {note}"})
                    transcript.append({"role": "user", "content": f"[harness note] {note}"})
            if steps_out is not None:
                rec["kind"] = "submit" if submitted else "bash"; steps_out.append(rec)
            if submitted:
                patch = session.get_patch()
                break
        else:
            patch = session.get_patch()               # out of turns: grade the accumulated edits
    finally:
        session.close()
        if hooks_rt is not None:
            transcript.append(summary_transcript_entry(hooks_rt))

    node_ids = list(inst.get("FAIL_TO_PASS", [])) + list(inst.get("PASS_TO_PASS", []))
    if not (patch or "").strip() or not node_ids:
        if not (patch or "").strip():
            transcript.append({"role": "user",
                               "content": "VERIFIER: no patch submitted (empty git diff) "
                                          "— nothing to grade"})
        return 0.0, transcript
    script = _judge._build_eval_script(inst, node_ids)
    log_text = _judge._run_in_docker(image, script, patch, inst.get("test_patch", ""), 1800)
    if log_text is None or "T2E_TEST_PATCH_FAILED" in log_text:
        return float("nan"), transcript
    resolved = _judge._resolved_from_log(inst, log_text)
    if not resolved:
        # Record the grader-log tail and a one-line verdict marker (as the TB loop does), so the
        # optimizer can diagnose failures from the verifier output and not only from the
        # agent's own observations.
        tail = _grader_tail_entry(log_text)
        if tail:
            transcript.append(tail)
        transcript.append({"role": "user",
                           "content": _verifier_marker(inst, log_text)})
    return (1.0 if resolved else 0.0), transcript


# Grader-log evidence. The one-line marker names which tests failed; the tail shows how they
# failed (assert expected vs. got, tracebacks). AHE likewise gives its debugger the verifier's
# last 60 lines, and Self-Harness scans verifier stdout. The prefix deliberately does not start
# with VERIFIER or TEST, so the marker scans of the verification tool and attribution match
# only the one-line marker, which stays the transcript's last line.
_GRADER_TAIL_LINES = 60
_GRADER_TAIL_CHARS = 2400
_GRADER_PREFIX = "GRADER LOG (ground truth; the agent never saw this):"


def _grader_tail_entry(log_text: str) -> dict | None:
    lines = [ln for ln in (log_text or "").strip().splitlines() if ln.strip()]
    if not lines:
        return None
    tail = "\n".join(lines[-_GRADER_TAIL_LINES:])[-_GRADER_TAIL_CHARS:]
    return {"role": "user", "content": f"{_GRADER_PREFIX}\n{tail}"}


def _verifier_marker(inst: dict, log_text: str) -> str:
    """One transcript line naming the expected tests that did not pass (with their status
    when parseable): the SWE failure signature read by attribution and the optimizer's
    evidence. Never raises, so grading does not fail on an unexpected log format."""
    try:
        status = _judge._parse_pytest_log(inst, log_text)
        by_suffix = {}
        for node, st in (status or {}).items():
            by_suffix[node.split("::")[-1]] = st
        rows = []
        for node in list(inst.get("FAIL_TO_PASS", []))[:8]:
            st = status.get(node) or by_suffix.get(node.split("::")[-1])
            if st is None or str(st).upper() not in ("PASSED", "XFAIL"):
                rows.append(f"{node.split('::')[-1]}[{st or 'not-run'}]")
        for node in list(inst.get("PASS_TO_PASS", [])):
            st = status.get(node) or by_suffix.get(node.split("::")[-1])
            if st is not None and str(st).upper() in ("FAILED", "ERROR"):
                rows.append(f"{node.split('::')[-1]}[{st}] (regression)")
                if len(rows) >= 12:
                    break
        if rows:
            return "VERIFIER TEST FAILURES: " + " ".join(rows[:12])
        tail = (log_text or "").strip().splitlines()
        return "VERIFIER RESULT: not resolved — " + (tail[-1][:200] if tail else "no log")
    except Exception:
        return "VERIFIER RESULT: not resolved"


def _run_task_text(inst, image, session, system_prompt, transcript, url, model, *,
                   max_turns=250, steps_out=None) -> tuple[float, list]:
    """Text-only executor loop against a chat-completions endpoint (e.g. vLLM). parse_action
    accepts ```bash fences, <bash> tags and multiple commands per reply. A reply without an
    action gets a reminder; after three in a row, the current diff is graded."""
    patch, noops = "", 0
    try:
        if not session.start():
            return float("nan"), transcript
        for turn in range(max_turns):
            reply = _chat(url, model, transcript)
            transcript.append({"role": "assistant", "content": reply})
            action = parse_action(reply)
            kind = action.get("kind")
            rec = {"turn": turn, "text": reply[:1000], "kind": kind, "commands": [], "observations": []}
            if kind == "bash":
                noops = 0
                obs_all = []
                for cmd in action.get("commands", [action.get("command", "")]):
                    obs = session.execute(cmd)
                    obs_all.append(truncate_observation(obs, 4096))
                    rec["commands"].append(cmd)
                obs_joined = "\n---\n".join(obs_all)
                rec["observations"] = [o[:1000] for o in obs_all]
                transcript.append({"role": "user", "content": obs_joined})
            elif kind == "patch":
                patch = action.get("patch", ""); steps_out.append(rec) if steps_out is not None else None
                break
            elif kind == "submit":
                patch = session.get_patch(); steps_out.append(rec) if steps_out is not None else None
                break
            else:
                noops += 1
                if noops >= 3:
                    patch = session.get_patch(); break
                transcript.append({"role": "user", "content":
                                   "Reply with a ```bash``` block to run a command, or ```diff``` "
                                   "with your final patch, or `submit` when done."})
            if steps_out is not None:
                steps_out.append(rec)
        else:
            patch = session.get_patch()
    finally:
        session.close()
    node_ids = list(inst.get("FAIL_TO_PASS", [])) + list(inst.get("PASS_TO_PASS", []))
    if not (patch or "").strip() or not node_ids:
        return 0.0, transcript
    script = _judge._build_eval_script(inst, node_ids)
    log_text = _judge._run_in_docker(image, script, patch, inst.get("test_patch", ""), 1800)
    if log_text is None or "T2E_TEST_PATCH_FAILED" in log_text:
        return float("nan"), transcript
    resolved = _judge._resolved_from_log(inst, log_text)
    if not resolved:
        tail = _grader_tail_entry(log_text)
        if tail:
            transcript.append(tail)
        transcript.append({"role": "user", "content": _verifier_marker(inst, log_text)})
    return (1.0 if resolved else 0.0), transcript




_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_JSON_FENCE_RE = re.compile(r"```(?:json)?[ \t]*\n(\{.*?\})\s*```", re.DOTALL)


def _json_candidates(reply: str):
    """Yield candidate JSON strings from an optimizer reply, most likely first.

    <think> blocks are removed first. A greedy \\{.*\\} match would run from the first '{'
    (often inside leaked thinking text) to the last '}' and fail to parse. Order: fenced
    ```json blocks, then balanced-brace objects starting at each '{', later ones first (the
    answer follows the thinking)."""
    reply = _THINK_RE.sub("", reply or "")
    for m in _JSON_FENCE_RE.finditer(reply):
        yield m.group(1)
    starts = [i for i, ch in enumerate(reply) if ch == "{"]
    for s in reversed(starts):
        depth, in_str, esc = 0, False, False
        for i in range(s, len(reply)):
            c = reply[i]
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = not in_str
            elif not in_str:
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        yield reply[s:i + 1]
                        break

