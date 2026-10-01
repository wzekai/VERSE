"""CPU unit tests for the local vLLM transport.

vLLM serves the Messages API (/v1/messages), so the Messages-format request body is sent as
is, with `model` added. Requests are routed by served model name through T2E_VLLM_MODELS;
models not listed there go to the Bedrock Converse API.
"""
import io
import json
import urllib.error

import pytest

from verse.runtime import executor as ex


@pytest.fixture
def served(monkeypatch):
    monkeypatch.setenv("T2E_VLLM_MODELS",
                       "qwen38-flash-next=http://127.0.0.1:8100/, qwen35-9b=http://10.0.0.5:8101")


def test_vllm_base_mapping(served):
    assert ex._vllm_base("qwen38-flash-next") == "http://127.0.0.1:8100"   # trailing / stripped
    assert ex._vllm_base("qwen35-9b") == "http://10.0.0.5:8101"            # whitespace tolerated
    assert ex._vllm_base("some-hosted-model") == ""                        # unlisted -> Bedrock
    assert ex._vllm_base("qwen35") == ""                                   # exact match only


def test_vllm_base_unset(monkeypatch):
    monkeypatch.delenv("T2E_VLLM_MODELS", raising=False)
    assert ex._vllm_base("qwen35-9b") == ""


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_invoke_bedrock_raw_routes_to_vllm_verbatim(served, monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data)
        seen["timeout"] = timeout
        return _Resp(json.dumps({
            "content": [{"type": "thinking", "thinking": "hm", "signature": "x"},
                        {"type": "tool_use", "id": "chatcmpl-tool-1", "name": "bash",
                         "input": {"command": "ls"}}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 11, "output_tokens": 7}}).encode())

    monkeypatch.setattr(ex.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(ex, "_bedrock_client",
                        lambda region: pytest.fail("Bedrock client must not be built"))
    body = {"max_tokens": 4096,
            "system": "sys", "tools": [{"name": "bash", "input_schema": {"type": "object"}}],
            "messages": [{"role": "user", "content": "Issue:\nx"}], "temperature": 0.0}
    ex._tok_reset()
    out = ex._invoke_bedrock_raw("qwen35-9b", "us-west-2,us-east-1", body)
    assert seen["url"] == "http://10.0.0.5:8101/v1/messages"
    assert seen["body"]["model"] == "qwen35-9b"
    for k in ("max_tokens", "system", "tools", "messages", "temperature"):
        assert seen["body"][k] == body[k]                 # everything else passes through
    assert "model" not in body                            # caller's body not mutated
    # the tool loop reads blocks + stop_reason as returned
    assert out["stop_reason"] == "tool_use"
    assert [b["type"] for b in out["content"]] == ["thinking", "tool_use"]
    assert ex._tok_read() == {"calls": 1, "input": 11, "output": 7}
    # text extraction ignores thinking blocks
    monkeypatch.setattr(ex, "_invoke_bedrock_raw", lambda *a, **k: out)
    assert ex._invoke_bedrock("qwen35-9b", "us-west-2", body) == ""


def test_invoke_bedrock_raw_retries_vllm_transients(served, monkeypatch):
    calls = []

    def flaky_urlopen(req, timeout=None):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.URLError("connection refused")     # server restarting
        if len(calls) == 2:
            raise urllib.error.HTTPError(req.full_url, 503, "busy", {}, None)
        return _Resp(json.dumps({"content": [{"type": "text", "text": "ok"}],
                                 "stop_reason": "end_turn", "usage": {}}).encode())

    monkeypatch.setattr(ex.urllib.request, "urlopen", flaky_urlopen)
    monkeypatch.setattr(ex.time, "sleep", lambda s: None)
    out = ex._invoke_bedrock_raw("qwen38-flash-next", "us-west-2",
                                 {"max_tokens": 8, "messages": []}, max_retries=3)
    assert out["content"][0]["text"] == "ok" and len(calls) == 3


def test_invoke_bedrock_raw_vllm_midstream_disconnect_retries(served, monkeypatch):
    import http.client
    calls = []

    def dropping_urlopen(req, timeout=None):
        calls.append(1)
        if len(calls) == 1:
            raise http.client.RemoteDisconnected("tunnel re-dialed")   # not an HTTPError
        if len(calls) == 2:
            raise ConnectionResetError(104, "reset")
        return _Resp(json.dumps({"content": [], "stop_reason": "end_turn", "usage": {}}).encode())

    monkeypatch.setattr(ex.urllib.request, "urlopen", dropping_urlopen)
    monkeypatch.setattr(ex.time, "sleep", lambda s: None)
    out = ex._invoke_bedrock_raw("qwen35-9b", "us-west-2", {"max_tokens": 8, "messages": []})
    assert out["stop_reason"] == "end_turn" and len(calls) == 3


def test_invoke_bedrock_raw_vllm_400_is_fatal(served, monkeypatch):
    def bad_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 400, "bad request", {}, None)

    monkeypatch.setattr(ex.urllib.request, "urlopen", bad_urlopen)
    monkeypatch.setattr(ex.time, "sleep", lambda s: pytest.fail("no retry on 400"))
    with pytest.raises(urllib.error.HTTPError):
        ex._invoke_bedrock_raw("qwen35-9b", "us-west-2", {"max_tokens": 8, "messages": []})


def test_temperature_policy_for_local_models():
    # the optimizer runs in thinking mode, so no temperature is sent; the executor is greedy
    assert any(t in "qwen38-flash-next" for t in ex._TEMP_DEPRECATED)
    assert not any(t in "qwen35-9b" for t in ex._TEMP_DEPRECATED)


def test_vllm_base_replica_affinity(monkeypatch):
    """Several replicas of one served name: choice is a stable function of the
    conversation (system + first message), rotates with the retry attempt, and
    a single-URL entry or a missing body is unaffected."""
    monkeypatch.setenv("T2E_VLLM_MODELS",
                       "qwen38-flash-next=http://127.0.0.1:8100,"
                       "qwen38-27b=http://127.0.0.1:8101|http://127.0.0.1:8102|"
                       "http://127.0.0.1:8103|http://127.0.0.1:8104/")
    pool = {f"http://127.0.0.1:810{i}" for i in (1, 2, 3, 4)}
    body = {"system": "harness A", "messages": [{"role": "user", "content": "task 1"}]}
    first = ex._vllm_base("qwen38-27b", body)
    assert first in pool
    # later turns: same system + same first message, longer history -> same replica
    later = dict(body, messages=body["messages"] + [{"role": "assistant", "content": "x"},
                                                    {"role": "user", "content": "y"}])
    assert all(ex._vllm_base("qwen38-27b", later) == first for _ in range(5))
    # different episodes spread over the pool
    picks = {ex._vllm_base("qwen38-27b", {"system": "harness A",
                                          "messages": [{"role": "user", "content": f"task {i}"}]})
             for i in range(200)}
    assert picks == pool
    # retry attempt rotates the replica; attempt 4 wraps back
    rot = {ex._vllm_base("qwen38-27b", body, salt=a) for a in range(4)}
    assert rot == pool and ex._vllm_base("qwen38-27b", body, salt=4) == first
    # single replica and missing body
    assert ex._vllm_base("qwen38-flash-next", body) == "http://127.0.0.1:8100"
    assert ex._vllm_base("qwen38-27b") == "http://127.0.0.1:8101"
    assert ex._vllm_base("unlisted", body) == ""
