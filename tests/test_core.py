import asyncio

import httpx
import numpy as np

from app import config, llm
from app.segmenter import Segmenter
from bench import normalize, wer


class FakeSeg:
    def __init__(self, start, samples):
        self.start, self.samples = start, samples


class FakeVAD:
    """Energy 'VAD': speech = frame RMS > 0.1; closes a segment after 2 silent frames."""

    def __init__(self):
        self.total, self.start, self.silent, self.buf, self.out = 0, None, 0, [], []

    def accept_waveform(self, f):
        loud = float(np.sqrt(np.mean(f**2))) > 0.1
        if loud:
            if self.start is None:
                self.start, self.buf = self.total, []
            self.silent = 0
        if self.start is not None:
            self.buf.append(f)
            if not loud:
                self.silent += 1
                if self.silent >= 2:
                    seg = np.concatenate(self.buf[:-2])
                    self.out.append(FakeSeg(self.start, seg))
                    self.start = None
        self.total += len(f)

    def empty(self):
        return not self.out

    @property
    def front(self):
        return self.out[0]

    def pop(self):
        self.out.pop(0)

    def is_speech_detected(self):
        return self.start is not None


def run_segmenter(pattern):
    seg = Segmenter(FakeVAD(), partial_interval_s=512 * 2 / 16000, lookback_s=0)
    events = []
    for loud in pattern:
        frame = np.full(512, 0.5 if loud else 0.0, dtype=np.float32)
        events += list(seg.feed(frame))
    return events


def test_segmenter_emits_partials_then_one_final_per_utterance():
    ev = run_segmenter([0] * 5 + [1] * 8 + [0] * 5 + [1] * 4 + [0] * 5)
    finals = [e for e in ev if e[0] == "final"]
    assert [e[1] for e in finals] == [1, 2]
    assert any(e[0] == "partial" and e[1] == 1 for e in ev)
    # final audio covers the whole loud run (plus bounded pre/post roll)
    assert np.sum(finals[0][2] > 0) == 8 * 512
    assert np.sum(finals[1][2] > 0) == 4 * 512
    # no partial for an utterance after its final
    last_final_idx = ev.index(finals[0])
    assert not any(e[0] == "partial" and e[1] == 1 for e in ev[last_final_idx:])


def test_segmenter_final_start_time():
    ev = run_segmenter([0] * 10 + [1] * 6 + [0] * 4)
    (final,) = [e for e in ev if e[0] == "final"]
    assert abs(final[3] - (10 * 512 / 16000 - 0.25)) < 1e-6


def test_wer_normalization():
    assert normalize("Straße, 3D-Modell!") == ["strasse", "3d", "modell"]
    assert wer("the cat sat", "the cat sat") == (0, 3)
    assert wer("the cat sat", "a cat sat down") == (2, 3)


def test_parse_sse_line():
    assert llm.parse_sse_line(": OPENROUTER PROCESSING") == ""
    assert llm.parse_sse_line('data: {"choices":[{"delta":{"content":"Hi"}}]}') == "Hi"
    assert llm.parse_sse_line("data: [DONE]") is None
    try:
        llm.parse_sse_line('data: {"error":{"message":"rate limited"}}')
    except llm.LLMError as e:
        assert "rate limited" in str(e)
    else:
        raise AssertionError


def test_stream_against_fake_openrouter(monkeypatch):
    sse = (
        ": OPENROUTER PROCESSING\n\n"
        'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    seen = {}

    def handler(req):
        seen["auth"] = req.headers["authorization"]
        seen["body"] = req.content
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(llm, "_client", httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    async def collect():
        msgs = llm.build_messages("What is CRISPR?", ["earlier"], "answer")
        return "".join([d async for d in llm.stream(msgs, "x/y")])

    assert asyncio.run(collect()) == "Hello"
    assert seen["auth"] == "Bearer sk-test"
    assert b'"stream": true' in seen["body"] or b'"stream":true' in seen["body"]


def test_stream_http_error(monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "bad")
    monkeypatch.setattr(llm, "_client", httpx.AsyncClient(transport=httpx.MockTransport(
        lambda r: httpx.Response(401, json={"error": {"message": "No auth credentials found"}}))))

    async def go():
        async for _ in llm.stream([{"role": "user", "content": "x"}]):
            pass

    try:
        asyncio.run(go())
    except llm.LLMError as e:
        assert "401" in str(e) and "No auth" in str(e)
    else:
        raise AssertionError


def test_settings_roundtrip(tmp_path, monkeypatch):
    from app import settings

    monkeypatch.setattr(settings, "PATH", tmp_path / "settings.json")
    monkeypatch.setattr(config, "GLOSSARY", tmp_path / "glossary.txt")
    for attr, _ in settings.FIELDS.values():
        monkeypatch.setattr(config, attr, getattr(config, attr))

    changed = settings.update({"openrouter_api_key": " sk-or-v1-abcdef123456 ", "openrouter_model": "x/y",
                               "audio_source": "monitor", "glossary": "Hamiltonian"})
    assert changed == {"openrouter_api_key", "openrouter_model", "glossary"}  # monitor == system already
    pub = settings.public()
    assert "openrouter_api_key" not in pub and pub["has_api_key"] and pub["api_key_hint"] == "…3456"
    assert pub["glossary"] == "Hamiltonian\n"
    assert (settings.PATH.stat().st_mode & 0o777) == 0o600

    config.OPENROUTER_API_KEY = ""
    settings.load()
    assert config.OPENROUTER_API_KEY == "sk-or-v1-abcdef123456"
    assert settings.update({"openrouter_model": "x/y"}) == set()

    assert settings.update({"openrouter_fallbacks": "false"}) == {"openrouter_fallbacks"}
    assert config.OPENROUTER_FALLBACKS is False
    for bad in ({"openrouter_provider": "Bad Provider"}, {"openrouter_sort": "fastest"}, {"asr_engine": "nope"}, {"language": "fr"}, {"vad_min_silence_s": 9},
                {"openrouter_model": "has space"}, {"unknown": 1}):
        try:
            settings.update(bad)
        except settings.SettingsError:
            pass
        else:
            raise AssertionError(bad)


def test_local_only_guard(monkeypatch):
    from app.main import _allowed

    monkeypatch.setattr(config, "PORT", 8765)
    assert _allowed("127.0.0.1:8765", None)
    assert _allowed("localhost:8765", "http://localhost:8765")
    assert not _allowed("127.0.0.1:8765", "https://evil.example")
    assert not _allowed("evil.example:8765", None)  # DNS rebinding


def test_provider_routing_in_request(monkeypatch):
    import json as _json

    bodies = []

    def handler(req):
        bodies.append(_json.loads(req.content))
        sse = 'data: {"provider":"DeepInfra","choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(llm, "_client", httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    async def run():
        meta = {}
        text = "".join([d async for d in llm.stream([{"role": "user", "content": "x"}], "a/b", meta)])
        return text, meta

    monkeypatch.setattr(config, "OPENROUTER_PROVIDER", "")
    monkeypatch.setattr(config, "OPENROUTER_SORT", "")
    asyncio.run(run())
    assert "provider" not in bodies[-1]

    monkeypatch.setattr(config, "OPENROUTER_SORT", "latency")
    asyncio.run(run())
    assert bodies[-1]["provider"] == {"sort": "latency"}

    monkeypatch.setattr(config, "OPENROUTER_SORT", "")
    monkeypatch.setattr(config, "OPENROUTER_PROVIDER", "deepinfra/fp8")
    monkeypatch.setattr(config, "OPENROUTER_FALLBACKS", False)
    text, meta = asyncio.run(run())
    assert bodies[-1]["provider"] == {"order": ["deepinfra/fp8"], "allow_fallbacks": False}
    assert text == "ok" and meta["provider"] == "DeepInfra"


def test_reasoning_off_falls_back_when_mandatory(monkeypatch):
    import json as _json

    bodies = []

    def handler(req):
        body = _json.loads(req.content)
        bodies.append(body)
        if body.get("reasoning") == {"enabled": False}:
            return httpx.Response(400, json={"error": {"message": "Reasoning is mandatory for this endpoint and cannot be disabled."}})
        sse = ('data: {"choices":[{"delta":{"reasoning":"hmm"}}]}\n\n'
               'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n')
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(config, "OPENROUTER_REASONING", "off")
    monkeypatch.setattr(llm, "_reasoning_mandatory", set())
    monkeypatch.setattr(llm, "_client", httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    thinking = []

    async def on_reasoning():
        thinking.append(True)

    async def run():
        return "".join([d async for d in llm.stream([{"role": "user", "content": "x"}], "g/flash",
                                                    on_reasoning=on_reasoning)])

    assert asyncio.run(run()) == "ok"
    assert [b.get("reasoning") for b in bodies] == [{"enabled": False}, {"effort": "minimal"}]
    assert thinking == [True]
    asyncio.run(run())  # remembered: no failing first attempt this time
    assert bodies[-1]["reasoning"] == {"effort": "minimal"} and len(bodies) == 3


def test_session_autosave_roundtrip(tmp_path, monkeypatch):
    from app import sessions

    monkeypatch.setattr(config, "SESSIONS_DIR", str(tmp_path))
    s = sessions.Session.new()
    assert not s.path.exists()  # nothing written until there is content
    s.add_utt({"utt_id": "a-1", "text": "Was ist ein Eigenwert?", "ts": 1_700_000_000, "dur": 1.2})
    s.add_utt({"utt_id": "a-2", "text": "The Hamiltonian is Hermitian.", "ts": 1_700_000_005, "dur": 2.0})
    s.save_thread("a-1", "answer", [{"q": None, "text": "first"}], "m/1", "P")
    s.save_thread("a-1", "answer", [{"q": None, "text": "second"}, {"q": "why?", "text": "because"}], "m/2", None)
    s.save_thread("a-2", "explain", [{"q": None, "text": "line 1\n\nline 2"}], "m/1", "P")
    s.rename("Quantum lecture")
    with s.path.open("a") as f:
        f.write('{"type": "utt", "utt_id": "a-3", "te')  # crash mid-write

    loaded = sessions.Session.load(s.path)
    assert loaded.meta["title"] == "Quantum lecture"
    assert list(loaded.utts) == ["a-1", "a-2"]
    assert loaded.threads[("a-1", "answer")]["turns"][-1] == {"q": "why?", "text": "because"}  # latest wins
    assert loaded.answered() == {"a-1": ["answer"], "a-2": ["explain"]}

    md = loaded.to_markdown()
    assert md.startswith("# Quantum lecture")
    assert "> **Q:** why?" in md and "> line 1\n>\n> line 2" in md and "first" not in md

    listed = sessions.list_sessions()
    assert [x["title"] for x in listed] == ["Quantum lecture"]
    assert listed[0]["sentences"] == 2 and listed[0]["answers"] == 2
    try:
        sessions.path_for("../etc/passwd")
    except ValueError:
        pass
    else:
        raise AssertionError
