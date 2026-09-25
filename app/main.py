"""FastAPI server: serves the UI, pushes transcript events and streams LLM answers over one WebSocket.

Run:  uv run python -m app.main     then open http://127.0.0.1:8765
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response

from . import audio, config, engines, llm, sessions, settings
from .pipeline import Pipeline

log = logging.getLogger("app")
STATIC = Path(__file__).parent / "static"


class Hub:
    """Owns the current session (transcript + saved answers) and fans events out to browsers."""

    def __init__(self):
        self.clients: set[WebSocket] = set()
        self.session = sessions.Session.new()
        self.queue: asyncio.Queue = asyncio.Queue()
        self.loop: asyncio.AbstractEventLoop | None = None
        # (session id, utt_id, mode) -> future resolved when that first answer is saved, so a
        # second click while it is still generating waits for it instead of paying twice.
        self.inflight: dict[tuple[str, str, str], asyncio.Future] = {}

    @property
    def finals(self) -> dict[str, dict]:
        return self.session.utts

    def emit_threadsafe(self, ev: dict) -> None:
        if self.loop:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, ev)

    async def run(self) -> None:
        while True:
            ev = await self.queue.get()
            if ev["type"] == "final":
                if not ev["text"]:
                    continue
                try:
                    self.session.add_utt(ev)
                except OSError as e:
                    log.error("could not save transcript: %s", e)
                    await self.broadcast({"type": "status_note", "message": f"Autosave failed: {e}"})
            await self.broadcast(ev)

    def session_event(self) -> dict:
        return {"type": "session", "session": self.session.summary(),
                "transcript": list(self.session.utts.values()), "answered": self.session.answered()}

    async def switch(self, session: "sessions.Session") -> None:
        self.session = session
        await self.broadcast(self.session_event())

    async def broadcast(self, ev: dict) -> None:
        msg = json.dumps(ev)
        for ws in list(self.clients):
            try:
                await ws.send_text(msg)
            except Exception:  # noqa: BLE001
                self.clients.discard(ws)

    def context_for(self, utt_id: str) -> tuple[str, list[str]]:
        if utt_id not in self.finals:
            raise KeyError(utt_id)
        ids = list(self.finals)
        i = ids.index(utt_id)
        before = ids[max(0, i - config.CONTEXT_UTTERANCES) : i]
        return self.finals[utt_id]["text"], [self.finals[k]["text"] for k in before]


hub = Hub()
pipeline = Pipeline(hub.emit_threadsafe)


async def keep_openrouter_warm() -> None:
    # Idle connections are dropped after ~a minute; ping more often while someone has the page open.
    await llm.keep_warm()
    while True:
        await asyncio.sleep(25)
        if hub.clients:
            await llm.keep_warm()


@asynccontextmanager
async def lifespan(_: FastAPI):
    hub.loop = asyncio.get_running_loop()
    task = asyncio.create_task(hub.run())
    warm = asyncio.create_task(keep_openrouter_warm())
    pipeline.start()
    log.info("open http://%s:%d", config.HOST, config.PORT)
    yield
    task.cancel()
    warm.cancel()


app = FastAPI(lifespan=lifespan)


# -- local-only guard ------------------------------------------------------
# The server holds your API key and transcript. Browsers let any website talk to
# localhost, so only accept requests whose Origin is this page (blocks cross-site
# requests), and whose Host is the address we listen on (blocks DNS rebinding).
def _allowed(host: str | None, origin: str | None) -> bool:
    ok_hosts = {f"127.0.0.1:{config.PORT}", f"localhost:{config.PORT}", f"{config.HOST}:{config.PORT}"}
    if host not in ok_hosts:
        return False
    return origin is None or urlsplit(origin).netloc == host


@app.middleware("http")
async def local_only(request: Request, call_next):
    if not _allowed(request.headers.get("host"), request.headers.get("origin")):
        return JSONResponse({"detail": "forbidden"}, status_code=403)
    return await call_next(request)


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


# -- settings API ----------------------------------------------------------
@app.get("/api/settings")
async def get_settings():
    installed = engines.installed()
    return {
        "values": settings.public(),
        "options": {
            "engines": [{"id": k, "installed": v} for k, v in installed.items()],
            "languages": settings.LANGUAGES,
            "answer_languages": settings.ANSWER_LANGUAGES,
            "provider_sorts": settings.PROVIDER_SORTS,
            "reasoning_modes": settings.REASONING_MODES,
            "platform": "mac" if audio.IS_MAC else "linux",
        },
        "config_path": str(settings.PATH),
    }


@app.get("/api/audio-sources")
async def audio_sources():
    return await asyncio.to_thread(audio.list_sources)


@app.put("/api/settings")
async def put_settings(changes: dict):
    if changes.get("asr_engine") and not engines.installed().get(changes["asr_engine"]):
        raise HTTPException(400, f"model for {changes['asr_engine']} is not downloaded: "
                                 f"run scripts/download_models.sh {changes['asr_engine']}")
    try:
        changed = settings.update(changes)
    except settings.SettingsError as e:
        raise HTTPException(400, str(e)) from None
    restart = bool(changed & (settings.ENGINE_FIELDS | settings.AUDIO_FIELDS))
    if restart:
        await asyncio.to_thread(pipeline.restart)
    if changed & {"openrouter_model", "openrouter_provider", "openrouter_sort"}:
        await hub.broadcast({"type": "model", "model": config.OPENROUTER_MODEL, "routing": routing_label()})
    return {"changed": sorted(changed), "restarted": restart, "values": settings.public()}


@app.post("/api/restart")
async def restart():
    await asyncio.to_thread(pipeline.restart)
    return {"ok": True}


@app.post("/api/test-key")
async def test_key(body: dict):
    try:
        info = await llm.key_info(body.get("key") or None)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}
    return {"ok": True, "label": info.get("label"), "usage": info.get("usage"),
            "limit": info.get("limit"), "limit_remaining": info.get("limit_remaining")}


@app.get("/api/models")
async def models():
    try:
        return await llm.list_models()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"could not load models from OpenRouter: {e}") from None


@app.get("/api/endpoints")
async def endpoints(model: str):
    try:
        return await llm.list_endpoints(model)
    except llm.LLMError as e:
        raise HTTPException(404, str(e)) from None
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"could not load providers from OpenRouter: {e}") from None


def routing_label() -> str:
    if config.OPENROUTER_PROVIDER:
        return config.OPENROUTER_PROVIDER
    return f"auto, {settings.PROVIDER_SORTS[config.OPENROUTER_SORT]}" if config.OPENROUTER_SORT else ""


# -- sessions API ----------------------------------------------------------
@app.get("/api/sessions")
async def list_sessions():
    items = await asyncio.to_thread(sessions.list_sessions)
    current = hub.session.summary()
    if not any(s["id"] == current["id"] for s in items):
        items.insert(0, current)  # not written yet (no sentence so far)
    return {"current": hub.session.id, "dir": str(sessions.sessions_dir()), "sessions": items}


def _load(session_id: str) -> sessions.Session:
    if session_id == hub.session.id:
        return hub.session
    try:
        return sessions.Session.load(sessions.path_for(session_id))
    except (OSError, ValueError):
        raise HTTPException(404, f"session {session_id} not found") from None


@app.post("/api/sessions/new")
async def new_session():
    await hub.switch(sessions.Session.new())
    return hub.session.summary()


@app.post("/api/sessions/{session_id}/open")
async def open_session(session_id: str):
    if session_id != hub.session.id:
        await hub.switch(await asyncio.to_thread(_load, session_id))
    return hub.session.summary()


@app.patch("/api/sessions/{session_id}")
async def rename_session(session_id: str, body: dict):
    title = str(body.get("title", "")).strip()[:200]
    if not title:
        raise HTTPException(400, "title cannot be empty")
    s = _load(session_id)
    s.rename(title)
    if s is hub.session:
        await hub.broadcast({"type": "session_title", "title": title})
    return s.summary()


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    s = _load(session_id)
    if s.path.exists():
        s.path.unlink()
    if s is hub.session:
        await hub.switch(sessions.Session.new())
    return {"ok": True}


@app.get("/api/sessions/{session_id}/export")
async def export_session(session_id: str, fmt: str = "md"):
    s = _load(session_id)
    if fmt == "jsonl":
        body = s.path.read_text(encoding="utf-8") if s.path.exists() else ""
        media, ext = "application/x-ndjson", "jsonl"
    else:
        body, media, ext = s.to_markdown(), "text/markdown; charset=utf-8", "md"
    return Response(body, media_type=media,
                    headers={"Content-Disposition": f'attachment; filename="{s.id}.{ext}"'})


# -- live channel ----------------------------------------------------------
async def answer(ws: WebSocket, msg: dict) -> None:
    """Show the saved thread for (sentence, mode) if there is one, otherwise generate and save it.

    msg: {utt_id, mode, question?: follow-up to append, regenerate?: ignore the saved answer}
    """
    req_id = msg.get("req_id")
    gone = False

    async def send(**kw):
        # Keep generating (and saving) even if the page was closed or reloaded meanwhile.
        nonlocal gone
        if not gone:
            try:
                await ws.send_text(json.dumps({"req_id": req_id, **kw}))
            except Exception:  # noqa: BLE001
                gone = True

    done: asyncio.Future | None = None
    key = None
    try:
        utt_id, mode = str(msg["utt_id"]), msg.get("mode", "explain")
        if mode not in llm.MODES:
            raise ValueError(f"unknown mode {mode!r}")
        question = (msg.get("question") or "").strip()
        session = hub.session
        key = (session.id, utt_id, mode)
        fresh = not question and not msg.get("regenerate")
        if fresh and not session.threads.get((utt_id, mode)) and key in hub.inflight:
            await send(type="llm_start", model=config.OPENROUTER_MODEL, followup=False)
            await asyncio.shield(hub.inflight[key])  # being generated by an earlier click
        saved = session.threads.get((utt_id, mode))
        if saved and fresh:
            await send(type="llm_cached", **{k: saved[k] for k in ("turns", "model", "provider", "ts")})
            return
        if not question and key not in hub.inflight:
            done = hub.inflight[key] = asyncio.get_running_loop().create_future()
        # Follow-ups continue the saved thread; the server's copy is the source of truth.
        prior = saved["turns"] if (saved and question) else []
        history = []
        for turn in prior:
            if turn.get("q"):
                history.append({"role": "user", "content": turn["q"]})
            history.append({"role": "assistant", "content": turn["text"]})
        sentence, context = hub.context_for(utt_id)
        messages = llm.build_messages(sentence, context, mode, question, history)
        model = config.OPENROUTER_MODEL
        await send(type="llm_start", model=model, followup=bool(question))
        t0 = time.perf_counter()
        first = True
        meta: dict = {}
        text_parts: list[str] = []

        async def thinking():
            await send(type="llm_thinking", ms=round((time.perf_counter() - t0) * 1000))

        async for delta in llm.stream(messages, model, meta, on_reasoning=thinking):
            if first:
                await send(type="llm_ttft", ms=round((time.perf_counter() - t0) * 1000),
                           provider=meta.get("provider"))
                first = False
            await send(type="llm_delta", text=delta)
            text_parts.append(delta)
        turns = [*prior, {"q": question or None, "text": "".join(text_parts)}]
        saved_ok = True
        try:
            session.save_thread(utt_id, mode, turns, model, meta.get("provider"))
        except OSError as e:
            log.error("could not save answer: %s", e)
            saved_ok = False
        await send(type="llm_done", ms=round((time.perf_counter() - t0) * 1000), saved=saved_ok)
        if session is hub.session:
            await hub.broadcast({"type": "answered", "utt_id": utt_id, "mode": mode})
    except asyncio.CancelledError:
        await send(type="llm_done", cancelled=True)
        raise
    except Exception as e:  # noqa: BLE001
        log.warning("LLM request failed: %s", e)
        await send(type="llm_error", message=str(e) or type(e).__name__)
    finally:
        if done is not None:
            hub.inflight.pop(key, None)
            done.set_result(None)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    if not _allowed(ws.headers.get("host"), ws.headers.get("origin")):
        await ws.close(code=1008)
        return
    await ws.accept()
    await ws.send_text(json.dumps({
        **hub.session_event(),
        "type": "hello",
        "state": {**pipeline.state, "paused": pipeline.paused.is_set()},
        "model": config.OPENROUTER_MODEL,
        "routing": routing_label(),
        "has_api_key": bool(config.OPENROUTER_API_KEY),
    }))
    hub.clients.add(ws)
    # Answers keep running when you click elsewhere (they get saved); only Stop cancels one.
    tasks: dict = {}
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            kind = msg.get("type")
            if kind == "ask":
                req_id = msg.get("req_id")
                tasks[req_id] = task = asyncio.create_task(answer(ws, msg))
                task.add_done_callback(lambda _, r=req_id: tasks.pop(r, None))
            elif kind == "cancel" and msg.get("req_id") in tasks:
                tasks[msg["req_id"]].cancel()
            elif kind == "pause":
                (pipeline.paused.set if msg.get("paused") else pipeline.paused.clear)()
                pipeline.set_state()
    except WebSocketDisconnect:
        pass
    finally:
        hub.clients.discard(ws)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings.load()
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="warning")


if __name__ == "__main__":
    main()
