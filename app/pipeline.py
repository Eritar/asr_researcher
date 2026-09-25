"""Audio -> VAD -> ASR, on two threads, publishing transcript events via a callback.

Capture thread: reads frames, runs the VAD/segmenter, queues decode jobs, reports input level.
ASR thread:     owns the engine. Finals are always decoded (in order); for partials only
                the newest audio of the current utterance is decoded, stale ones are dropped,
                so a slow CPU just shows fewer partial updates instead of falling behind.

Both threads belong to one "run"; restart() (after a settings change) stops the run and
starts a new one. A failing audio source or engine never takes the server down: the error is
reported in `state` and shown in the UI.
"""

import logging
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable

import numpy as np

from . import audio, config, settings
from .engines import Engine
from .segmenter import Segmenter, make_vad

log = logging.getLogger(__name__)
LEVEL_INTERVAL_S = 0.15


class Pipeline:
    def __init__(self, emit: Callable[[dict], None]):
        self.emit = emit
        self.paused = threading.Event()
        self._lock = threading.Lock()
        self._engine_lock = threading.Lock()
        self._engine: Engine | None = None
        self._engine_key = None
        self._run: _Run | None = None
        self.state = {"engine": "", "engine_status": "idle", "audio": "", "audio_status": "idle",
                      "message": ""}

    def set_state(self, **kw) -> None:
        self.state.update(kw)
        self.emit({"type": "state", **self.state, "paused": self.paused.is_set()})

    def start(self) -> None:
        with self._lock:
            self._run = _Run(self)
            self._run.start()

    def restart(self) -> None:
        with self._lock:
            if self._run:
                self._run.stop()
            self._run = _Run(self)
            self._run.start()

    def get_engine(self) -> Engine:
        """Reuse the loaded engine unless engine / language / glossary changed."""
        key = (config.ASR_ENGINE, config.LANGUAGE,
               settings.read_glossary() if config.ASR_ENGINE == "qwen3" else "")
        with self._engine_lock:
            if self._engine is None or self._engine_key != key:
                self._engine = None  # free the old model before loading the next one
                self._engine = Engine(config.ASR_ENGINE)
                self._engine.warmup()
                self._engine_key = key
            return self._engine


class _Run:
    def __init__(self, p: Pipeline):
        self.p = p
        self.stopped = threading.Event()
        self.cv = threading.Condition()
        self.finals: deque = deque()
        self.partial = None  # (utt_id, audio, start_s)
        self.finalized: set[int] = set()
        self.source: audio.Source | None = None
        self.t0 = time.time()  # wall clock of the first captured sample
        self.token = secrets.token_hex(4)  # makes utterance ids unique across runs and sessions
        self.threads = [threading.Thread(target=self._asr_loop, name="asr", daemon=True),
                        threading.Thread(target=self._capture_loop, name="capture", daemon=True)]

    def start(self) -> None:
        for t in self.threads:
            t.start()

    def stop(self) -> None:
        self.stopped.set()
        if self.source:
            self.source.close()
        with self.cv:
            self.cv.notify_all()
        for t in self.threads:
            t.join(timeout=5)

    # -- capture -----------------------------------------------------------
    def _status(self, msg: str) -> None:
        if not self.stopped.is_set():
            self.p.set_state(audio_status="waiting", message=msg)

    def _capture_loop(self) -> None:
        spec = config.AUDIO_SOURCE
        self.p.set_state(audio=spec, audio_status="starting", message="")
        try:
            self.source = audio.open_source(spec, on_status=self._status)
            if self.stopped.is_set():
                self.source.close()
            seg = Segmenter(make_vad())
            peak, last_level, capturing = 0.0, time.monotonic(), False
            for frame in self.source.frames():
                if self.stopped.is_set():
                    break
                if not capturing:
                    capturing = True
                    self.t0 = time.time() - len(frame) / config.SAMPLE_RATE
                    self.p.set_state(audio_status="capturing", message="")
                peak = max(peak, float(np.sqrt(np.mean(frame * frame))))
                now = time.monotonic()
                if now - last_level >= LEVEL_INTERVAL_S:
                    db = 20 * np.log10(peak) if peak > 1e-6 else -120.0
                    self.p.emit({"type": "level", "db": round(float(db), 1)})
                    peak, last_level = 0.0, now
                if self.p.paused.is_set():
                    continue
                for kind, uid, pcm, start_s in seg.feed(frame):
                    with self.cv:
                        if kind == "final":
                            self.finals.append((uid, pcm, start_s, time.perf_counter()))
                            if self.partial and self.partial[0] == uid:
                                self.partial = None
                        else:
                            self.partial = (uid, pcm, start_s)
                        self.cv.notify()
            if not self.stopped.is_set():
                self.p.set_state(audio_status="ended", message="audio source ended")
        except Exception as e:  # noqa: BLE001
            if not self.stopped.is_set():
                log.error("audio capture failed: %s", e)
                self.p.set_state(audio_status="error", message=str(e))
        finally:
            if self.source:
                self.source.close()

    # -- ASR ---------------------------------------------------------------
    def _asr_loop(self) -> None:
        self.p.set_state(engine=config.ASR_ENGINE, engine_status="loading")
        try:
            engine = self.p.get_engine()
        except Exception as e:  # noqa: BLE001
            log.error("ASR engine failed: %s", e)
            self.p.set_state(engine_status="error", message=f"ASR engine: {e}")
            return
        if self.stopped.is_set():
            return
        self.p.set_state(engine_status="ready")

        while not self.stopped.is_set():
            with self.cv:
                while not self.finals and self.partial is None and not self.stopped.is_set():
                    self.cv.wait(0.5)
                if self.stopped.is_set():
                    return
                if self.finals:
                    job, final = self.finals.popleft(), True
                else:
                    job, final = self.partial, False
                    self.partial = None

            if final:
                uid, pcm, start_s, queued_at = job
            else:
                uid, pcm, start_s = job
                if uid in self.finalized:
                    continue
            t = time.perf_counter()
            try:
                text = engine.transcribe(pcm)
            except Exception:  # noqa: BLE001
                log.exception("decode failed")
                continue
            now = time.perf_counter()
            ev = {
                "type": "final" if final else "partial",
                "utt_id": f"{self.token}-{uid}",
                "text": text,
                "ts": round(self.t0 + start_s, 2),  # unix time the utterance started
                "dur": round(len(pcm) / config.SAMPLE_RATE, 2),
                "asr_ms": round((now - t) * 1000),
            }
            if final:
                self.finalized.add(uid)
                # speech end -> text: VAD silence hangover + queueing + decode
                ev["latency_ms"] = round((now - queued_at) * 1000 + config.VAD_MIN_SILENCE_S * 1000)
            if not self.stopped.is_set():
                self.p.emit(ev)
