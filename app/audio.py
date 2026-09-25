"""Audio sources yielding 16 kHz mono float32 frames of FRAME samples.

Source specs (setting `audio_source`):
  system          everything your computer plays
                    Linux: PipeWire/PulseAudio monitor of the default output, via `parec`
                    macOS: Core Audio process tap, via the bundled `systap` helper (14.2+)
  app:<bundle>    macOS only: just one app (e.g. app:com.microsoft.teams2), helpers included
  pulse:<name>    Linux only: a specific PulseAudio/PipeWire source (see list_sources())
  file:<path>     replay an audio file in real time (testing)
"""

import json
import logging
import shutil
import subprocess
import sys
import threading
import time
import wave
from collections import deque
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

from . import config

log = logging.getLogger(__name__)
FRAME = 512  # 32 ms at 16 kHz; also Silero VAD's window size
IS_MAC = sys.platform == "darwin"

SYSTAP_SRC = Path(__file__).parent / "native" / "systap.swift"
SYSTAP_BIN = config.ROOT / "build" / "systap"


class AudioError(RuntimeError):
    pass


def normalize_spec(spec: str) -> str:
    if spec in ("", "monitor", "auto"):
        return "system"
    if spec.startswith("monitor:"):
        return "pulse:" + spec[8:]
    return spec


def open_source(spec: str, on_status: Callable[[str], None] = lambda m: None) -> "Source":
    spec = normalize_spec(spec)
    if spec.startswith("file:"):
        return FileSource(Path(spec[5:]).expanduser())
    if IS_MAC:
        if spec == "system":
            return SystapSource([], on_status)
        if spec.startswith("app:"):
            return SystapSource(["--bundle", spec[4:]], on_status)
        raise AudioError(f"audio source {spec!r} is not available on macOS")
    if spec == "system":
        return ParecSource("@DEFAULT_MONITOR@")
    if spec.startswith("pulse:"):
        return ParecSource(spec[6:])
    raise AudioError(f"unknown audio source {spec!r}")


class Source:
    def __init__(self):
        self.closed = threading.Event()

    def frames(self) -> Iterator[np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:
        self.closed.set()


class _Proc:
    """A child process whose stdout is raw float32 audio; stderr is kept for error messages."""

    def __init__(self, cmd: list[str]):
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self.err: deque[str] = deque(maxlen=20)
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self) -> None:
        for line in iter(self.p.stderr.readline, b""):
            text = line.decode(errors="replace").rstrip()
            self.err.append(text)
            log.log(logging.DEBUG if "waiting for" in text else logging.INFO, "%s", text)

    def frames(self) -> Iterator[np.ndarray]:
        n = FRAME * 4
        while True:
            chunks, got = [], 0
            while got < n:
                b = self.p.stdout.read(n - got)
                if not b:
                    return
                chunks.append(b)
                got += len(b)
            yield np.frombuffer(b"".join(chunks), dtype=np.float32)

    def wait(self) -> int:
        code = self.p.wait()
        time.sleep(0.05)  # let the stderr reader catch up
        return code

    def kill(self) -> None:
        if self.p.poll() is None:
            self.p.terminate()


class ParecSource(Source):
    def __init__(self, device: str):
        super().__init__()
        if not shutil.which("parec"):
            raise AudioError("`parec` not found. Install it with: sudo apt install pulseaudio-utils")
        self.proc = _Proc(["parec", f"--device={device}", "--format=float32le",
                           f"--rate={config.SAMPLE_RATE}", "--channels=1", "--latency-msec=20", "--raw"])

    def frames(self):
        yield from self.proc.frames()
        if self.closed.is_set():
            return
        self.proc.wait()
        detail = " ".join(line for line in self.proc.err if not line.startswith("W: ")) or "no output"
        hint = ""
        if "Connection refused" in detail or "pa_context_connect" in detail:
            hint = (" Is PipeWire/PulseAudio running? Check `systemctl --user status pipewire-pulse`.")
        raise AudioError(f"parec stopped: {detail}.{hint}")

    def close(self):
        super().close()
        self.proc.kill()


def ensure_systap() -> Path:
    """Compile the macOS capture helper on first use (needs Xcode command line tools)."""
    if SYSTAP_BIN.exists() and SYSTAP_BIN.stat().st_mtime >= SYSTAP_SRC.stat().st_mtime:
        return SYSTAP_BIN
    if not shutil.which("swiftc"):
        raise AudioError("building the macOS audio helper needs the Swift compiler: run `xcode-select --install`")
    SYSTAP_BIN.parent.mkdir(parents=True, exist_ok=True)
    log.info("compiling %s", SYSTAP_SRC.name)
    r = subprocess.run(["swiftc", "-O", "-swift-version", "5", "-o", str(SYSTAP_BIN), str(SYSTAP_SRC)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise AudioError(f"compiling systap failed: {r.stderr[-800:]}")
    return SYSTAP_BIN


class SystapSource(Source):
    """Runs systap and restarts it when the output device or the app's processes change."""

    def __init__(self, args: list[str], on_status: Callable[[str], None]):
        super().__init__()
        self.cmd = [str(ensure_systap()), *args]
        self.on_status = on_status
        self.proc: _Proc | None = None

    def frames(self):
        failures = 0
        while not self.closed.is_set():
            self.proc = _Proc(self.cmd)
            got_audio = False
            for frame in self.proc.frames():
                got_audio = True
                yield frame
            if self.closed.is_set():
                return
            code = self.proc.wait()
            last = self.proc.err[-1] if self.proc.err else ""
            if code == 2:  # requested app isn't using audio yet
                self.on_status(last.removeprefix("systap: "))
                self.closed.wait(1.0)
            elif code == 0:  # device / process set changed
                self.on_status("audio device changed, reconnecting")
            else:
                failures = 0 if got_audio else failures + 1
                if failures >= 3:
                    raise AudioError(f"macOS audio capture failed: {' '.join(self.proc.err) or f'exit {code}'}")
                self.closed.wait(0.5)

    def close(self):
        super().close()
        if self.proc:
            self.proc.kill()


class FileSource(Source):
    def __init__(self, path: Path):
        super().__init__()
        if not path.exists():
            raise AudioError(f"file not found: {path}")
        self.path = path

    def frames(self):
        audio = load_audio(self.path)
        # Trailing silence so the last utterance gets closed by the VAD.
        audio = np.concatenate([audio, np.zeros(config.SAMPLE_RATE, dtype=np.float32)])
        t0 = time.monotonic()
        for i in range(0, len(audio) - FRAME + 1, FRAME):
            if self.closed.is_set():
                return
            delay = t0 + (i + FRAME) / config.SAMPLE_RATE - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            yield audio[i : i + FRAME]


def load_audio(path: Path) -> np.ndarray:
    """Decode any audio file to 16 kHz mono float32 (stdlib for 16 kHz WAV, ffmpeg otherwise)."""
    try:
        with wave.open(str(path), "rb") as w:
            if w.getframerate() == config.SAMPLE_RATE and w.getsampwidth() == 2:
                data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
                data = data.reshape(-1, w.getnchannels()).mean(axis=1)
                return (data / 32768.0).astype(np.float32)
    except (wave.Error, EOFError):
        pass
    if not shutil.which("ffmpeg"):
        raise AudioError(f"{path}: need ffmpeg to decode non-16kHz/non-WAV audio")
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
         "-f", "f32le", "-ac", "1", "-ar", str(config.SAMPLE_RATE), "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def list_sources() -> list[dict]:
    """Choices for the settings UI: [{id, label, active?}]."""
    out = [{"id": "system", "label": "All system audio (everything you hear)"}]
    try:
        out += _mac_apps() if IS_MAC else _pulse_sources()
    except Exception as e:  # noqa: BLE001
        log.warning("listing audio sources failed: %s", e)
    return out


def _mac_apps() -> list[dict]:
    rows = json.loads(subprocess.run([str(ensure_systap()), "--list"], capture_output=True,
                                     text=True, timeout=10, check=True).stdout)
    rows = [r for r in rows if not r["bundle"].startswith("com.apple.")]
    bundles = {r["bundle"] for r in rows}
    apps: dict[str, dict] = {}
    for r in rows:
        # Fold helper processes (com.foo.App.helper...) into their main app.
        main = min((b for b in bundles if r["bundle"].startswith(b + ".")), key=len, default=r["bundle"])
        app = apps.setdefault(main, {"id": f"app:{main}", "label": main, "active": False})
        if r["bundle"] == main:
            app["label"] = r["name"].strip("‎ ") or main
        app["active"] |= r["playing"]
    for a in apps.values():
        a["label"] = f"Only {a['label']}" + ("  (playing now)" if a["active"] else "")
    return sorted(apps.values(), key=lambda a: (not a["active"], a["label"].lower()))


def _pulse_sources() -> list[dict]:
    if not shutil.which("pactl"):
        return []
    r = subprocess.run(["pactl", "-f", "json", "list", "sources"], capture_output=True, text=True, timeout=5)
    if r.returncode == 0 and r.stdout.strip().startswith("["):
        items = [(s["name"], s.get("description") or s["name"]) for s in json.loads(r.stdout)]
    else:
        r = subprocess.run(["pactl", "list", "short", "sources"], capture_output=True, text=True, timeout=5)
        items = [(line.split("\t")[1], line.split("\t")[1]) for line in r.stdout.splitlines() if "\t" in line]
    # Monitors (what an output plays) first; microphones last.
    items.sort(key=lambda x: not x[0].endswith(".monitor"))
    return [{"id": f"pulse:{name}", "label": ("Output: " if name.endswith(".monitor") else "Input: ") + desc}
            for name, desc in items]
