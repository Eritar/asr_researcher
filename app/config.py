"""All settings come from environment variables (or a .env file in the project root)."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


HOST = _env("HOST", "127.0.0.1")
PORT = int(_env("PORT", "8765"))

# --- Audio -----------------------------------------------------------------
# "system"        -> everything you hear (Linux: PipeWire/Pulse monitor, macOS: process tap)
# "app:<bundle>"  -> macOS: only one app, e.g. app:com.microsoft.teams2
# "pulse:<name>"  -> Linux: a specific source (see `pactl list short sources`)
# "file:<path>"   -> play an audio file in real time (testing / replay)
# Usually chosen in the web UI settings instead.
AUDIO_SOURCE = _env("AUDIO_SOURCE", "system")
SAMPLE_RATE = 16000

# --- ASR -------------------------------------------------------------------
ASR_ENGINE = _env("ASR_ENGINE", "parakeet")  # parakeet | qwen3 | cohere | canary
NUM_THREADS = int(_env("NUM_THREADS", str(min(8, os.cpu_count() or 4))))
ASR_PROVIDER = _env("ASR_PROVIDER", "cpu")  # cpu | cuda (needs sherpa-onnx CUDA build)
# Only these languages matter; used by engines that accept a language hint.
LANGUAGE = _env("LANGUAGE", "")  # "" = auto, or "en" / "de"
GLOSSARY = Path(_env("GLOSSARY", str(ROOT / "glossary.txt")))
# How often the growing utterance is re-decoded to show live partial text.
PARTIAL_INTERVAL_S = float(_env("PARTIAL_INTERVAL_S", "0.3"))

# --- VAD / segmentation ----------------------------------------------------
VAD_THRESHOLD = float(_env("VAD_THRESHOLD", "0.5"))
VAD_MIN_SILENCE_S = float(_env("VAD_MIN_SILENCE_S", "0.35"))
VAD_MIN_SPEECH_S = float(_env("VAD_MIN_SPEECH_S", "0.2"))
MAX_UTTERANCE_S = float(_env("MAX_UTTERANCE_S", "20"))

# --- OpenRouter ------------------------------------------------------------
OPENROUTER_API_KEY = _env("OPENROUTER_API_KEY", "")
if OPENROUTER_API_KEY.endswith("..."):  # placeholder copied from .env.example
    OPENROUTER_API_KEY = ""
OPENROUTER_MODEL = _env("OPENROUTER_MODEL", "google/gemini-3.8-flash")
# Provider routing: a specific endpoint tag (e.g. "google-ai-studio", "deepinfra/fp8"), "" = let
# OpenRouter choose, optionally sorted by latency | throughput | price.
OPENROUTER_PROVIDER = _env("OPENROUTER_PROVIDER", "")
OPENROUTER_SORT = _env("OPENROUTER_SORT", "latency")
# "Thinking" before answering adds seconds before the first visible word: off | low | on.
# Models that cannot switch it off automatically fall back to minimal effort.
OPENROUTER_REASONING = _env("OPENROUTER_REASONING", "off")
OPENROUTER_FALLBACKS = _env("OPENROUTER_FALLBACKS", "true").lower() not in ("0", "false", "no")
OPENROUTER_URL = _env("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
# Where conversations are autosaved (one .jsonl file per session).
SESSIONS_DIR = _env("SESSIONS_DIR", str(Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
                                        / "call-assist" / "sessions"))
CONTEXT_UTTERANCES = int(_env("CONTEXT_UTTERANCES", "20"))
ANSWER_LANGUAGE = _env("ANSWER_LANGUAGE", "auto")  # auto (= sentence language) | en | de
ABOUT_ME = _env("ABOUT_ME", "")  # your background, so answers are pitched at the right level


def glossary_terms() -> list[str]:
    if not GLOSSARY.exists():
        return []
    terms = []
    for line in GLOSSARY.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            terms.append(line)
    return terms
