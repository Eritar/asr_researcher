"""User settings edited from the web UI.

Stored in ~/.config/call-assist/settings.json (mode 600, it holds the API key) and applied on
top of the defaults / environment in config.py: settings.json > environment/.env > defaults.
"""

import json
import re
import logging
import os
from pathlib import Path

from . import config
from .audio import normalize_spec
from .engines import MODEL_DIRS

log = logging.getLogger(__name__)

PATH = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "call-assist" / "settings.json"

def _bool(v) -> bool:
    return v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")


# setting name -> (config attribute, type)
FIELDS = {
    "openrouter_api_key": ("OPENROUTER_API_KEY", str),
    "openrouter_model": ("OPENROUTER_MODEL", str),
    "openrouter_provider": ("OPENROUTER_PROVIDER", str),
    "openrouter_sort": ("OPENROUTER_SORT", str),
    "openrouter_fallbacks": ("OPENROUTER_FALLBACKS", _bool),
    "openrouter_reasoning": ("OPENROUTER_REASONING", str),
    "answer_language": ("ANSWER_LANGUAGE", str),
    "about_me": ("ABOUT_ME", str),
    "asr_engine": ("ASR_ENGINE", str),
    "language": ("LANGUAGE", str),
    "audio_source": ("AUDIO_SOURCE", str),
    "sessions_dir": ("SESSIONS_DIR", str),
    "vad_min_silence_s": ("VAD_MIN_SILENCE_S", float),
}
# Which part of the pipeline has to restart when a setting changes.
ENGINE_FIELDS = {"asr_engine", "language", "glossary"}
AUDIO_FIELDS = {"audio_source", "vad_min_silence_s"}

LANGUAGES = {"": "Auto-detect", "en": "English", "de": "German"}
REASONING_MODES = {"off": "Off (fastest, recommended for live use)",
                   "low": "Low (a few seconds slower, sometimes more accurate)",
                   "on": "Model default (slowest)"}
PROVIDER_SORTS = {"": "balanced", "latency": "lowest latency", "throughput": "highest throughput",
                  "price": "cheapest"}
ANSWER_LANGUAGES = {"auto": "Same as the sentence", "en": "English", "de": "German"}


class SettingsError(ValueError):
    pass


def load() -> None:
    if not PATH.exists():
        return
    try:
        data = json.loads(PATH.read_text())
    except (OSError, ValueError) as e:
        log.warning("ignoring unreadable %s: %s", PATH, e)
        return
    for name, value in data.items():
        if name in FIELDS:
            attr, typ = FIELDS[name]
            setattr(config, attr, typ(value))
    config.AUDIO_SOURCE = normalize_spec(config.AUDIO_SOURCE)


def _save() -> None:
    PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {name: getattr(config, attr) for name, (attr, _) in FIELDS.items()}
    tmp = PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.chmod(tmp, 0o600)
    tmp.replace(PATH)


def read_glossary() -> str:
    return config.GLOSSARY.read_text(encoding="utf-8") if config.GLOSSARY.exists() else ""


def public() -> dict:
    """Current settings for the UI. The API key itself is never sent back."""
    key = config.OPENROUTER_API_KEY
    out = {name: getattr(config, attr) for name, (attr, _) in FIELDS.items() if name != "openrouter_api_key"}
    out["has_api_key"] = bool(key)
    out["api_key_hint"] = f"…{key[-4:]}" if len(key) > 8 else ""
    out["glossary"] = read_glossary()
    return out


def update(changes: dict) -> set[str]:
    """Validate, apply and persist; returns the names that actually changed."""
    changed: set[str] = set()
    values = {}
    for name, value in changes.items():
        if name == "glossary":
            continue
        if name not in FIELDS:
            raise SettingsError(f"unknown setting {name!r}")
        attr, typ = FIELDS[name]
        try:
            value = typ(value.strip() if isinstance(value, str) else value)
        except (TypeError, ValueError):
            raise SettingsError(f"{name}: invalid value {value!r}") from None
        values[name] = _validate(name, value)

    for name, value in values.items():
        attr, _ = FIELDS[name]
        if getattr(config, attr) != value:
            setattr(config, attr, value)
            changed.add(name)

    if "glossary" in changes:
        text = str(changes["glossary"]).replace("\r\n", "\n")
        if text != read_glossary():
            config.GLOSSARY.write_text(text if text.endswith("\n") or not text else text + "\n", encoding="utf-8")
            changed.add("glossary")

    if changed - {"glossary"}:
        _save()
    return changed


def _validate(name: str, value):
    if name == "asr_engine" and value not in MODEL_DIRS:
        raise SettingsError(f"unknown ASR engine {value!r}")
    if name == "language" and value not in LANGUAGES:
        raise SettingsError(f"language must be one of {list(LANGUAGES)}")
    if name == "answer_language" and value not in ANSWER_LANGUAGES:
        raise SettingsError(f"answer language must be one of {list(ANSWER_LANGUAGES)}")
    if name == "vad_min_silence_s" and not 0.1 <= value <= 3.0:
        raise SettingsError("pause length must be between 0.1 and 3 seconds")
    if name == "openrouter_model" and (not value or " " in value):
        raise SettingsError("model must be an OpenRouter model id like google/gemini-3.8-flash")
    if name == "openrouter_reasoning" and value not in REASONING_MODES:
        raise SettingsError(f"thinking must be one of {list(REASONING_MODES)}")
    if name == "openrouter_sort" and value not in PROVIDER_SORTS:
        raise SettingsError(f"provider sort must be one of {list(PROVIDER_SORTS)}")
    if name == "openrouter_provider" and not re.fullmatch(r"[a-z0-9._/-]*", value):
        raise SettingsError("provider must be an OpenRouter endpoint tag like google-ai-studio or deepinfra/fp8")
    if name == "sessions_dir":
        if not value:
            raise SettingsError("folder for saved sessions cannot be empty")
        p = Path(value).expanduser()
        if not p.is_absolute():
            raise SettingsError("folder for saved sessions must be an absolute path")
        try:
            p.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise SettingsError(f"cannot use {p}: {e.strerror}") from None
        value = str(p)
    if name == "audio_source":
        value = normalize_spec(value)
    return value
