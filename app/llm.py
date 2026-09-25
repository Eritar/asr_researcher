"""Streaming chat completions from OpenRouter (OpenAI-compatible SSE)."""

import json
import re
import time
from collections.abc import AsyncIterator, Iterable

import httpx

from . import config

SYSTEM = (
    "You assist someone listening to a live call or talk, often full of scientific and "
    "technical language, in English or German. You get a transcript excerpt produced by "
    "automatic speech recognition: it may contain misrecognised technical terms, names or "
    "formulas - silently infer what the speaker most likely said. Be concise and concrete; "
    "the user is reading while still listening. Use short paragraphs or bullet points, "
    "no preamble. Unless told otherwise, answer in the language of the selected sentence."
)

MODES = {
    "explain": "Explain the selected sentence: what it means, and define any technical "
               "terms, abbreviations or concepts it relies on.",
    "expand": "Expand on the selected sentence: give the background, the underlying "
              "reasoning, and relevant related facts the speaker is likely alluding to.",
    "answer": "The selected sentence is (or contains) a question. Answer it directly and "
              "accurately, then briefly justify the answer.",
}


ANSWER_IN = {"en": "Always answer in English.", "de": "Antworte immer auf Deutsch."}


def system_prompt() -> str:
    parts = [SYSTEM]
    if config.ANSWER_LANGUAGE in ANSWER_IN:
        parts.append(ANSWER_IN[config.ANSWER_LANGUAGE])
    if config.ABOUT_ME.strip():
        parts.append("About the user (pitch the depth accordingly): " + config.ABOUT_ME.strip())
    return "\n\n".join(parts)


def build_messages(sentence: str, context: Iterable[str], mode: str,
                   question: str = "", history: list[dict] | None = None) -> list[dict]:
    ctx = "\n".join(f"- {c}" for c in context) or "(none)"
    task = MODES.get(mode, MODES["explain"])
    user = (
        f"Recent transcript (oldest first):\n{ctx}\n\n"
        f"Selected sentence:\n\"{sentence}\"\n\n{task}"
    )
    msgs = [{"role": "system", "content": system_prompt()}, {"role": "user", "content": user}]
    if history:
        msgs += history
    if question:
        msgs.append({"role": "user", "content": question})
    return msgs


class LLMError(Exception):
    pass


def parse_sse_line(line: str, meta: dict | None = None) -> str | None:
    """Return the content delta in one SSE line, '' for no content, None at [DONE].

    If `meta` is given, the provider that served the request is recorded in meta["provider"].
    """
    if not line.startswith("data:"):
        return ""  # blank lines, ": OPENROUTER PROCESSING" keep-alive comments
    data = line[5:].strip()
    if data == "[DONE]":
        return None
    try:
        chunk = json.loads(data)
    except json.JSONDecodeError:
        return ""
    if "error" in chunk:
        err = chunk["error"]
        raise LLMError(err.get("message", str(err)) if isinstance(err, dict) else str(err))
    if meta is not None and chunk.get("provider") and "provider" not in meta:
        meta["provider"] = chunk["provider"]
    choices = chunk.get("choices") or [{}]
    delta = choices[0].get("delta") or {}
    if meta is not None and (delta.get("reasoning") or delta.get("reasoning_details")):
        meta["reasoning"] = True
    return delta.get("content") or ""


_client: httpx.AsyncClient | None = None


def client() -> httpx.AsyncClient:
    # One shared client keeps the TLS connection to OpenRouter warm -> lower time-to-first-token.
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0), http2=False)
    return _client


def provider_prefs() -> dict | None:
    """OpenRouter provider routing from settings (https://openrouter.ai/docs/features/provider-routing)."""
    prefs: dict = {}
    if config.OPENROUTER_PROVIDER:
        prefs["order"] = [config.OPENROUTER_PROVIDER]
        prefs["allow_fallbacks"] = config.OPENROUTER_FALLBACKS
    if config.OPENROUTER_SORT:
        prefs["sort"] = config.OPENROUTER_SORT
    return prefs or None


# Models that refused {"enabled": false} ("Reasoning is mandatory"): ask for minimal effort
# instead, without paying for the failed attempt again.
_reasoning_mandatory: set[str] = set()


def reasoning_param(model: str) -> dict | None:
    mode = config.OPENROUTER_REASONING
    if mode == "off":
        return {"effort": "minimal"} if model in _reasoning_mandatory else {"enabled": False}
    if mode == "low":
        return {"effort": "low"}
    return None


async def stream(messages: list[dict], model: str | None = None, meta: dict | None = None,
                 on_reasoning=None) -> AsyncIterator[str]:
    """Yield content deltas. `on_reasoning()` is awaited once when the model starts thinking."""
    if not config.OPENROUTER_API_KEY:
        raise LLMError("no OpenRouter API key: add it in Settings")
    model = model or config.OPENROUTER_MODEL
    meta = {} if meta is None else meta
    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Call Assist",
    }
    for attempt in range(2):
        body = {"model": model, "messages": messages, "stream": True}
        if prefs := provider_prefs():
            body["provider"] = prefs
        if (reasoning := reasoning_param(model)) is not None:
            body["reasoning"] = reasoning
        async with client().stream("POST", config.OPENROUTER_URL, json=body, headers=headers) as r:
            if r.status_code != 200:
                raw = (await r.aread()).decode(errors="replace")
                try:
                    msg = json.loads(raw)["error"]["message"]
                except (ValueError, KeyError, TypeError):
                    msg = raw[:300]
                if (attempt == 0 and r.status_code == 400 and "mandatory" in msg.lower()
                        and body.get("reasoning") == {"enabled": False}):
                    _reasoning_mandatory.add(model)
                    continue
                raise LLMError(f"OpenRouter {r.status_code}: {msg}")
            notified = False
            async for line in r.aiter_lines():
                delta = parse_sse_line(line, meta)
                if delta is None:
                    return
                if not notified and meta.get("reasoning") and on_reasoning:
                    notified = True
                    await on_reasoning()
                if delta:
                    yield delta
            return


async def keep_warm() -> None:
    """Keep a TLS connection to OpenRouter open so a click doesn't pay for the handshake."""
    if not config.OPENROUTER_API_KEY:
        return
    base = config.OPENROUTER_URL.rsplit("/chat/completions", 1)[0]
    try:
        await client().get(f"{base}/key", headers={"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"})
    except httpx.HTTPError:
        pass


async def key_info(key: str | None = None) -> dict:
    """Validate an API key: OpenRouter's /key endpoint returns its label, usage and limit."""
    key = key or config.OPENROUTER_API_KEY
    if not key:
        raise LLMError("no API key set")
    base = config.OPENROUTER_URL.rsplit("/chat/completions", 1)[0]
    r = await client().get(f"{base}/key", headers={"Authorization": f"Bearer {key}"})
    if r.status_code != 200:
        try:
            msg = r.json()["error"]["message"]
        except (ValueError, KeyError, TypeError):
            msg = r.text[:200]
        raise LLMError(f"OpenRouter {r.status_code}: {msg}")
    return r.json().get("data", {})


_models_cache: tuple[float, list[dict]] | None = None


async def list_models() -> list[dict]:
    """Text-output models from OpenRouter (public endpoint), cached for an hour."""
    global _models_cache
    if _models_cache and time.time() - _models_cache[0] < 3600:
        return _models_cache[1]
    base = config.OPENROUTER_URL.rsplit("/chat/completions", 1)[0]
    r = await client().get(f"{base}/models")
    r.raise_for_status()
    models = []
    for m in r.json()["data"]:
        arch = m.get("architecture") or {}
        if "text" not in (arch.get("output_modalities") or ["text"]):
            continue
        p = m.get("pricing") or {}
        models.append({
            "id": m["id"],
            "name": m.get("name", m["id"]),
            "context": m.get("context_length"),
            "prompt": float(p.get("prompt") or 0) * 1e6,  # USD per million tokens
            "completion": float(p.get("completion") or 0) * 1e6,
            "created": m.get("created", 0),
        })
    models.sort(key=lambda m: -m["created"])
    _models_cache = (time.time(), models)
    return models


_endpoints_cache: dict[str, tuple[float, list[dict]]] = {}


async def list_endpoints(model: str) -> list[dict]:
    """Providers serving `model`, from OpenRouter's public per-model endpoints list (cached 10 min)."""
    if not re.fullmatch(r"[\w.:~-]+/[\w.:~-]+", model):
        raise LLMError(f"not a model id: {model!r}")
    hit = _endpoints_cache.get(model)
    if hit and time.time() - hit[0] < 600:
        return hit[1]
    base = config.OPENROUTER_URL.rsplit("/chat/completions", 1)[0]
    r = await client().get(f"{base}/models/{model}/endpoints")
    if r.status_code == 404:
        raise LLMError(f"OpenRouter has no model {model!r}")
    r.raise_for_status()
    out = []
    for e in r.json()["data"]["endpoints"]:
        p = e.get("pricing") or {}
        out.append({
            "tag": e.get("tag") or "",
            "provider": e.get("provider_name") or "",
            "quantization": e.get("quantization") if e.get("quantization") not in (None, "unknown") else "",
            "prompt": float(p.get("prompt") or 0) * 1e6,  # USD per million tokens
            "completion": float(p.get("completion") or 0) * 1e6,
            "context": e.get("context_length"),
            "uptime": e.get("uptime_last_30m"),
            "status": e.get("status", 0),  # < 0 = degraded / down
        })
    out = [e for e in out if e["tag"]]
    _endpoints_cache[model] = (time.time(), out)
    return out
