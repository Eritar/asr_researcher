"""Conversation sessions, autosaved as append-only JSON Lines files.

One file per session in SESSIONS_DIR (default ~/.local/share/call-assist/sessions/). Every final
sentence and every completed answer is appended the moment it exists, so a crash loses nothing.
Records (one JSON object per line):

  {"type": "meta",   "id", "created", "title"}          first line; later ones rename the session
  {"type": "utt",    "utt_id", "text", "ts", "dur"}      a final transcript sentence
  {"type": "answer", "utt_id", "mode", "turns": [{"q", "text"}], "model", "provider", "ts"}
                                                         the whole thread for (sentence, mode);
                                                         the last record for a pair wins
"""

import json
import logging
import os
import re
import time
from pathlib import Path

from . import config

log = logging.getLogger(__name__)
MODE_TITLES = {"explain": "Explain", "expand": "Expand", "answer": "Answer"}


def sessions_dir() -> Path:
    return Path(config.SESSIONS_DIR).expanduser()


class Session:
    def __init__(self, path: Path, meta: dict):
        self.path = path
        self.meta = meta
        self.utts: dict[str, dict] = {}  # utt_id -> utt record (insertion ordered)
        self.threads: dict[tuple[str, str], dict] = {}  # (utt_id, mode) -> answer record

    @property
    def id(self) -> str:
        return self.meta["id"]

    @classmethod
    def new(cls) -> "Session":
        now = time.time()
        sid = time.strftime("%Y-%m-%d_%H%M%S", time.localtime(now))
        title = time.strftime("Session %d %b %Y, %H:%M", time.localtime(now))
        # The file is only created on the first write, so idle restarts leave no empty files.
        return cls(sessions_dir() / f"{sid}.jsonl", {"type": "meta", "id": sid, "created": now, "title": title})

    @classmethod
    def load(cls, path: Path) -> "Session":
        s = None
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:  # e.g. a line cut off by a crash
                    continue
                kind = rec.get("type")
                if kind == "meta":
                    if s is None:
                        s = cls(path, rec)
                    else:
                        s.meta["title"] = rec.get("title", s.meta["title"])
                elif s is None:
                    continue
                elif kind == "utt":
                    s.utts[rec["utt_id"]] = rec
                elif kind == "answer":
                    s.threads[(rec["utt_id"], rec["mode"])] = rec
        if s is None:
            raise ValueError(f"{path.name} is not a session file")
        return s

    # -- writing -----------------------------------------------------------
    def _append(self, rec: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        with self.path.open("a", encoding="utf-8") as f:
            if new:
                f.write(json.dumps(self.meta, ensure_ascii=False) + "\n")
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def add_utt(self, ev: dict) -> None:
        rec = {"type": "utt", "utt_id": ev["utt_id"], "text": ev["text"], "ts": ev.get("ts"),
               "dur": ev.get("dur")}
        self.utts[rec["utt_id"]] = rec
        self._append(rec)

    def save_thread(self, utt_id: str, mode: str, turns: list[dict], model: str, provider: str | None) -> dict:
        rec = {"type": "answer", "utt_id": utt_id, "mode": mode, "turns": turns, "model": model,
               "provider": provider, "ts": time.time()}
        self.threads[(utt_id, mode)] = rec
        self._append(rec)
        return rec

    def rename(self, title: str) -> None:
        self.meta["title"] = title
        if self.path.exists():
            self._append({"type": "meta", "id": self.id, "title": title})

    # -- reading -----------------------------------------------------------
    def answered(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for utt_id, mode in self.threads:
            out.setdefault(utt_id, []).append(mode)
        return out

    def summary(self) -> dict:
        first = next(iter(self.utts.values()), None)
        return {
            "id": self.id,
            "title": self.meta["title"],
            "created": self.meta["created"],
            "updated": self.path.stat().st_mtime if self.path.exists() else self.meta["created"],
            "sentences": len(self.utts),
            "answers": len(self.threads),
            "preview": first["text"][:120] if first else "",
            "path": str(self.path),
        }

    def to_markdown(self) -> str:
        fmt = lambda ts: time.strftime("%H:%M:%S", time.localtime(ts)) if ts else ""  # noqa: E731
        lines = [f"# {self.meta['title']}", "",
                 time.strftime("_%A, %d %B %Y, %H:%M_", time.localtime(self.meta["created"])), ""]
        for utt_id, u in self.utts.items():
            lines += [f"**{fmt(u.get('ts'))}**  {u['text']}", ""]
            for mode in MODE_TITLES:
                t = self.threads.get((utt_id, mode))
                if not t:
                    continue
                lines.append(f"> **{MODE_TITLES[mode]}** · _{t['model']}_")
                for turn in t["turns"]:
                    if turn.get("q"):
                        lines += [">", f"> **Q:** {turn['q']}"]
                    lines += [">"] + [f"> {ln}" if ln else ">" for ln in turn["text"].splitlines()]
                lines.append("")
        return "\n".join(lines).rstrip() + "\n"


def list_sessions() -> list[dict]:
    d = sessions_dir()
    out = []
    for p in d.glob("*.jsonl") if d.exists() else []:
        try:
            out.append(Session.load(p).summary())
        except (OSError, ValueError) as e:
            log.warning("skipping %s: %s", p, e)
    return sorted(out, key=lambda s: -s["updated"])


def path_for(session_id: str) -> Path:
    if not re.fullmatch(r"[\w.-]+", session_id):
        raise ValueError(f"bad session id {session_id!r}")
    return sessions_dir() / f"{session_id}.jsonl"
