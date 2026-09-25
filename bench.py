"""Measure WER / speed of the ASR engines on YOUR audio.

Put pairs of files in a folder:  talk1.wav + talk1.txt (reference transcript), ...
Any format ffmpeg can read works for audio (.wav/.mp3/.m4a/.ogg/.flac).

  uv run python bench.py --data samples/ --engines parakeet qwen3
"""

import argparse
import re
import sys
import time
import unicodedata
from pathlib import Path

from app import audio, config

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".opus", ".webm"}


def normalize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("ß", "ss")
    text = re.sub(r"[^\w\s']|_", " ", text)
    return text.split()


def edit_distance(ref: list, hyp: list) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def wer(ref: str, hyp: str) -> tuple[int, int]:
    r, h = normalize(ref), normalize(hyp)
    return edit_distance(r, h), len(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=config.ROOT / "samples")
    ap.add_argument("--engines", nargs="+", default=["parakeet"])
    ap.add_argument("-v", "--verbose", action="store_true", help="print every hypothesis")
    args = ap.parse_args()

    pairs = sorted(
        (p, p.with_suffix(".txt")) for p in args.data.iterdir()
        if p.suffix.lower() in AUDIO_EXT and p.with_suffix(".txt").exists()
    )
    if not pairs:
        sys.exit(f"no <name>.wav + <name>.txt pairs in {args.data}")

    from app.engines import Engine

    clips = [(a.name, audio.load_audio(a), t.read_text(encoding="utf-8").strip()) for a, t in pairs]
    total_audio = sum(len(x) for _, x, _ in clips) / config.SAMPLE_RATE
    print(f"{len(clips)} clips, {total_audio:.1f}s audio, {config.NUM_THREADS} threads\n")

    for name in args.engines:
        t = time.perf_counter()
        eng = Engine(name)
        eng.warmup()
        load_s = time.perf_counter() - t
        errs = words = 0
        decode_s = 0.0
        for fname, x, ref in clips:
            t = time.perf_counter()
            hyp = eng.transcribe(x)
            decode_s += time.perf_counter() - t
            e, n = wer(ref, hyp)
            errs, words = errs + e, words + n
            if args.verbose:
                print(f"  [{name}] {fname}  WER {e / max(n, 1):.1%}\n    REF: {ref}\n    HYP: {hyp}")
        print(f"{name:9s} WER {errs / max(words, 1):6.2%}   RTF {decode_s / total_audio:.3f}"
              f"   decode {decode_s:.2f}s   load {load_s:.1f}s")


if __name__ == "__main__":
    main()
