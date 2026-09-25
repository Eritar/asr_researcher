# Call Assist

Live transcript of **what your computer plays** (the other people in a call), fully local on CPU.
Click any sentence and an OpenRouter model explains it, expands on it, or answers it, streamed live.
Runs on Ubuntu (PipeWire/PulseAudio) and macOS 14.2+.

- ASR: [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (ONNX Runtime, no PyTorch), Silero VAD.
  Default model: NVIDIA **Parakeet TDT 0.6B v3** INT8 (English + German + 23 more languages, auto-detected).
- UI: one static HTML page served by FastAPI on `localhost`, with no build step, no Node and no CDN.
- Audio: only what your computer plays, never your mic.
  - Ubuntu: `parec` records the PipeWire/PulseAudio *monitor* of an output.
  - macOS: a small Swift helper (`app/native/systap.swift`, compiled automatically on first run) uses a
    Core Audio process tap. It can record everything, or **just one app** (Teams, Zoom, a browser…) so music and
    notifications stay out of the transcript. No BlackHole or other virtual driver is needed.

## Setup

```bash
# Ubuntu: sudo apt install pulseaudio-utils ffmpeg     (parec is usually present already)
# macOS:  xcode-select --install                        (Swift compiler for the audio helper)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
scripts/download_models.sh parakeet          # ~460 MB (add: qwen3 cohere canary)
uv run python -m app.main                    # then open http://127.0.0.1:8765
```

Then open **Settings** in the page, paste your OpenRouter API key (press **Test** to check it), pick a model,
and choose what to listen to. Settings are saved to `~/.config/call-assist/settings.json` (readable only by
you) and take precedence over `.env` / environment variables, which are optional.

**macOS permission:** the first capture asks to allow *System Audio Recording* for the app that started the server
(Terminal, iTerm, VS Code…). If the level meter stays empty while audio plays, enable it under
System Settings → Privacy & Security → Screen & System Audio Recording, then press **Retry**.

Keys: click a sentence = explain · `1` Explain · `2` Expand · `3` Answer · `Esc` stop. Use the box at the bottom for follow-up questions.
**Provider** (Settings, under Model) picks who runs the model on OpenRouter: *Automatic · lowest latency* is a good
choice for live use, or pin one specific provider/endpoint (the list shows each one's price and uptime) with an
optional fallback to others. The answer footer shows which provider actually replied.
**Thinking** (Settings) is off by default: most "flash" models otherwise reason silently for several seconds
before the first word. With thinking off and *lowest latency* routing, the first word typically arrives in
0.4–1 s. Models that cannot disable thinking (e.g. Gemini Flash) automatically use minimal effort instead.
In Settings, "About you" tells the model your background so answers are pitched at the right level, and
"Answer language" can force English or German answers.

### Sessions and saved answers

Everything is autosaved while it happens: each finished sentence and each completed answer is appended to
`~/.local/share/call-assist/sessions/<date_time>.jsonl` (the folder can be changed in Settings). Nothing is lost if the
app or computer crashes.

- Answers are generated once per sentence and mode. Clicking the sentence again shows the saved answer instantly
  (no new request, no cost); **Regenerate** asks again and replaces it. Follow-up questions are saved with the thread.
  A dot on Explain/Expand/Answer shows which answers exist for the selected sentence.
- Clicking another sentence while an answer is still being written doesn't throw it away: it finishes in the
  background and is saved. Only **Stop** / `Esc` cancels.
- **New** starts a fresh session (the old one stays saved). **Sessions** lists them: open one to read it or to
  continue it live, rename, delete, or export as Markdown (readable) or the raw `.jsonl`.

If nothing is transcribed on Ubuntu, pick the right output under Settings → *Listen to* (useful when the call goes
to a headset that is not the default output).

## Engines (`ASR_ENGINE`)

| engine     | size   | notes |
|------------|--------|-------|
| `parakeet` | 0.6B   | default. Fastest (~3% of real time on an M1), strong WER, auto language. No vocabulary biasing. |
| `qwen3`    | 0.6B   | Qwen3-ASR. ~4× slower than Parakeet but still real-time; uses the **glossary** (Settings) as hotwords for your jargon. |
| `cohere`   | 2B     | Cohere Transcribe, best leaderboard WER; heavy, needs `LANGUAGE=en` or `de` (one language per session). |
| `canary`   | 180M   | tiny, for weak CPUs; needs `LANGUAGE` (it translates otherwise). |

Pick with data, not leaderboards. Record a few minutes of the kind of talk you care about, write the reference
transcripts, put them in `samples/` as `name.wav` + `name.txt`, then:

```bash
uv run python bench.py --data samples --engines parakeet qwen3 cohere -v
```

## Latency

Utterances are cut by the VAD. While someone speaks, the growing utterance is re-decoded every
`PARTIAL_INTERVAL_S` (0.3 s) for live grey text, and the final text arrives about `VAD_MIN_SILENCE_S` (0.35 s)
+ decode time after they stop. That was 0.4–0.8 s on an M1 Pro with Parakeet; hover over a line to see its numbers.
If the CPU can't keep up, stale partial decodes are skipped, so it never falls behind.

## Other options

- Settings → *Listen to* → "Audio file" replays a recording in real time (for testing).
- GPU: sherpa-onnx has CUDA builds (see the sherpa-onnx docs, "Install the CUDA version"); then set `ASR_PROVIDER=cuda`.
- Everything configurable is in `app/config.py` / `.env.example`.

## Layout

```
app/audio.py      parec / macOS tap / file sources -> 16 kHz float32 frames
app/native/       systap.swift, the macOS system-audio helper
app/settings.py   settings edited in the UI (~/.config/call-assist/settings.json)
app/sessions.py   autosaved sessions + saved answers (.jsonl), Markdown export
app/segmenter.py  Silero VAD -> partial/final utterance audio
app/engines.py    sherpa-onnx recognizers (parakeet/qwen3/cohere/canary)
app/pipeline.py   capture thread + ASR thread, drops stale partials
app/llm.py        OpenRouter streaming + prompts
app/main.py       FastAPI + WebSocket
app/static/index.html
bench.py          WER/speed on your own clips
```

Known limit: overlapping speech is transcribed as one mixed line (none of these models separate simultaneous speakers).
