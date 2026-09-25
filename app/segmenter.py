"""Turns a frame stream into utterances using Silero VAD (via sherpa-onnx).

Emits, per utterance:
  ("partial", utt_id, audio_so_far, start_s)   every PARTIAL_INTERVAL_S while speech lasts
  ("final",   utt_id, audio,        start_s)   once the VAD closes the segment
"""

from collections.abc import Iterator

import numpy as np

from . import config

PRE_ROLL_S = 0.25  # audio kept before the VAD's segment start (soft onsets)
POST_ROLL_S = 0.10


def make_vad():
    import sherpa_onnx

    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = str(config.MODELS_DIR / "silero_vad.onnx")
    cfg.silero_vad.threshold = config.VAD_THRESHOLD
    cfg.silero_vad.min_silence_duration = config.VAD_MIN_SILENCE_S
    cfg.silero_vad.min_speech_duration = config.VAD_MIN_SPEECH_S
    cfg.silero_vad.max_speech_duration = config.MAX_UTTERANCE_S
    cfg.silero_vad.window_size = 512
    cfg.sample_rate = config.SAMPLE_RATE
    cfg.num_threads = 1
    return sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=config.MAX_UTTERANCE_S + 10)


class Segmenter:
    def __init__(self, vad, sample_rate: int = config.SAMPLE_RATE,
                 partial_interval_s: float = config.PARTIAL_INTERVAL_S,
                 lookback_s: float = config.VAD_MIN_SPEECH_S + 0.1):
        self.vad = vad
        self.sr = sample_rate
        self.partial_step = int(partial_interval_s * sample_rate)
        self.lookback = int((lookback_s + PRE_ROLL_S) * sample_rate)
        self.pre = int(PRE_ROLL_S * sample_rate)
        self.post = int(POST_ROLL_S * sample_rate)
        # Rolling history of recent audio, addressed by absolute sample index.
        self.keep = int((config.MAX_UTTERANCE_S + 10) * sample_rate)
        self.hist = np.zeros(0, dtype=np.float32)
        self.hist_base = 0  # absolute index of hist[0]
        self.total = 0
        self.next_id = 0
        self.utt_id: int | None = None
        self.utt_start = 0
        self.last_partial_at = 0
        self.last_final_end = 0

    def _slice(self, a: int, b: int) -> np.ndarray:
        a = max(a, self.hist_base)
        return self.hist[a - self.hist_base : b - self.hist_base].copy()

    def feed(self, frame: np.ndarray) -> Iterator[tuple]:
        self.hist = np.concatenate([self.hist, frame])
        self.total += len(frame)
        if len(self.hist) > self.keep:
            drop = len(self.hist) - self.keep
            self.hist = self.hist[drop:]
            self.hist_base += drop

        self.vad.accept_waveform(frame)

        while not self.vad.empty():
            # `front` is a view into the VAD's buffer: copy what we need before pop().
            seg = self.vad.front
            samples = np.array(seg.samples, dtype=np.float32)
            start = seg.start
            self.vad.pop()
            end = start + len(samples)
            a = max(start - self.pre, self.last_final_end)
            b = min(end + self.post, self.total)
            audio = self._slice(a, b)
            if len(audio) < len(samples):  # history too short, fall back to VAD copy
                audio = samples
            uid = self.utt_id if self.utt_id is not None else self._new_id()
            yield ("final", uid, audio, a / self.sr)
            self.utt_id = None
            self.last_final_end = b

        if self.vad.is_speech_detected():
            if self.utt_id is None:
                self.utt_id = self._new_id()
                self.utt_start = max(self.total - self.lookback, self.last_final_end)
                self.last_partial_at = self.total
            if self.total - self.last_partial_at >= self.partial_step:
                self.last_partial_at = self.total
                yield ("partial", self.utt_id, self._slice(self.utt_start, self.total),
                       self.utt_start / self.sr)

    def _new_id(self) -> int:
        self.next_id += 1
        return self.next_id
