"""ASR engines. All run through sherpa-onnx (ONNX Runtime, CPU by default, no torch).

Each engine is non-streaming but fast; live partials come from re-decoding the growing
utterance (see pipeline.py).

  parakeet  NVIDIA Parakeet TDT 0.6B v3 (25 EU langs, auto language). Fastest, strong WER.
  qwen3     Qwen3-ASR 0.6B (52 langs, auto language). Accepts glossary hotwords.
  cohere    Cohere Transcribe 2B (14 langs). Best leaderboard WER, heaviest. Needs LANGUAGE.
  canary    NVIDIA Canary 180M flash (en/de/es/fr). Tiny, for weak CPUs. Needs LANGUAGE.
"""

import time

import numpy as np

from . import config

MODEL_DIRS = {
    "parakeet": "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
    "qwen3": "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25",
    "cohere": "sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01",
    "canary": "sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8",
}


def installed() -> dict[str, bool]:
    return {name: (config.MODELS_DIR / d).exists() for name, d in MODEL_DIRS.items()}


class Engine:
    def __init__(self, name: str | None = None):
        import sherpa_onnx

        name = name or config.ASR_ENGINE
        if name not in MODEL_DIRS:
            raise ValueError(f"unknown ASR_ENGINE {name!r}; choose from {list(MODEL_DIRS)}")
        d = config.MODELS_DIR / MODEL_DIRS[name]
        if not d.exists():
            raise FileNotFoundError(f"{d} missing. Run: scripts/download_models.sh {name}")
        self.name = name
        self.language = config.LANGUAGE
        common = dict(num_threads=config.NUM_THREADS, provider=config.ASR_PROVIDER)
        R = sherpa_onnx.OfflineRecognizer

        if name == "parakeet":
            self.rec = R.from_transducer(
                encoder=str(d / "encoder.int8.onnx"), decoder=str(d / "decoder.int8.onnx"),
                joiner=str(d / "joiner.int8.onnx"), tokens=str(d / "tokens.txt"),
                model_type="nemo_transducer", feature_dim=128, **common)
        elif name == "qwen3":
            self.rec = R.from_qwen3_asr(
                conv_frontend=str(d / "conv_frontend.onnx"), encoder=str(d / "encoder.int8.onnx"),
                decoder=str(d / "decoder.int8.onnx"), tokenizer=str(d / "tokenizer"),
                hotwords=",".join(t.replace(",", " ") for t in config.glossary_terms()),
                max_total_len=1024, max_new_tokens=256, **common)
        elif name == "cohere":
            self.language = self.language or "en"
            self.rec = R.from_cohere_transcribe(
                encoder=str(d / "encoder.int8.onnx"), decoder=str(d / "decoder.int8.onnx"),
                tokens=str(d / "tokens.txt"), language=self.language, **common)
        elif name == "canary":
            self.language = self.language or "en"
            self.rec = R.from_nemo_canary(
                encoder=str(d / "encoder.int8.onnx"), decoder=str(d / "decoder.int8.onnx"),
                tokens=str(d / "tokens.txt"), src_lang=self.language, tgt_lang=self.language,
                **common)

    def transcribe(self, audio: np.ndarray) -> str:
        s = self.rec.create_stream()
        if self.name == "cohere":
            s.set_option("language", self.language)
        s.accept_waveform(config.SAMPLE_RATE, audio)
        self.rec.decode_stream(s)
        return s.result.text.strip()

    def warmup(self) -> float:
        t = time.perf_counter()
        self.transcribe(np.zeros(config.SAMPLE_RATE, dtype=np.float32))
        return time.perf_counter() - t
