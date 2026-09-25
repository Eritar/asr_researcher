#!/usr/bin/env bash
# Download ASR + VAD models into ./models.
# Usage: scripts/download_models.sh [parakeet] [qwen3] [cohere] [canary]   (default: parakeet)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models && cd models

BASE=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models
model_dir() {
  case "$1" in
    parakeet) echo sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8 ;;
    qwen3)    echo sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25 ;;
    cohere)   echo sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01 ;;
    canary)   echo sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8 ;;
    *) echo "unknown model '$1' (choose: parakeet qwen3 cohere canary)" >&2; exit 1 ;;
  esac
}

if [ ! -f silero_vad.onnx ]; then
  curl -fL -o silero_vad.onnx "$BASE/silero_vad.onnx"
fi

for key in "${@:-parakeet}"; do
  name=$(model_dir "$key")
  if [ -d "$name" ]; then echo "$name already present"; continue; fi
  curl -fL "$BASE/$name.tar.bz2" | tar xjf -
done
ls
