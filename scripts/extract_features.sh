#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-python}
PY_E2V=${PY_E2V:-$PY}
L=logs/features
mkdir -p "$L" features

$PY -m ser.meta
$PY -m ser.feat_handcrafted > "$L/handcrafted.log" 2>&1
: > "$L/ssl.log"
while read -r name model; do
  $PY -m ser.feat_ssl "$name" "$model" >> "$L/ssl.log" 2>&1
done <<'EOF'
wavlm_large microsoft/wavlm-large
hubert_large facebook/hubert-large-ll60k
xlsr_300m facebook/wav2vec2-xls-r-300m
EOF
$PY -m ser.feat_ssl whisper_large_v3 openai/whisper-large-v3 > "$L/whisper.log" 2>&1
$PY_E2V -m ser.feat_e2v e2v_base emotion2vec/emotion2vec_base > "$L/emotion2vec.log" 2>&1
$PY_E2V -m ser.feat_e2v e2v_plus_large emotion2vec/emotion2vec_plus_large >> "$L/emotion2vec.log" 2>&1
