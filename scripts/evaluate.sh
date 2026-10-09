#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONWARNINGS="ignore::FutureWarning"
PY=${PY:-python}
L=logs/evaluation
mkdir -p "$L" results/predictions

: > results/summary.jsonl
: > "$L/eval_ssl.log"
for m in wavlm_large xlsr_300m hubert_large whisper_large_v3; do
  $PY -m ser.run_eval ssl "$m" >> "$L/eval_ssl.log" 2>&1
done
$PY -m ser.run_eval handcrafted > "$L/eval_handcrafted.log" 2>&1
$PY -m ser.run_eval e2v e2v_base e2v_plus_large > "$L/eval_e2v.log" 2>&1
$PY -m ser.run_eval emobox wavlm_large hubert_large xlsr_300m whisper_large_v3 > "$L/eval_emobox.log" 2>&1
$PY -m ser.run_eval best wavlm_large > "$L/eval_best.log" 2>&1
$PY -m ser.eval_e2v_zeroshot > "$L/eval_e2v_zeroshot.log" 2>&1
cp features/e2v_plus_large_scores.npy features/e2v_plus_large_labels.json results/predictions/
$PY -m ser.run_eval inflation wavlm_large xlsr_300m hubert_large whisper_large_v3 handcrafted egemaps compare > "$L/eval_inflation.log" 2>&1
$PY -m ser.run_eval perm wavlm_large 0 1 2 > "$L/eval_perm.log" 2>&1
$PY -m ser.collect_ft results/finetune
