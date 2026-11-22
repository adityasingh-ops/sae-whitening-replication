#!/bin/zsh
# Short ReLU runs over an L1 grid, plain and whitened, to find the L1 that gives L0 ~ 32.
cd "$(dirname "$0")/../.."
LOG=whitening/results/calibrate_relu_$(date +%Y%m%d_%H%M).log
# Retry a training run up to 3 times; train_sae.py resumes from its latest checkpoint
train() {
  for attempt in 1 2 3; do
    out=$(uv run python whitening/scripts/train_sae.py "$@" 2>&1 | grep -aE '^saved|^resuming|Error|Traceback')
    echo "$out" | tee -a $LOG
    echo "$out" | grep -q '^saved' && return 0
    echo "[$(date +%H:%M)] attempt $attempt failed, retrying in 60s" | tee -a $LOG
    sleep 60
  done
  echo "[$(date +%H:%M)] gave up: $*" | tee -a $LOG
}
# L0 ~ 32 needs much larger L1 than first tried (L1=1 gave L0 ~ 575 plain)
for w in "" "--whiten"; do
  # fill in the CE-vs-L0 curves: whitened L1=4 still gave L0~240, plain L1=5 gave L0~230
  if [ -z "$w" ]; then grid=(3 7); else grid=(6 8 12 16); fi
  for l1 in $grid; do
    echo "[$(date +%H:%M)] relu l1=$l1 ${w:-plain}" | tee -a $LOG
    train --arch relu --l1 $l1 --tokens 2_000_000 --seed 1 $w
  done
done
echo "[$(date +%H:%M)] eval" | tee -a $LOG
uv run python whitening/scripts/eval_sae.py whitening/results/*_relu_*w0.05_*_2M_s1 --batches 5 2>&1 | grep -aE '^pythia|Error|Traceback' | tee -a $LOG
echo "[$(date +%H:%M)] done" | tee -a $LOG
