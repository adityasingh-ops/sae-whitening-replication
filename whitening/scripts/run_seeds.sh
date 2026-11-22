#!/bin/zsh
# Plain vs whitened (ARCH, default topk) k=32 on Pythia-70M L3, 10M tokens, seeds 1-3, then eval.
# Usage: ./run_seeds.sh [topk|batchtopk]
# caffeinate keeps the Mac awake for the whole run.
cd "$(dirname "$0")/../.."
ARCH=${1:-topk}
LOG=whitening/results/seeds_${ARCH}_$(date +%Y%m%d_%H%M).log
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
for seed in 1 2 3; do
  for w in "" "--whiten"; do
    echo "[$(date +%H:%M)] train seed=$seed ${w:-plain}" | tee -a $LOG
    train --arch $ARCH --k 32 --tokens 10_000_000 --seed $seed $w
  done
done
echo "[$(date +%H:%M)] eval" | tee -a $LOG
uv run python whitening/scripts/eval_sae.py whitening/results/*_${ARCH}_k32_*_10M_s* --batches 20 2>&1 | grep -aE '^pythia|Error|Traceback' | tee -a $LOG
echo "[$(date +%H:%M)] done" | tee -a $LOG
