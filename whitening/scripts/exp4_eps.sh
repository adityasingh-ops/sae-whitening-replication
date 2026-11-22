#!/bin/zsh
# Exp 4: partial whitening. TopK k=32 on Pythia-70M L3, 10M tokens, seed 1, sweeping the eps
# added to the covariance eigenvalues (median eigenvalue ~0.08, so eps=1e-3 is full whitening
# and eps=100 is close to plain). The eps=1e-3 and plain runs already exist from run_seeds.sh.
# Usage: ./exp4_eps.sh [seeds] [eps values], e.g. ./exp4_eps.sh "2 3" "0.1 1 10"
cd "$(dirname "$0")/../.."
SEEDS=(${=1:-1})
EPS=(${=2:-0.01 0.1 1 10 100})
LOG=whitening/results/exp4_eps_$(date +%Y%m%d_%H%M).log
# wait for any other training on this Mac to finish first
while pgrep -f "calibrate_relu|run_seeds" > /dev/null; do sleep 300; done
# Retry a training run up to 3 times; train_sae.py resumes from its latest checkpoint
train() {
  for attempt in 1 2 3; do
    out=$(uv run python whitening/scripts/train_sae.py "$@" 2>&1 | grep -aE '^saved|^resuming|^estimating whitening with|Error|Traceback')
    echo "$out" | tee -a $LOG
    echo "$out" | grep -q '^saved' && return 0
    echo "[$(date +%H:%M)] attempt $attempt failed, retrying in 60s" | tee -a $LOG
    sleep 60
  done
  echo "[$(date +%H:%M)] gave up: $*" | tee -a $LOG
}
for seed in $SEEDS; do
  for eps in $EPS; do
    echo "[$(date +%H:%M)] train seed=$seed eps=$eps" | tee -a $LOG
    train --arch topk --k 32 --tokens 10_000_000 --seed $seed --whiten --whiten-eps $eps
  done
done
echo "[$(date +%H:%M)] eval" | tee -a $LOG
uv run python whitening/scripts/eval_sae.py whitening/results/*_topk_k32_x8_whitee*_10M_s* --batches 20 2>&1 | grep -aE '^pythia|Error|Traceback' | tee -a $LOG
echo "[$(date +%H:%M)] done" | tee -a $LOG
