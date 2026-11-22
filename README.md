# sae-whitening-replication

Replication of [Data Whitening Improves Sparse Autoencoder Learning](https://arxiv.org/abs/2511.13981) using SAELens's `normalize_activations="covariance_whitening"`, for [SAELens #647](https://github.com/decoderesearch/SAELens/issues/647).

Work in progress. Results so far are on Pythia-70M and Pythia-160M with fewer tokens than the paper. See the caveats at the bottom.

## Results

All numbers are means over 3 seeds, 8x expansion, `resid_post`, evaluated on held-out Pile val with SAELens `run_evals`.

**Pythia-160M layer 8, TopK k=32, 30M tokens** (trained on Kaggle 2x T4)

| | CE score | explained var | dead |
|---|---|---|---|
| plain | 0.926 | 0.943 | 12.4% |
| whitened | 0.935 | 0.921 | 0.8% |

**Pythia-70M layer 3, k=32, 10M tokens** (M1 Mac)

| | CE score | explained var | dead |
|---|---|---|---|
| TopK plain | 0.848 | 0.926 | 21.5% |
| TopK whitened | 0.888 | 0.901 | 5.9% |
| BatchTopK plain | 0.820 | 0.919 | 40.6% |
| BatchTopK whitened | 0.891 | 0.900 | 8.0% |

**Partial whitening** (TopK, Pythia-70M). `estimate_whitening` adds `eps` to the covariance eigenvalues, so each direction ends up with variance `lambda / (lambda + eps)`. The median eigenvalue here is ~0.08, so the default `1e-3` is full whitening.

| eps | CE score | explained var | dead |
|---|---|---|---|
| 1e-3 (default) | 0.888 | 0.901 | 5.9% |
| 0.1 | 0.896 | 0.914 | 6.3% |
| 1 | 0.896 | 0.929 | 7.9% |
| 10 | 0.883 | 0.933 | 12.2% |
| plain | 0.848 | 0.926 | 21.5% |

**Where the CE gap comes from** (`exp2_directions.py`, `exp2b_joint.py`, results in `whitening/results/exp2/`)

- **Plain SAEs reconstruct the low-variance directions badly.** Those directions are the covariance eigenvectors with small eigenvalues.
- **Individually these directions barely matter, but together they do.** Ablating the bottom 384 of 512 directions (~5.5% of the variance) costs about 4x the sum of the individual ablations.
- **They account for most of the plain SAE's CE gap.** Patching the true activation back in along only those directions removes 77-79% of the plain SAE's CE gap, vs 54-56% for the whitened SAE.

## Layout

- `whitening/scripts/train_sae.py`: trains one SAE (`--arch relu|topk|batchtopk`, `--whiten`, `--whiten-eps`, `--seed`). It saves checkpoints and resumes after a crash.
- `whitening/scripts/eval_sae.py`: `run_evals` on held-out data, writes `eval.json` next to each run
- `whitening/scripts/run_seeds.sh`, `exp4_eps.sh`, `calibrate_relu.sh`: the sweeps behind the tables above
- `whitening/scripts/cache_pile.py`: caches the first Pile docs locally, in the same order as streaming, so runs don't depend on the network
- `kaggle/whiten160m/`: the Kaggle job for the Pythia-160M runs (`python build.py 30_000_000 "(1, 2, 3)" run_whiten160m.py`)
- `whitening/results/*/`: `run.json` (all arguments) and `eval.json` per SAE. Weights are not included.

## Running

Needs `sae-lens` 6.x. A few things in this repo depended on SAELens fixes:

- MPS float64 fixes: SAELens #797, merged
- BatchTopK on MPS: SAELens #802
- the runner ignoring `cfg.seed`: SAELens #800. `train_sae.py` seeds torch itself, so this doesn't matter here.

```bash
uv run python whitening/scripts/cache_pile.py
uv run python whitening/scripts/train_sae.py --arch topk --k 32 --tokens 10_000_000 --seed 1 --whiten --whiten-eps 1
uv run python whitening/scripts/eval_sae.py whitening/results/*_s1
```

## Caveats

- **Models:** no GPT-2 small or Gemma yet.
- **Tokens:** 10-30M per SAE, against the paper's 500M.
- **Pythia-160M coverage:** TopK only, at a single sparsity.
- **ReLU:** only short 2M-token calibration runs so far, not a proper L0 sweep.
- **Two bad runs:** `exp4_eps_20261122_1312.log` contains seed-3 eps=1 and eps=10 runs that silently trained with the default eps after a SAELens version change. Those runs were discarded and rerun (`exp4_eps_20261122_1737.log`), and `train_sae.py` now forces the eps.
- **Interpretability:** not measured yet. SAEBench metrics (sparse probing, SCR/TPP, absorption) are next.
