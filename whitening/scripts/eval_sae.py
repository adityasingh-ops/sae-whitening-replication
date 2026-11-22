"""Evaluate trained SAEs on the Pile validation split with SAELens's run_evals.

Saved SAEs have whitening folded into their weights, so every SAE is evaluated
on raw activations with an identity ActivationScaler.

Example:
    uv run python whitening/scripts/eval_sae.py whitening/results/*/
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from sae_lens import SAE
from sae_lens.analysis.compat import default_model_class_name
from sae_lens.evals import EvalConfig, run_evals
from sae_lens.load_model import load_model
from sae_lens.training.activation_scaler import ActivationScaler
from sae_lens.training.activations_store import ActivationsStore

# training streams the train split; this file is never trained on
VAL_FILES = {"validation": "val.jsonl.zst"}
LOCAL_VAL = Path(__file__).resolve().parents[2] / "data" / "pile_val"  # from cache_pile.py

KEEP = [
    "model_performance_preservation/ce_loss_score",
    "model_performance_preservation/ce_loss_with_sae",
    "model_performance_preservation/ce_loss_without_sae",
    "model_behavior_preservation/kl_div_score",
    "reconstruction_quality/explained_variance",
    "reconstruction_quality/mse",
    "sparsity/l0",
    "sparsity/l1",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+", type=Path)
    p.add_argument("--batches", type=int, default=20)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()

    model = None
    for run_dir in args.run_dirs:
        if not (run_dir / "run.json").exists():
            print(f"skipping {run_dir.name}: no run.json (incomplete run)")
            continue
        run = json.loads((run_dir / "run.json").read_text())
        sae = SAE.load_from_disk(run_dir / "sae", device=args.device)
        if model is None:
            model = load_model(default_model_class_name(), run["model"], device=args.device)

        if LOCAL_VAL.exists():
            dataset = load_dataset(str(LOCAL_VAL), split="train", streaming=True)
        else:
            dataset = load_dataset(
                "monology/pile-uncopyrighted", data_files=VAL_FILES, split="validation", streaming=True
            )
        store = ActivationsStore.from_sae(
            model=model,
            sae=sae,
            dataset=dataset,
            context_size=run["context_size"],
            store_batch_size_prompts=16,
            n_batches_in_buffer=4,
            device=args.device,
        )
        eval_cfg = EvalConfig(
            batch_size_prompts=16,
            n_eval_reconstruction_batches=args.batches,
            n_eval_sparsity_variance_batches=args.batches,
            compute_kl=True,
            compute_ce_loss=True,
            compute_sparsity_metrics=True,
            compute_variance_metrics=True,
            compute_featurewise_density_statistics=True,
        )
        metrics, feature_metrics = run_evals(sae, store, model, ActivationScaler(), eval_cfg)

        density = torch.tensor(feature_metrics["feature_density"])
        summary = {}
        for key in KEEP:
            group, name = key.split("/")
            if name in metrics.get(group, {}):
                summary[name] = metrics[group][name]
        summary["dead_frac"] = (density == 0).float().mean().item()
        (run_dir / "eval.json").write_text(json.dumps(summary, indent=2))
        print(run["name"], json.dumps({k: round(v, 4) for k, v in summary.items()}))


if __name__ == "__main__":
    main()
