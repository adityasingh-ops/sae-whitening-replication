"""Train one SAE, with or without covariance whitening.

Example (Mac pilot):
    uv run python whitening/scripts/train_sae.py --arch topk --k 32 --tokens 2_000_000
    uv run python whitening/scripts/train_sae.py --arch topk --k 32 --tokens 2_000_000 --whiten
"""

import argparse
import json
import time
from pathlib import Path

import torch
from sae_lens import (
    BatchTopKTrainingSAEConfig,
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    LoggingConfig,
    StandardTrainingSAEConfig,
    TopKTrainingSAEConfig,
)
from sae_lens.training.activation_scaler import ActivationScaler

RESULTS = Path(__file__).resolve().parents[1] / "results"
LOCAL_TRAIN = Path(__file__).resolve().parents[2] / "data" / "pile_train"
DEFAULT_EPS = 1e-3  # estimate_whitening's default


def latest_checkpoint(out_dir: Path) -> Path | None:
    """Newest checkpoint under out_dir/checkpoints/<run id>/<samples seen>/, if any."""
    steps = [p for p in (out_dir / "checkpoints").glob("*/*") if p.is_dir() and p.name.isdigit()]
    return max(steps, key=lambda p: int(p.name)) if steps else None


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_sae_cfg(args):
    common = dict(
        d_in=args.d_in,
        d_sae=args.d_in * args.expansion,
        normalize_activations="covariance_whitening" if args.whiten else "none",
        device=args.device,
        dtype="float32",
    )
    if args.arch == "relu":
        # ramp the L1 penalty up over the first steps so features form before sparsity
        # pressure kills them (without it, high-L1 runs ended with 55-66% dead latents)
        warmup = int(args.l1_warmup_frac * args.tokens / args.batch_tokens)
        return StandardTrainingSAEConfig(l1_coefficient=args.l1, l1_warm_up_steps=warmup, **common)
    if args.arch == "topk":
        return TopKTrainingSAEConfig(k=args.k, **common)
    if args.arch == "batchtopk":
        return BatchTopKTrainingSAEConfig(k=args.k, **common)
    raise ValueError(args.arch)


def run_name(args) -> str:
    sparsity = f"l1{args.l1}w{args.l1_warmup_frac:g}" if args.arch == "relu" else f"k{args.k}"
    whiten = "white" if args.whiten else "plain"
    if args.whiten and args.whiten_eps != DEFAULT_EPS:
        whiten += f"e{args.whiten_eps:g}"
    short_model = args.model.split("/")[-1]
    seed = "" if args.seed is None else f"_s{args.seed}"
    return f"{short_model}_L{args.layer}_{args.arch}_{sparsity}_x{args.expansion}_{whiten}_{args.tokens // 1_000_000}M{seed}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--d-in", type=int, default=512)
    p.add_argument("--arch", choices=["relu", "topk", "batchtopk"], default="topk")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--l1", type=float, default=5.0)
    p.add_argument("--l1-warmup-frac", type=float, default=0.05)
    p.add_argument("--expansion", type=int, default=8)
    p.add_argument("--whiten", action="store_true")
    p.add_argument(
        "--whiten-eps",
        type=float,
        default=DEFAULT_EPS,
        help="added to the covariance eigenvalues; large eps = partial whitening (closer to plain)",
    )
    p.add_argument("--tokens", type=int, default=2_000_000)
    p.add_argument("--batch-tokens", type=int, default=4096)
    p.add_argument("--context-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default=default_device())
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--data",
        default=str(LOCAL_TRAIN) if LOCAL_TRAIN.exists() else "monology/pile-uncopyrighted",
        help="local cache from cache_pile.py (same docs in the same order) or the HF dataset",
    )
    p.add_argument("--n-checkpoints", type=int, default=4)
    args = p.parse_args()

    # LanguageModelSAETrainingRunner never applies cfg.seed, so seed torch here:
    # this fixes the SAE init and the mixing-buffer shuffles for the run
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # force the whitening eps: older SAELens has no config field for it, and newer
    # versions pass whitening_eps explicitly, which would override a changed default
    if args.whiten_eps != DEFAULT_EPS:
        estimate_whitening = ActivationScaler.estimate_whitening

        def estimate_whitening_with_eps(self, *a, **kw):
            print(f"estimating whitening with eps={args.whiten_eps}", flush=True)
            return estimate_whitening(self, *a, **{**kw, "eps": args.whiten_eps})

        ActivationScaler.estimate_whitening = estimate_whitening_with_eps

    name = run_name(args)
    out_dir = RESULTS / name
    if (out_dir / "run.json").exists() and (out_dir / "sae").exists():
        print(f"saved {name} already -> {out_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    resume = latest_checkpoint(out_dir)
    if resume is not None:
        print(f"resuming {name} from {resume}", flush=True)

    cfg = LanguageModelSAERunnerConfig(
        model_name=args.model,
        hook_name=f"blocks.{args.layer}.hook_resid_post",
        dataset_path=args.data,
        is_dataset_tokenized=False,
        streaming=True,
        context_size=args.context_size,
        sae=build_sae_cfg(args),
        lr=args.lr,
        lr_warm_up_steps=0,
        lr_decay_steps=(args.tokens // args.batch_tokens) // 5,
        train_batch_size_tokens=args.batch_tokens,
        training_tokens=args.tokens,
        n_sequences_in_buffer=32,
        store_batch_size_prompts=16,
        logger=LoggingConfig(log_to_wandb=args.wandb, wandb_project="whitening-replication", run_name=name),
        device=args.device,
        act_store_device=args.device,
        n_checkpoints=args.n_checkpoints,
        checkpoint_path=str(out_dir / "checkpoints"),
        save_final_checkpoint=False,
        resume_from_checkpoint=str(resume) if resume is not None else None,
        dtype="float32",
        seed=args.seed if args.seed is not None else 42,
    )

    start = time.time()
    sae = LanguageModelSAETrainingRunner(cfg).run()
    elapsed = time.time() - start

    # inference form, as SAELens's runner saves it (BatchTopK becomes JumpReLU)
    sae.save_inference_model(str(out_dir / "sae"))
    (out_dir / "run.json").write_text(
        json.dumps({**vars(args), "name": name, "train_seconds": round(elapsed, 1),
                    "resumed_from": str(resume) if resume is not None else None}, indent=2)
    )
    print(f"saved {name} in {elapsed / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
