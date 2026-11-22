"""Exp 2/3: variance vs importance of activation directions, and where SAE error goes.

For blocks.{layer}.hook_resid_post of Pythia-70M, on held-out Pile validation text:
- principal directions e_i of the activations and their variance lambda_i
- importance_i: increase in next-token loss when direction i is mean-ablated
  (the component (x - mu) . e_i is set to zero) at that hook
- grad_importance_i: E[(dL/dx . e_i)^2], a cheap first-order cross-check
- for each trained SAE: E[((x_hat - x) . e_i)^2], the reconstruction error along e_i

Example:
    uv run python whitening/scripts/exp2_directions.py --n-seqs 64 --out whitening/results/exp2
"""

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sae_lens import SAE
from transformer_lens.model_bridge import TransformerBridge

ROOT = Path(__file__).resolve().parents[1]


def load_tokens(tokenizer, n_seqs: int, ctx: int) -> torch.Tensor:
    """Concatenate Pile validation docs (EOS-separated) into n_seqs rows of ctx tokens."""
    ds = load_dataset(
        "monology/pile-uncopyrighted",
        data_files={"validation": "val.jsonl.zst"},
        split="validation",
        streaming=True,
    )
    flat: list[int] = []
    for doc in ds:
        flat.extend(tokenizer.encode(doc["text"]) + [tokenizer.eos_token_id])
        if len(flat) >= n_seqs * ctx:
            break
    return torch.tensor(flat[: n_seqs * ctx]).view(n_seqs, ctx)


EOS_ID = 0  # Pythia's <|endoftext|>


def next_token_loss(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    # Skip <|endoftext|> targets: Pythia scores them so low (loss ~1000) that a single
    # document boundary dominates the mean. SAELens's evals exclude special tokens too.
    targets = tokens[:, 1:].reshape(-1).clone()
    targets[targets == EOS_ID] = -100
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), targets)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--n-seqs", type=int, default=64)
    p.add_argument("--ctx", type=int, default=128)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--sae-glob", default="pythia-70m-deduped_L3_topk_k32_x8_*_10M_s*")
    p.add_argument("--max-dirs", type=int, default=None, help="only ablate the top-N variance dirs (for quick tests)")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--out", default=str(ROOT / "results" / "exp2"))
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hook = f"blocks.{args.layer}.hook_resid_post"
    t0 = time.time()

    model = TransformerBridge.boot_transformers(args.model, device=args.device)
    model.eval()
    tokens = load_tokens(model.tokenizer, args.n_seqs, args.ctx).to(args.device)
    batches = tokens.split(args.batch)

    # 1. activations, clean loss, and gradients of the loss w.r.t. the activations
    acts, grads, clean_losses = [], [], []
    for b in batches:
        store = {}

        def keep(act, hook):
            act.retain_grad()
            store["act"] = act
            return act

        logits = model.run_with_hooks(b, fwd_hooks=[(hook, keep)])
        loss = next_token_loss(logits, b)
        loss.backward()
        acts.append(store["act"].detach().reshape(-1, store["act"].shape[-1]).float().cpu())
        # mean loss over counted targets, so scale per-token gradients back up
        n_targets = (b[:, 1:] != EOS_ID).sum().item()
        grads.append((store["act"].grad * n_targets).reshape(-1, store["act"].shape[-1]).float().cpu())
        clean_losses.append(loss.item())
        model.zero_grad(set_to_none=True)
    X = torch.cat(acts)
    G = torch.cat(grads)
    clean_loss = float(np.mean(clean_losses))

    # 2. principal directions (float64 on CPU)
    mu = X.double().mean(0)
    cov = torch.cov((X.double() - mu).T)
    lam, E = torch.linalg.eigh(cov)
    order = torch.argsort(lam, descending=True)
    lam, E = lam[order], E[:, order]
    E_f = E.float()
    grad_importance = ((G @ E_f) ** 2).mean(0).numpy()

    # 3. ablation importance: zero the centered component along e_i at the hook
    n_dirs = E.shape[1] if args.max_dirs is None else args.max_dirs
    importance = np.zeros(E.shape[1])
    mu_dev = mu.float().to(args.device)
    with torch.no_grad():
        for i in range(n_dirs):
            e = E_f[:, i].to(args.device)

            def ablate(act, hook, e=e):
                coef = (act - mu_dev) @ e
                return act - coef.unsqueeze(-1) * e

            losses = [next_token_loss(model.run_with_hooks(b, fwd_hooks=[(hook, ablate)]), b).item() for b in batches]
            importance[i] = float(np.mean(losses)) - clean_loss
            if i % 64 == 0:
                print(f"dir {i}/{n_dirs}  lambda={lam[i]:.4f}  dloss={importance[i]:.5f}  ({time.time() - t0:.0f}s)", flush=True)

    # 4. per-direction reconstruction error for each trained SAE
    sae_err = {}
    for run_dir in sorted(glob.glob(str(ROOT / "results" / args.sae_glob))):
        sae = SAE.load_from_disk(Path(run_dir) / "sae", device="cpu")
        with torch.no_grad():
            err = sae(X) - X
        sae_err[Path(run_dir).name] = ((err @ E_f) ** 2).mean(0).numpy()

    np.savez(
        out / "directions.npz",
        lam=lam.numpy(),
        importance=importance,
        grad_importance=grad_importance,
        n_ablated=n_dirs,
        **{f"err__{k}": v for k, v in sae_err.items()},
    )
    (out / "meta.json").write_text(
        json.dumps(
            {**vars(args), "clean_loss": clean_loss, "n_tokens": int(X.shape[0]),
             "saes": list(sae_err), "seconds": round(time.time() - t0)},
            indent=2,
        )
    )
    print(f"done in {time.time() - t0:.0f}s, clean loss {clean_loss:.4f} -> {out}")


if __name__ == "__main__":
    main()
