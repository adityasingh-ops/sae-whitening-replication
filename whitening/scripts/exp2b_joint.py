"""Exp 2b: do low-variance directions matter collectively, and does that explain SAE CE loss?

A. Joint vs individual ablation: mean-ablate the k lowest-variance principal directions
   together, and compare the loss increase with the sum of their single-direction
   effects from exp2 (same tokens, same basis).
B. Subspace repair: splice each SAE's reconstruction in at the hook, then put the true
   activation back only inside one subspace (the low-variance or the top-variance
   directions). How much of the SAE's loss increase does each repair remove?

    uv run python whitening/scripts/exp2b_joint.py
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sae_lens import SAE
from transformer_lens.model_bridge import TransformerBridge

sys.path.insert(0, str(Path(__file__).parent))
from exp2_directions import load_tokens, next_token_loss  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--n-seqs", type=int, default=64)
    p.add_argument("--ctx", type=int, default=128)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--sae-glob", default="pythia-70m-deduped_L3_topk_k32_x8_*_10M_s*")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--exp2", default=str(ROOT / "results" / "exp2"))
    args = p.parse_args()

    out = Path(args.exp2)
    hook = f"blocks.{args.layer}.hook_resid_post"
    t0 = time.time()
    model = TransformerBridge.boot_transformers(args.model, device=args.device)
    model.eval()
    tokens = load_tokens(model.tokenizer, args.n_seqs, args.ctx).to(args.device)
    batches = tokens.split(args.batch)

    def mean_loss(fwd_hooks) -> float:
        with torch.no_grad():
            return float(np.mean([next_token_loss(model.run_with_hooks(b, fwd_hooks=fwd_hooks), b).item() for b in batches]))

    # same activations and principal basis as exp2
    acts = []
    def keep(act, hook):
        acts.append(act.detach().reshape(-1, act.shape[-1]).float().cpu())
        return act
    clean = mean_loss([(hook, keep)])
    X = torch.cat(acts)
    mu = X.double().mean(0)
    lam, E = torch.linalg.eigh(torch.cov((X.double() - mu).T))
    order = torch.argsort(lam, descending=True)
    lam, E = lam[order].numpy(), E[:, order].float()
    single = np.load(out / "directions.npz")["importance"]
    mu_d = mu.float().to(args.device)

    def ablate_subspace(idx):
        B = E[:, idx].to(args.device)
        def fn(act, hook):
            return act - ((act - mu_d) @ B) @ B.T
        return fn

    # A. joint vs individual
    joint = []
    for k in [16, 64, 128, 256, 384, 448]:
        idx = list(range(512 - k, 512))
        dl = mean_loss([(hook, ablate_subspace(idx))]) - clean
        joint.append({"bottom_k": k, "variance_share": float(lam[idx].sum() / lam.sum()),
                      "joint_dloss": dl, "sum_of_single_dloss": float(single[idx].sum())})
        print(f"bottom {k:3d}: joint {dl:.4f} vs sum of singles {single[idx].sum():.4f}", flush=True)
    for k in [1, 4, 16, 64]:
        idx = list(range(k))
        dl = mean_loss([(hook, ablate_subspace(idx))]) - clean
        joint.append({"top_k": k, "variance_share": float(lam[idx].sum() / lam.sum()),
                      "joint_dloss": dl, "sum_of_single_dloss": float(single[idx].sum())})
        print(f"top {k:3d}: joint {dl:.4f} vs sum of singles {single[idx].sum():.4f}", flush=True)

    # B. subspace repair of each SAE's reconstruction
    repair = {}
    subspaces = {"none": [], "low_384": list(range(128, 512)), "top_128": list(range(128))}
    for run_dir in sorted(glob.glob(str(ROOT / "results" / args.sae_glob))):
        sae = SAE.load_from_disk(Path(run_dir) / "sae", device=args.device)
        name = Path(run_dir).name
        repair[name] = {}
        for label, idx in subspaces.items():
            B = E[:, idx].to(args.device) if idx else None
            def splice(act, hook, B=B):
                with torch.no_grad():
                    recon = sae(act.float()).to(act.dtype)
                if B is None:
                    return recon
                return recon + ((act - recon) @ B) @ B.T
            repair[name][label] = mean_loss([(hook, splice)]) - clean
        r = repair[name]
        print(f"{name}: sae dloss {r['none']:.4f} | repair low-var {r['low_384']:.4f} | repair top {r['top_128']:.4f}", flush=True)

    (out / "joint.json").write_text(json.dumps({"clean_loss": clean, "joint": joint, "repair": repair,
                                                "seconds": round(time.time() - t0)}, indent=2))
    print(f"done in {time.time() - t0:.0f}s -> {out / 'joint.json'}")


if __name__ == "__main__":
    main()
