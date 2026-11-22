"""Plots and summary numbers for exp2_directions.py output.

H2: do some low-variance directions matter a lot to the model?
H3: do plain SAEs leave their error in those directions, and whitened SAEs not?

    uv run python whitening/scripts/analyze_exp2.py whitening/results/exp2
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

PLAIN, WHITE = "#2a78d6", "#eb6834"  # categorical slots 1-2, validated light-mode pair
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10,
})


def main(exp_dir: Path):
    d = np.load(exp_dir / "directions.npz")
    lam, imp = d["lam"], d["importance"]
    n = int(d["n_ablated"])
    lam, imp = lam[:n], imp[:n]
    results = exp_dir.parents[0]

    # per-SAE fraction of each direction's variance left unreconstructed
    frac = {"plain": [], "white": []}
    ce_delta, ev, damage = {}, {}, {}
    for key in d.files:
        if not key.startswith("err__"):
            continue
        name = key[len("err__"):]
        f = d[key][:n] / lam
        frac["white" if "_white_" in name else "plain"].append(f)
        ev_json = json.loads((results / name / "eval.json").read_text())
        ce_delta[name] = ev_json["ce_loss_with_sae"] - ev_json["ce_loss_without_sae"]
        ev[name] = ev_json["explained_variance"]
        damage[name] = float(np.sum(np.clip(imp, 0, None) * np.clip(f, 0, 1)))

    rank = np.arange(1, n + 1)
    bins = np.unique(np.geomspace(1, n + 1, 14).astype(int))

    # ---- Figure A: variance vs importance
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    pos = imp > 0
    ax.scatter(lam[pos], imp[pos], s=14, color=PLAIN, alpha=0.75, edgecolors=SURFACE, linewidths=0.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("variance of direction (log)")
    ax.set_ylabel("loss increase when removed (log)")
    rho = spearmanr(lam, imp).correlation
    ax.set_title(f"Variance vs importance, {n} principal directions (Spearman ρ = {rho:.2f})", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(exp_dir / "fig_a_variance_vs_importance.png", dpi=160); plt.close(fig)

    # ---- Figure B: unreconstructed fraction by variance rank
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for kind, color, label in [("plain", PLAIN, "plain"), ("white", WHITE, "whitened")]:
        arr = np.array(frac[kind])
        centers, mid, lo, hi = [], [], [], []
        for a, b in zip(bins[:-1], bins[1:]):
            sel = (rank >= a) & (rank < b)
            per_seed = arr[:, sel].mean(1)
            centers.append(np.sqrt(a * (b - 1)) if b - 1 > a else a)
            mid.append(per_seed.mean()); lo.append(per_seed.min()); hi.append(per_seed.max())
        ax.fill_between(centers, lo, hi, color=color, alpha=0.18, linewidth=0)
        ax.plot(centers, mid, color=color, linewidth=2, marker="o", markersize=5, label=label)
        ax.annotate(label, (centers[-1], mid[-1]), xytext=(6, 0), textcoords="offset points", color=INK, va="center", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("direction rank by variance (1 = largest, log)")
    ax.set_ylabel("fraction of direction NOT reconstructed")
    ax.set_title("Where each SAE's reconstruction error lives (mean, band = seed range)", fontsize=10, loc="left")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(exp_dir / "fig_b_error_by_rank.png", dpi=160); plt.close(fig)

    # ---- Figure C: importance-weighted damage by variance group
    groups = [("top 16", rank <= 16), ("17–128", (rank > 16) & (rank <= 128)), ("129–512", rank > 128)]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    width = 0.36
    for j, (kind, color, label) in enumerate([("plain", PLAIN, "plain"), ("white", WHITE, "whitened")]):
        arr = np.array(frac[kind])
        vals = [np.mean([np.sum(np.clip(imp[s], 0, None) * np.clip(f[s], 0, 1)) for f in arr]) for _, s in groups]
        xs = np.arange(len(groups)) + (j - 0.5) * width
        ax.bar(xs, vals, width=width - 0.03, color=color, label=label)
    ax.set_xticks(np.arange(len(groups)), [g for g, _ in groups])
    ax.set_xlabel("directions grouped by variance rank")
    ax.set_ylabel("Σ importance × fraction lost")
    ax.set_title("Estimated damage to the model, by where the error is", fontsize=10, loc="left")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(exp_dir / "fig_c_damage_by_group.png", dpi=160); plt.close(fig)

    # ---- summary numbers
    order_low = rank > n // 4
    names = sorted(ce_delta)
    summary = {
        "n_directions": n,
        "spearman_variance_vs_importance": round(float(rho), 3),
        "share_of_variance_in_bottom_75pct_dirs": round(float(lam[order_low].sum() / lam.sum()), 4),
        "share_of_importance_in_bottom_75pct_dirs": round(float(np.clip(imp, 0, None)[order_low].sum() / np.clip(imp, 0, None).sum()), 4),
        "n_dirs_with_nonpositive_importance": int((imp <= 0).sum()),
        "per_sae": {k: {"ce_delta": round(ce_delta[k], 4), "explained_variance": round(ev[k], 4), "importance_weighted_damage": round(damage[k], 4)} for k in names},
        "spearman_ce_delta_vs_explained_variance": round(float(spearmanr([ce_delta[k] for k in names], [ev[k] for k in names]).correlation), 3),
        "spearman_ce_delta_vs_weighted_damage": round(float(spearmanr([ce_delta[k] for k in names], [damage[k] for k in names]).correlation), 3),
    }
    (exp_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
