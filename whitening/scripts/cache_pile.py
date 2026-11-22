"""Cache the first N Pile train docs and M validation docs as local parquet, in stream order.

Training on the cache sees the same token stream as streaming monology/pile-uncopyrighted
from the start, but no longer depends on the network mid-run.

    uv run python whitening/scripts/cache_pile.py --train-docs 40000 --val-docs 4000
"""

import argparse
from pathlib import Path

from datasets import Dataset, load_dataset

DATA = Path(__file__).resolve().parents[2] / "data"


def cache(split_kwargs: dict, n: int, out: Path, shard: int = 10_000):
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("monology/pile-uncopyrighted", streaming=True, **split_kwargs)
    rows, part = [], 0
    for i, doc in enumerate(ds):
        if i >= n:
            break
        rows.append({"text": doc["text"]})
        if len(rows) == shard:
            Dataset.from_list(rows).to_parquet(out / f"train-{part:05d}.parquet")
            rows, part = [], part + 1
    if rows:
        Dataset.from_list(rows).to_parquet(out / f"train-{part:05d}.parquet")
    print(f"cached {min(i, n)} docs -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-docs", type=int, default=40_000)
    p.add_argument("--val-docs", type=int, default=4_000)
    args = p.parse_args()
    cache({"split": "train"}, args.train_docs, DATA / "pile_train")
    cache({"data_files": {"validation": "val.jsonl.zst"}, "split": "validation"}, args.val_docs, DATA / "pile_val")


if __name__ == "__main__":
    main()
