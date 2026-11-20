#!/usr/bin/env python3
"""Build data/<dataset>/item_text.jsonl, the item text used by the BM25 and Dense rerankers.

ML-1M: one row per item with its title and genres (from movies.dat).
Yelp:  business name and categories joined into one text field, for the items listed in
       configs/yelp_item_text_iids.txt (the items whose business record the Yelp profile run
       looked up; the rest of the catalogue has no item text in the experiments).

Run after the dataset has been prepared (data/ml1m from run_ml1m_minipilot.py,
data/yelp_day1 from run_yelp_day1_drift.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def build_ml1m(datadir: Path, movies: Path) -> int:
    item2id = json.loads((datadir / "mappings.json").read_text(encoding="utf-8"))["item2id"]
    meta = {}
    with movies.open(encoding="latin-1") as fh:
        for line in fh:
            movie_id, title, genres = line.rstrip("\n").split("::")
            meta[movie_id] = (title, genres.replace("|", " "))
    rows = sorted((int(iid), *meta[mid]) for mid, iid in item2id.items())
    with (datadir / "item_text.jsonl").open("w", encoding="utf-8", newline="\n") as out:
        for iid, title, genres in rows:
            out.write(json.dumps({"iid": iid, "title": title, "genres": genres}) + "\n")
    return len(rows)


def build_yelp(datadir: Path, reviews: Path, business: Path) -> int:
    from run_yelp_profile_rerank_pilot import (
        current_to_old_item_map,
        load_business_records,
        reconstruct_needed_old_to_business,
    )

    iids = [int(x) for x in (ROOT / "configs" / "yelp_item_text_iids.txt").read_text().split()]
    cur_to_old = current_to_old_item_map(datadir)
    old_to_bid = reconstruct_needed_old_to_business(reviews, sorted({cur_to_old[i] for i in iids}), 0)
    records = load_business_records(business, old_to_bid.values())
    n = 0
    with (datadir / "item_text.jsonl").open("w", encoding="utf-8", newline="\n") as out:
        for iid in iids:
            row = records.get(old_to_bid.get(cur_to_old[iid], ""))
            if row is None:
                continue
            name = str(row.get("name") or "").strip()
            categories = [c.strip() for c in str(row.get("categories") or "").split(",") if c.strip()]
            text = " ".join([name] + categories).strip()
            if text:
                out.write(json.dumps({"iid": iid, "text": text}) + "\n")
                n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=["ml1m", "yelp"])
    p.add_argument("--movies", type=Path, default=ROOT / "data/raw/ml-1m/movies.dat")
    p.add_argument("--reviews", type=Path, default=ROOT / "data/raw/yelp/yelp_academic_dataset_review.json")
    p.add_argument("--business", type=Path, default=ROOT / "data/raw/yelp/yelp_academic_dataset_business.json")
    args = p.parse_args()
    if args.dataset == "ml1m":
        n = build_ml1m(ROOT / "data/ml1m", args.movies)
    else:
        n = build_yelp(ROOT / "data/yelp_day1", args.reviews, args.business)
    print(f"wrote {n} items")


if __name__ == "__main__":
    main()
