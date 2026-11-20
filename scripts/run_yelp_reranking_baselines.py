#!/usr/bin/env python3
"""External reranking baselines for Yelp over cached LightGCN candidates.

Implements the no-API, non-LLM `content` baseline on Yelp. Yelp item metadata
is reconstructed from the raw Yelp dataset: the 5.3GB review file is streamed
*directly from the dataset zip* (never extracted to disk, to respect the small
disk budget) to recover the old business ordering, and business names/categories
are read from the (small) business file. The recovered old->business map is
cached so re-runs are instant.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPT_DIR))

from temporal_popularity.data import infer_shape, read_interaction_split  # noqa: E402
from run_llm_selective_invocation_pilot import ensure_dirs  # noqa: E402
from run_yelp_profile_rerank_pilot import (  # noqa: E402
    BUSINESS_RE,
    YelpMetadata,
    build_ordered_histories,
    business_text,
    category_tokens,
    current_to_old_item_map,
    load_business_records,
    simple_tokens,
)
from run_amazon_books_profile_rerank_pilot import (  # noqa: E402
    metrics_from_ranked,
    reliability,
    summary,
)
from run_amazon_books_direct_rerank_pilot import markdown_table  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=Path, default=ROOT / "data/yelp_day1")
    parser.add_argument("--zip-path", type=Path, default=Path(__file__).resolve().parents[1].joinpath("data/raw/yelp/yelp-dataset.zip"))
    parser.add_argument("--review-inner", default="yelp_academic_dataset_review.json")
    parser.add_argument("--business-path", type=Path, default=Path("/tmp/yelp_business.json"))
    parser.add_argument(
        "--candidate-run-dir",
        type=Path,
        default=ROOT / "results/egpr_profile_repair/yelp_seed42_deepseek_1000_expressive5",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "results/egpr_profile_repair/yelp_direct_vs_profile/reranking_baselines",
    )
    parser.add_argument("--methods", nargs="+", default=["content"], choices=["content"])
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--history-limit", type=int, default=20)
    parser.add_argument("--content-lambda", type=float, default=0.5)
    parser.add_argument("--progress-every", type=int, default=2_000_000)
    parser.add_argument("--limit-users", type=int, default=0)
    return parser.parse_args()


def minmax(values: np.ndarray) -> np.ndarray:
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def set_overlap(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / float((len(a) * len(b)) ** 0.5)


def disruption_vs_base(candidates: np.ndarray, ranked_full: np.ndarray, topk: int) -> Dict[str, float]:
    overlaps: List[float] = []
    shifts: List[float] = []
    for row in range(candidates.shape[0]):
        base = candidates[row]
        new = ranked_full[row]
        base_pos = {int(iid): pos for pos, iid in enumerate(base)}
        overlaps.append(len(set(new[:topk].tolist()) & set(base[:topk].tolist())) / float(topk))
        diffs = [abs(new_pos - base_pos[int(iid)]) for new_pos, iid in enumerate(new)]
        shifts.append(float(np.mean(diffs)) if diffs else 0.0)
    return {
        "Top20OverlapVsBase": float(np.mean(overlaps)) if overlaps else 1.0,
        "MeanAbsRankShiftVsBase": float(np.mean(shifts)) if shifts else 0.0,
    }


def reconstruct_old_to_business_streaming(
    zip_path: Path,
    inner_name: str,
    needed_old: Sequence[int],
    progress_every: int,
) -> Dict[int, str]:
    """Stream the review file inside the zip to recover old_id -> business_id.

    old_id is the index of the business in order of first appearance in the
    review file, matching current_to_old_item_map's convention.
    """
    needed = set(int(x) for x in needed_old)
    found: Dict[int, str] = {}
    business_to_old: Dict[bytes, int] = {}
    lines = 0
    with zipfile.ZipFile(zip_path) as zf, zf.open(inner_name, "r") as handle:
        for line in handle:
            match = BUSINESS_RE.search(line)
            if match:
                business = match.group(1)
            else:
                try:
                    business = str(json.loads(line)["business_id"]).encode("utf-8")
                except Exception:
                    continue
            old_id = business_to_old.get(business)
            if old_id is None:
                old_id = len(business_to_old)
                business_to_old[business] = old_id
                if old_id in needed:
                    found[old_id] = business.decode("utf-8")
                    if len(found) == len(needed):
                        break
            lines += 1
            if progress_every and lines % progress_every == 0:
                print(
                    f"review scan lines={lines:,} businesses={len(business_to_old):,} "
                    f"found={len(found):,}/{len(needed):,}",
                    flush=True,
                )
    missing = needed.difference(found)
    if missing:
        print(f"warning: {len(missing)} needed old ids not found in review scan", flush=True)
    return found


def build_yelp_token_sets(
    args: argparse.Namespace,
    needed_iids: Sequence[int],
    n_items: int,
) -> List[set]:
    iid_to_old = current_to_old_item_map(args.datadir)
    needed_iids_set = sorted(set(int(iid) for iid in needed_iids))
    needed_old = [iid_to_old[iid] for iid in needed_iids_set if iid in iid_to_old]

    cache_path = args.outdir / "old_to_business_cache.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        old_to_business = {int(k): v for k, v in cached.items()}
        print(f"loaded cached old->business map ({len(old_to_business)} entries)", flush=True)
        if not set(needed_old).issubset(old_to_business.keys()):
            print("cache incomplete for needed ids; rescanning review file", flush=True)
            old_to_business = reconstruct_old_to_business_streaming(
                args.zip_path, args.review_inner, needed_old, args.progress_every
            )
            cache_path.write_text(json.dumps(old_to_business), encoding="utf-8")
    else:
        old_to_business = reconstruct_old_to_business_streaming(
            args.zip_path, args.review_inner, needed_old, args.progress_every
        )
        cache_path.write_text(json.dumps(old_to_business), encoding="utf-8")

    business_records = load_business_records(args.business_path, old_to_business.values())

    token_sets: List[set] = [set(simple_tokens(f"Business {iid}")) for iid in range(n_items)]
    filled = 0
    for iid in needed_iids_set:
        old_id = iid_to_old.get(iid)
        business_id = old_to_business.get(old_id) if old_id is not None else None
        row = business_records.get(business_id) if business_id is not None else None
        if row is None:
            continue
        _, cats, _, _, text = business_text(row)
        token_sets[iid] = set(simple_tokens(text)) | set(category_tokens(cats))
        filled += 1
    print(f"token sets filled for {filled:,}/{len(needed_iids_set):,} needed items", flush=True)
    return token_sets


def content_rerank(
    users: np.ndarray,
    candidates: np.ndarray,
    base_scores: np.ndarray,
    histories: Sequence[np.ndarray],
    token_sets: List[set],
    history_limit: int,
    lam: float,
    topk: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_cand = candidates.shape[1]
    ranked = np.zeros((len(users), topk), dtype=np.int64)
    ranked_full = np.zeros((len(users), n_cand), dtype=np.int64)
    for row, uid_np in enumerate(users):
        uid = int(uid_np)
        hist_set: set = set()
        for iid_np in histories[uid][-history_limit:]:
            hist_set |= token_sets[int(iid_np)]
        content = np.zeros(n_cand, dtype=np.float64)
        for pos, iid_np in enumerate(candidates[row]):
            content[pos] = set_overlap(hist_set, token_sets[int(iid_np)])
        base_norm = minmax(base_scores[row].astype(np.float64))
        content_norm = minmax(content)
        final = (1.0 - lam) * base_norm + lam * content_norm
        order = np.lexsort((candidates[row], -final))
        ranked_full[row] = candidates[row, order]
        ranked[row] = ranked_full[row, :topk]
    return ranked, ranked_full


def main() -> None:
    args = parse_args()
    ensure_dirs(args.outdir)
    train, val, test, _ = read_interaction_split(args.datadir)
    n_users, n_items = infer_shape(train, val, test)
    histories = build_ordered_histories(train, n_users)

    npz = np.load(args.candidate_run_dir / "candidates_lightgcn_test.npz")
    users = npz["users"].astype(np.int64)
    targets = npz["targets"].astype(np.int64)
    candidates = npz["candidates"][:, : args.candidates].astype(np.int64)
    base_scores = npz["scores"][:, : args.candidates].astype(np.float32)
    if args.limit_users and args.limit_users > 0:
        n = int(args.limit_users)
        users, targets, candidates, base_scores = users[:n], targets[:n], candidates[:n], base_scores[:n]

    needed_iids: set = set(int(x) for x in candidates.reshape(-1).tolist())
    for uid_np in users:
        for iid_np in histories[int(uid_np)][-args.history_limit:]:
            needed_iids.add(int(iid_np))
    token_sets = build_yelp_token_sets(args, sorted(needed_iids), n_items)

    base_ranked = candidates[:, : args.topk]
    base_metrics = metrics_from_ranked(base_ranked, targets, args.topk)
    summary_rows = [{"Method": "LightGCN", **summary(base_metrics)}]
    reliability_rows = [{
        "Method": "LightGCN", "HarmRate": 0.0, "PositiveGainRate": 0.0,
        "MeanDeltaNDCG@20": 0.0, "PositiveGainSum": 0.0, "NegativeGainSum": 0.0,
        "GainHarmRatio": np.nan,
    }]

    disruption_rows = [{"Method": "LightGCN", "Top20OverlapVsBase": 1.0, "MeanAbsRankShiftVsBase": 0.0}]
    for method in args.methods:
        if method == "content":
            ranked, ranked_full = content_rerank(
                users, candidates, base_scores, histories, token_sets,
                args.history_limit, args.content_lambda, args.topk,
            )
            label = "Content Rerank"
        else:
            raise ValueError(method)
        method_metrics = metrics_from_ranked(ranked, targets, args.topk)
        summary_rows.append({"Method": label, **summary(method_metrics)})
        reliability_rows.append({"Method": label, **reliability(base_metrics, method_metrics)})
        disruption_rows.append({"Method": label, **disruption_vs_base(candidates, ranked_full, args.topk)})
        np.save(args.outdir / f"ranked_{method}_top20.npy", ranked)

    summary_frame = pd.DataFrame(summary_rows)
    reliability_frame = pd.DataFrame(reliability_rows)
    disruption_frame = pd.DataFrame(disruption_rows)
    summary_frame.to_csv(args.outdir / "reranking_baselines_summary.csv", index=False)
    reliability_frame.to_csv(args.outdir / "reranking_baselines_reliability.csv", index=False)
    disruption_frame.to_csv(args.outdir / "reranking_baselines_disruption.csv", index=False)

    manifest = {
        "status": "completed",
        "dataset": "yelp_day1",
        "candidate_run_dir": str(args.candidate_run_dir),
        "users": int(len(users)),
        "candidates": args.candidates,
        "topk": args.topk,
        "history_limit": args.history_limit,
        "content_lambda": args.content_lambda,
        "methods": args.methods,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Yelp External Reranking Baselines",
        "",
        f"Candidate source: `{args.candidate_run_dir}`. Candidates: top-{args.candidates}. Output: top-{args.topk}.",
        f"Users: {len(users)}. History limit: {args.history_limit}. Content lambda: {args.content_lambda}.",
        "",
        "## Performance",
        "",
        markdown_table(summary_frame),
        "",
        "## Reliability",
        "",
        markdown_table(reliability_frame),
        "",
    ]
    (args.outdir / "yelp_reranking_baselines_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Done. Report: {args.outdir / 'yelp_reranking_baselines_report.md'}", flush=True)


if __name__ == "__main__":
    main()
