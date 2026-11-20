#!/usr/bin/env python3
"""External reranking baselines for ML-1M over cached LightGCN candidates.

Currently implements the no-API, non-LLM `content` baseline (history token
centroid mixed with the normalized base score) so the ALIS interfaces are
compared against a strong, zero-cost reranker on ML-1M as well. The candidate
set, users, and targets are taken from the same cached npz used by the profile
pilots so every number is paired.
"""

from __future__ import annotations

import argparse
import json
import sys
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
from run_egpr_profile_repair_pilot import (  # noqa: E402
    build_ordered_histories,
    history_lines,
    item_text_tokens,
)
from run_llm_selective_invocation_pilot import ensure_dirs, read_item_metadata  # noqa: E402
from run_amazon_books_profile_rerank_pilot import (  # noqa: E402
    metrics_from_ranked,
    reliability,
    summary,
)
from run_amazon_books_direct_rerank_pilot import DirectReranker, markdown_table  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=Path, default=ROOT / "data/ml1m")
    parser.add_argument(
        "--movies-path",
        type=Path,
        default=Path(__file__).resolve().parents[1].joinpath("data/raw/ml-1m/movies.dat"),
    )
    parser.add_argument(
        "--candidate-run-dir",
        type=Path,
        default=ROOT / "results/egpr_profile_repair/ml1m_seed42_deepseek_2000_expressive5",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "results/egpr_profile_repair/ml1m_2000_fair_direct_candidates/reranking_baselines",
    )
    parser.add_argument("--methods", nargs="+", default=["content"], choices=["content", "llmrank"])
    parser.add_argument("--candidates", type=int, default=50, help="candidates per user to rerank")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--history-limit", type=int, default=20)
    parser.add_argument("--content-lambda", type=float, default=0.5)
    parser.add_argument("--limit-users", type=int, default=0)
    # LLM reranking baseline (provider robustness): shared with the Amazon engine.
    parser.add_argument("--provider", choices=["deepseek", "openai", "moonshot"], default="deepseek")
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--api-timeout", type=float, default=120.0)
    parser.add_argument("--api-max-retries", type=int, default=8)
    parser.add_argument("--api-max-output-tokens", type=int, default=4000)
    parser.add_argument("--input-price-per-1m", type=float, default=0.14)
    parser.add_argument("--output-price-per-1m", type=float, default=0.28)
    args = parser.parse_args()
    if args.model is None:
        args.model = {"deepseek": "deepseek-chat", "moonshot": "kimi-k2.5"}.get(args.provider, "gpt-4.1-mini")
    return args


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
    """Top-K overlap and mean absolute rank shift of a reranking vs the base order.

    The base order is the candidate column order (base-score ranked). ranked_full
    is the full candidate set in the method's new order.
    """
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


def ml1m_candidate_lines(candidates: Sequence[int], meta) -> List[str]:
    rows: List[str] = []
    for pos, iid_np in enumerate(candidates, start=1):
        iid = int(iid_np)
        genres = "|".join(meta.genres[iid]) if meta.genres[iid] else "Unknown"
        rows.append(f"{pos}. {meta.titles[iid]} | {genres}")
    return rows


def build_llmrank_prompt(uid: int, candidate_ids: Sequence[int], histories, meta, history_limit: int) -> str:
    lines = [
        "You are a recommender ranking agent.",
        "I will provide the user's movie interaction history (most recent last) and a numbered candidate list.",
        "Rank ALL candidates from most to least likely to be the user's next watch.",
        "Base your decision on the user's preferences inferred from the history, not on listing order or popularity.",
        "Return valid JSON only with schema {\"ranking\":[candidate_number, ...]}.",
        "Rules: use each candidate number exactly once; do not invent numbers.",
        "",
        f"User: U{uid}",
        "User history (oldest to newest):",
    ]
    lines.extend(history_lines(int(uid), histories, meta, history_limit) or ["None"])
    lines.extend(["", "Candidates:"])
    lines.extend(ml1m_candidate_lines(candidate_ids, meta))
    return "\n".join(lines)


def run_llmrank(
    reranker: DirectReranker,
    users: np.ndarray,
    candidates: np.ndarray,
    histories,
    meta,
    history_limit: int,
    topk: int,
) -> Tuple[np.ndarray, List[Dict[str, object]], int]:
    ranked = np.zeros((len(users), topk), dtype=np.int64)
    cost_rows: List[Dict[str, object]] = []
    parse_failures = 0
    for row, uid_np in enumerate(users):
        uid = int(uid_np)
        candidate_ids = [int(x) for x in candidates[row]]
        prompt = build_llmrank_prompt(uid, candidate_ids, histories, meta, history_limit)
        perm, cost, failed = reranker.rerank_one(prompt, len(candidate_ids))
        order = [candidate_ids[i] for i in perm]
        ranked[row] = np.asarray(order[:topk], dtype=np.int64)
        parse_failures += int(failed)
        cost_rows.append({"uid": uid, **cost, "parse_failure": int(failed)})
        if (row + 1) % 25 == 0 or row + 1 == len(users):
            print(f"llmrank rerank {row + 1}/{len(users)} users", flush=True)
    return ranked, cost_rows, parse_failures


def content_rerank(
    users: np.ndarray,
    candidates: np.ndarray,
    base_scores: np.ndarray,
    histories: Sequence[np.ndarray],
    item_tokens: List[set],
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
            hist_set |= item_tokens[int(iid_np)]
        content = np.zeros(n_cand, dtype=np.float64)
        for pos, iid_np in enumerate(candidates[row]):
            content[pos] = set_overlap(hist_set, item_tokens[int(iid_np)])
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
    meta = read_item_metadata(args.movies_path, args.datadir / "mappings.json", n_items)
    item_tokens: List[set] = [set(item_text_tokens(iid, meta)) for iid in range(n_items)]

    npz = np.load(args.candidate_run_dir / "candidates_lightgcn_test.npz")
    users = npz["users"].astype(np.int64)
    targets = npz["targets"].astype(np.int64)
    candidates = npz["candidates"][:, : args.candidates].astype(np.int64)
    base_scores = npz["scores"][:, : args.candidates].astype(np.float32)
    if args.limit_users and args.limit_users > 0:
        n = int(args.limit_users)
        users, targets, candidates, base_scores = users[:n], targets[:n], candidates[:n], base_scores[:n]

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
                users, candidates, base_scores, histories, item_tokens,
                args.history_limit, args.content_lambda, args.topk,
            )
            label = "Content Rerank"
        elif method == "llmrank":
            reranker = DirectReranker(args)
            ranked, cost_rows, parse_failures = run_llmrank(
                reranker, users, candidates, histories, meta, args.history_limit, args.topk,
            )
            ranked_full = candidates  # full-list disruption not tracked for LLM (top-k only)
            label = f"LLMRank ({args.provider}:{args.model})"
            cost_frame = pd.DataFrame(cost_rows)
            cost_frame.to_csv(args.outdir / "reranking_baselines_llmrank_cost.csv", index=False)
            in_tok = float(cost_frame.get("input_tokens", pd.Series(dtype=float)).sum())
            out_tok = float(cost_frame.get("output_tokens", pd.Series(dtype=float)).sum())
            print(
                f"LLMRank tokens in={in_tok:.0f} out={out_tok:.0f} parse_failures={parse_failures} "
                f"est_cost=${in_tok/1e6*args.input_price_per_1m + out_tok/1e6*args.output_price_per_1m:.4f}",
                flush=True,
            )
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
        "dataset": "ml1m",
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
        "# ML-1M External Reranking Baselines",
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
    (args.outdir / "ml1m_reranking_baselines_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Done. Report: {args.outdir / 'ml1m_reranking_baselines_report.md'}", flush=True)


if __name__ == "__main__":
    main()
