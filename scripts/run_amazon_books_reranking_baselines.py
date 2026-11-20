#!/usr/bin/env python3
"""External reranking baselines for Amazon Books over cached LightGCN candidates.

This script adds *external, published* reranking baselines so that the ALIS
interfaces (Direct / Raw Profile / Remove / EGPR) are no longer compared only
against each other. All baselines operate on the SAME cached LightGCN candidate
sets, the SAME users and targets, and reuse the SAME metric helpers as the rest
of the pipeline, so every number is paired and directly comparable.

Baselines implemented
---------------------
1. content   -- No-API, non-LLM strong baseline. Builds a user-history token
                centroid and reranks candidates by token-overlap similarity
                mixed with the (normalized) base score. Requires NO API key and
                NO GPU, so it can be run anywhere to provide an immediate
                non-trivial comparison point.
2. llmrank   -- Hou et al. (2024) "LLMs are Zero-Shot Rankers for Recommender
                Systems": a single listwise pass with an instruction prompt that
                explicitly asks the model to use the interaction history and to
                rank ALL candidates. Reuses the project DeepSeek/OpenAI plumbing.
3. rankgpt   -- Sun et al. (2023) RankGPT: sliding-window listwise permutation
                generation. The candidate list is reranked window by window from
                the tail to the head so the whole list is covered with bounded
                prompt length.

Outputs mirror run_amazon_books_direct_rerank_pilot.py so the results plug into
the existing aggregation / ALIS panel without changes:
  - reranking_baselines_summary.csv
  - reranking_baselines_reliability.csv
  - reranking_baselines_cost_trace.csv
  - ranked_<method>_top20.npy
  - run_manifest.json
  - amazon_books_reranking_baselines_report.md
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

from temporal_popularity.data import read_interaction_split  # noqa: E402
from run_amazon_books_profile_rerank_pilot import (  # noqa: E402
    BookMeta,
    history_lines,
    load_metadata,
    metrics_from_ranked,
    reliability,
    summary,
    token_overlap_score,
)
from run_amazon_books_direct_rerank_pilot import (  # noqa: E402
    DirectReranker,
    candidate_lines,
    load_test_batch,
    markdown_table,
)
from run_egpr_profile_repair_pilot import build_ordered_histories  # noqa: E402
from run_llm_selective_invocation_pilot import ensure_dirs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=Path, default=ROOT / "data/amazon_books_subset")
    parser.add_argument(
        "--candidate-run-dir",
        type=Path,
        default=ROOT / "results/egpr_profile_repair/amazon_books_seed42_lightgcn_1000",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "results/egpr_profile_repair/amazon_books_direct_vs_profile/reranking_baselines_1000_top100",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["content", "llmrank", "rankgpt"],
        choices=["content", "llmrank", "rankgpt"],
    )
    parser.add_argument("--provider", choices=["deepseek", "openai", "moonshot"], default="deepseek")
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--api-timeout", type=float, default=90.0)
    parser.add_argument("--api-max-retries", type=int, default=12)
    parser.add_argument("--api-max-output-tokens", type=int, default=900)
    parser.add_argument("--history-limit", type=int, default=20)
    parser.add_argument("--candidates", type=int, default=100, help="candidates per user to rerank")
    parser.add_argument("--limit-users", type=int, default=0, help="if >0, only process the first N users (for cheap pilots)")
    parser.add_argument("--topk", type=int, default=20)
    # content baseline mixing weight (selected on val elsewhere; fixed here for transparency)
    parser.add_argument("--content-lambda", type=float, default=0.5)
    # RankGPT sliding-window parameters
    parser.add_argument("--rankgpt-window", type=int, default=20)
    parser.add_argument("--rankgpt-step", type=int, default=10)
    parser.add_argument("--input-price-per-1m", type=float, default=None)
    parser.add_argument("--output-price-per-1m", type=float, default=None)
    args = parser.parse_args()
    if args.model is None:
        args.model = {
            "deepseek": "deepseek-v4-flash",
            "moonshot": "kimi-k2.5",
        }.get(args.provider, "gpt-4.1-mini")
    if args.input_price_per_1m is None:
        args.input_price_per_1m = {"deepseek": 0.14, "moonshot": 0.60}.get(args.provider, 0.40)
    if args.output_price_per_1m is None:
        args.output_price_per_1m = {"deepseek": 0.28, "moonshot": 2.50}.get(args.provider, 1.60)
    return args


# ---------------------------------------------------------------------------
# Baseline 1: content-based reranker (no API, no GPU)
# ---------------------------------------------------------------------------
def minmax(values: np.ndarray) -> np.ndarray:
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


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


def overlap_only(candidates: np.ndarray, ranked_topk: np.ndarray, topk: int) -> Dict[str, float]:
    overlaps = [
        len(set(ranked_topk[row, :topk].tolist()) & set(candidates[row, :topk].tolist())) / float(topk)
        for row in range(candidates.shape[0])
    ]
    return {"Top20OverlapVsBase": float(np.mean(overlaps)) if overlaps else 1.0, "MeanAbsRankShiftVsBase": float("nan")}


def content_rerank(
    users: np.ndarray,
    targets: np.ndarray,
    candidates: np.ndarray,
    base_scores: np.ndarray,
    histories: Sequence[np.ndarray],
    meta: BookMeta,
    history_limit: int,
    lam: float,
    topk: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rerank by mixing normalized base score with history-centroid token overlap."""
    n_cand = candidates.shape[1]
    ranked = np.zeros((len(users), topk), dtype=np.int64)
    ranked_full = np.zeros((len(users), n_cand), dtype=np.int64)
    for row, uid_np in enumerate(users):
        uid = int(uid_np)
        hist_tokens: Dict[str, float] = {}
        for iid_np in histories[uid][-history_limit:]:
            for tok in meta.tokens[int(iid_np)]:
                hist_tokens[tok] = hist_tokens.get(tok, 0.0) + 1.0
        hist_token_tuple = tuple(hist_tokens.keys())
        content = np.zeros(n_cand, dtype=np.float32)
        for pos, iid_np in enumerate(candidates[row]):
            iid = int(iid_np)
            # similarity between the candidate tokens and the user history tokens
            content[pos] = token_overlap_score(" ".join(hist_token_tuple), meta.tokens[iid])
        base_norm = minmax(base_scores[row].astype(np.float64)) if base_scores is not None else np.zeros(n_cand)
        content_norm = minmax(content.astype(np.float64))
        final = (1.0 - lam) * base_norm + lam * content_norm
        # stable tie-break on the candidate id to keep determinism
        order = np.lexsort((candidates[row], -final))
        ranked_full[row] = candidates[row, order]
        ranked[row] = ranked_full[row, :topk]
    return ranked, ranked_full


# ---------------------------------------------------------------------------
# Baseline 2: LLMRank (Hou et al. 2024) single listwise pass
# ---------------------------------------------------------------------------
def build_llmrank_prompt(uid: int, candidates: Sequence[int], histories, meta: BookMeta, history_limit: int) -> str:
    lines = [
        "You are a recommender ranking agent.",
        "I will provide the user's interaction history (most recent last) and a numbered candidate list.",
        "Rank ALL candidates from most to least likely to be the user's next interaction.",
        "Base your decision on the user's preferences inferred from the history, not on the listing order or popularity.",
        "Return valid JSON only with schema {\"ranking\":[candidate_number, ...]}.",
        "Rules: use each candidate number exactly once; do not invent numbers.",
        "",
        f"User: U{uid}",
        "User history (oldest to newest):",
    ]
    lines.extend(history_lines(int(uid), histories, meta, history_limit) or ["None"])
    lines.extend(["", "Candidates:"])
    lines.extend(candidate_lines(candidates, meta))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Baseline 3: RankGPT (Sun et al. 2023) sliding-window listwise permutation
# ---------------------------------------------------------------------------
def build_rankgpt_window_prompt(
    uid: int,
    window_ids: Sequence[int],
    histories,
    meta: BookMeta,
    history_limit: int,
) -> str:
    lines = [
        "You are RankGPT, an intelligent assistant that ranks recommendation candidates by how well they match a user.",
        "I will provide the user's interaction history and a numbered list of candidate items.",
        "Rank the candidates by descending match to the user's preferences.",
        "Return valid JSON only with schema {\"ranking\":[candidate_number, ...]} using every number exactly once.",
        "",
        f"User: U{uid}",
        "User history (oldest to newest):",
    ]
    lines.extend(history_lines(int(uid), histories, meta, history_limit) or ["None"])
    lines.extend(["", "Candidates:"])
    lines.extend(candidate_lines(window_ids, meta))
    return "\n".join(lines)


def rankgpt_sliding_window(
    reranker: DirectReranker,
    uid: int,
    candidate_ids: Sequence[int],
    histories,
    meta: BookMeta,
    history_limit: int,
    window: int,
    step: int,
) -> Tuple[List[int], Dict[str, float], bool]:
    """Apply the RankGPT sliding-window permutation from tail to head.

    Returns the globally reordered candidate id list, accumulated cost, and a
    parse-failure flag (True if any window failed to parse).
    """
    order = list(int(x) for x in candidate_ids)
    n = len(order)
    total_cost = {"input_tokens": 0.0, "output_tokens": 0.0, "latency_seconds": 0.0}
    any_fail = False
    start = max(n - window, 0)
    while True:
        end = min(start + window, n)
        window_slice = order[start:end]
        prompt = build_rankgpt_window_prompt(uid, window_slice, histories, meta, history_limit)
        perm, cost, failed = reranker.rerank_one(prompt, len(window_slice))
        any_fail = any_fail or failed
        for key in total_cost:
            total_cost[key] += cost.get(key, 0.0)
        order[start:end] = [window_slice[i] for i in perm]
        if start == 0:
            break
        start = max(start - step, 0)
    return order, total_cost, any_fail


def run_llm_listwise(
    reranker: DirectReranker,
    method: str,
    users: np.ndarray,
    candidates: np.ndarray,
    histories,
    meta: BookMeta,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, List[Dict[str, object]], int]:
    ranked = np.zeros((len(users), args.topk), dtype=np.int64)
    cost_rows: List[Dict[str, object]] = []
    parse_failures = 0
    for row, uid_np in enumerate(users):
        uid = int(uid_np)
        candidate_ids = [int(x) for x in candidates[row]]
        if method == "llmrank":
            prompt = build_llmrank_prompt(uid, candidate_ids, histories, meta, args.history_limit)
            perm, cost, failed = reranker.rerank_one(prompt, len(candidate_ids))
            order = [candidate_ids[i] for i in perm]
        elif method == "rankgpt":
            order, cost, failed = rankgpt_sliding_window(
                reranker, uid, candidate_ids, histories, meta,
                args.history_limit, args.rankgpt_window, args.rankgpt_step,
            )
        else:
            raise ValueError(method)
        ranked[row] = np.asarray(order[: args.topk], dtype=np.int64)
        parse_failures += int(failed)
        cost_rows.append({"uid": uid, "method": method, **cost, "parse_failure": int(failed)})
        if (row + 1) % 50 == 0 or row + 1 == len(users):
            print(f"{method} rerank {row + 1}/{len(users)} users", flush=True)
    return ranked, cost_rows, parse_failures


def main() -> None:
    args = parse_args()
    ensure_dirs(args.outdir)
    train, _, _, _ = read_interaction_split(args.datadir)
    n_users = int(train["uid"].max()) + 1
    histories = build_ordered_histories(train, n_users)
    meta = load_metadata(args.datadir)

    # Reuse the same loader as the Direct pilot so candidate/user/target sets match.
    users, targets, candidates = load_test_batch(args.candidate_run_dir, args.candidates)
    # base scores are needed by the content baseline; reload them from the npz.
    npz = np.load(args.candidate_run_dir / "candidates_lightgcn_test.npz")
    base_scores = npz["scores"][:, : args.candidates].astype(np.float32)

    if args.limit_users and args.limit_users > 0:
        n = int(args.limit_users)
        users, targets, candidates = users[:n], targets[:n], candidates[:n]
        base_scores = base_scores[:n]

    base_ranked = candidates[:, : args.topk]
    base_metrics = metrics_from_ranked(base_ranked, targets, args.topk)

    summary_rows = [{"Method": "LightGCN", **summary(base_metrics)}]
    reliability_rows = [{
        "Method": "LightGCN", "HarmRate": 0.0, "PositiveGainRate": 0.0,
        "MeanDeltaNDCG@20": 0.0, "PositiveGainSum": 0.0, "NegativeGainSum": 0.0,
        "GainHarmRatio": np.nan,
    }]
    all_cost_rows: List[Dict[str, object]] = []
    manifest_methods: Dict[str, object] = {}
    disruption_rows = [{"Method": "LightGCN", "Top20OverlapVsBase": 1.0, "MeanAbsRankShiftVsBase": 0.0}]

    reranker: DirectReranker | None = None
    for method in args.methods:
        if method == "content":
            ranked, ranked_full = content_rerank(
                users, targets, candidates, base_scores, histories, meta,
                args.history_limit, args.content_lambda, args.topk,
            )
            method_label = "Content Rerank"
            cost_rows: List[Dict[str, object]] = []
            parse_failures = 0
            disruption = disruption_vs_base(candidates, ranked_full, args.topk)
        else:
            if reranker is None:
                reranker = DirectReranker(args)
            ranked, cost_rows, parse_failures = run_llm_listwise(
                reranker, method, users, candidates, histories, meta, args,
            )
            method_label = "RankGPT" if method == "rankgpt" else "LLMRank"
            disruption = overlap_only(candidates, ranked, args.topk)

        method_metrics = metrics_from_ranked(ranked, targets, args.topk)
        summary_rows.append({"Method": method_label, **summary(method_metrics)})
        reliability_rows.append({"Method": method_label, **reliability(base_metrics, method_metrics)})
        disruption_rows.append({"Method": method_label, **disruption})
        np.save(args.outdir / f"ranked_{method}_top20.npy", ranked)
        all_cost_rows.extend(cost_rows)

        cost_usd = 0.0
        if cost_rows:
            cost_df = pd.DataFrame(cost_rows)
            cost_usd = float(
                (cost_df["input_tokens"].sum() / 1_000_000.0) * args.input_price_per_1m
                + (cost_df["output_tokens"].sum() / 1_000_000.0) * args.output_price_per_1m
            )
        manifest_methods[method] = {
            "label": method_label,
            "estimated_cost_usd": cost_usd,
            "parse_failures": int(parse_failures),
        }

    summary_frame = pd.DataFrame(summary_rows)
    reliability_frame = pd.DataFrame(reliability_rows)
    disruption_frame = pd.DataFrame(disruption_rows)
    summary_frame.to_csv(args.outdir / "reranking_baselines_summary.csv", index=False)
    reliability_frame.to_csv(args.outdir / "reranking_baselines_reliability.csv", index=False)
    disruption_frame.to_csv(args.outdir / "reranking_baselines_disruption.csv", index=False)
    if all_cost_rows:
        pd.DataFrame(all_cost_rows).to_csv(args.outdir / "reranking_baselines_cost_trace.csv", index=False)

    manifest = {
        "status": "completed",
        "dataset": "amazon_books_subset",
        "provider": args.provider,
        "model": args.model,
        "candidate_run_dir": str(args.candidate_run_dir),
        "users": int(len(users)),
        "candidates": args.candidates,
        "topk": args.topk,
        "history_limit": args.history_limit,
        "methods": manifest_methods,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Amazon Books External Reranking Baselines",
        "",
        f"Dataset: Amazon Books subset. Candidate source: `{args.candidate_run_dir}`.",
        f"Candidates: top-{args.candidates}. Output: top-{args.topk}. History limit: {args.history_limit}.",
        f"Methods: {', '.join(args.methods)}.",
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
    (args.outdir / "amazon_books_reranking_baselines_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Done. Report: {args.outdir / 'amazon_books_reranking_baselines_report.md'}", flush=True)


if __name__ == "__main__":
    main()
