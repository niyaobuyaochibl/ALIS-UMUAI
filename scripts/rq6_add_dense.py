#!/usr/bin/env python3
"""Compute per-user Dense-reranker rankings on the ML-1M per-user panel so that
the free Sentence-BERT dense reranker can be added to the RQ6 per-user analysis.

It reranks the SAME candidate pool used by the other per-user interfaces
(candidates_lightgcn_{val,test}.npz), tunes lambda on val (max NDCG@20), and
writes ranked_dense_top20.npy aligned to the panel order in
per_user_lightgcn.csv (identical convention to ranked_content_top20.npy).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_reranking_baselines_strong import (  # noqa: E402
    load_item_docs, build_history, sim_matrix, rerank, ndcg_recall, minmax,
)

PANEL = ROOT / "results" / "egpr_profile_repair" / "ml1m_2000_fair_direct_candidates"
NPZDIR = ROOT / "results" / "egpr_profile_repair" / "ml1m_seed42_deepseek_2000_expressive5"
ITEM_TEXT = ROOT / "data" / "ml1m" / "item_text.jsonl"
TRAIN = ROOT / "data" / "ml1m" / "train.csv"
GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
TOPK = 20


def main() -> None:
    order = []
    with (PANEL / "per_user_lightgcn.csv").open() as f:
        for r in csv.DictReader(f):
            order.append((int(r["uid"]), int(r["target"])))

    docs = load_item_docs(ITEM_TEXT, "iid", ["title", "genres"])
    hist = build_history(TRAIN, 50)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    max_id = max(docs) + 1
    texts = [docs.get(i, "") for i in range(max_id)]
    dense_emb = np.asarray(model.encode(texts, batch_size=256, show_progress_bar=False))

    val = np.load(NPZDIR / "candidates_lightgcn_val.npz")
    test = np.load(NPZDIR / "candidates_lightgcn_test.npz")

    sim_val = sim_matrix("dense", val["users"], val["candidates"].astype(np.int64), hist,
                         {}, None, dense_emb)
    best_lam, best = GRID[0], -1.0
    for lam in GRID:
        rv = rerank(sim_val, val["scores"], val["candidates"].astype(np.int64), lam)
        nd, _ = ndcg_recall(rv, val["targets"].astype(np.int64), TOPK)
        if nd.mean() > best:
            best, best_lam = nd.mean(), lam
    print(f"tuned lambda_c={best_lam} (val NDCG@20={best:.4f})")

    sim_test = sim_matrix("dense", test["users"], test["candidates"].astype(np.int64), hist,
                          {}, None, dense_emb)
    r_test = rerank(sim_test, test["scores"], test["candidates"].astype(np.int64), best_lam)
    nd_test, _ = ndcg_recall(r_test, test["targets"].astype(np.int64), TOPK)
    print(f"test Dense NDCG@20={nd_test.mean():.4f} over {len(nd_test)} users")

    # row index by uid in npz, then emit ranked top-20 aligned to panel order
    uid_to_row = {int(u): i for i, u in enumerate(test["users"])}
    out = np.zeros((len(order), TOPK), dtype=np.int64)
    for i, (uid, _tgt) in enumerate(order):
        row = uid_to_row[uid]
        out[i] = r_test[row, :TOPK]
    np.save(PANEL / "reranking_baselines" / "ranked_dense_top20.npy", out)
    print(f"wrote ranked_dense_top20.npy ({out.shape}) to {PANEL/'reranking_baselines'}")


if __name__ == "__main__":
    main()
