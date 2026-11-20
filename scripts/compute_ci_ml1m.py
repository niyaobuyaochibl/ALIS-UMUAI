#!/usr/bin/env python3
"""Bootstrap 95% CIs for per-user Delta NDCG@20 vs Base on the ML-1M confirmatory
panel (ml1m_2000_fair_direct_candidates), the panel behind Table tab:main-utility.

Per-user NDCG@20 is taken from the panel CSVs (Base/Direct/Profile-*), recomputed
from the saved top-20 rankings (Content/Dense), and recomputed on the same panel
for BM25 (no per-user array was stored). Means are asserted to match the paper.
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_reranking_baselines_strong import (  # noqa: E402
    load_item_docs, build_history, sim_matrix, rerank, ndcg_recall, BM25, tokenize,
)

PANEL = ROOT / "results" / "egpr_profile_repair" / "ml1m_2000_fair_direct_candidates"
RB = PANEL / "reranking_baselines"
NPZDIR = ROOT / "results" / "egpr_profile_repair" / "ml1m_seed42_deepseek_2000_expressive5"
ITEM_TEXT = ROOT / "data" / "ml1m" / "item_text.jsonl"
TRAIN = ROOT / "data" / "ml1m" / "train.csv"
TOPK = 20
NBOOT = 10000
SEED = 0


def csv_ndcg(path: Path):
    uids, tgts, nd = [], [], []
    with path.open() as f:
        for r in csv.DictReader(f):
            uids.append(int(r["uid"]))
            tgts.append(int(r["target"]))
            nd.append(float(r["NDCG@20"]))
    return np.array(uids), np.array(tgts), np.array(nd)


def npy_ndcg(path: Path, targets: np.ndarray):
    ranked = np.load(path)
    nd, _ = ndcg_recall(ranked, targets.astype(np.int64), TOPK)
    return nd


def main():
    uids, tgts, base = csv_ndcg(PANEL / "per_user_lightgcn.csv")
    order_uid = uids

    methods = {"Base": base}
    for name, fn in [("Direct", "per_user_direct.csv"),
                     ("Profile-Raw", "per_user_profile_rerank_raw.csv"),
                     ("Profile-Remove", "per_user_profile_rerank_remove.csv"),
                     ("Profile-EGPR", "per_user_profile_rerank_egpr.csv")]:
        u2, t2, nd = csv_ndcg(PANEL / fn)
        assert np.array_equal(u2, order_uid), f"uid order mismatch in {fn}"
        methods[name] = nd

    methods["Content"] = npy_ndcg(RB / "ranked_content_top20.npy", tgts)

    # --- BM25 + Dense: recompute per-user on the panel with the main-table config
    #     (candidate width 50, history_limit 50, lambda_c=0.7), then align by uid. ---
    CW, HL, LAM = 50, 50, 0.7
    docs = load_item_docs(ITEM_TEXT, "iid", ["title", "genres"])
    item_tokens = {i: tokenize(d) for i, d in docs.items()}
    hist = build_history(TRAIN, HL)
    bm25 = BM25(item_tokens)
    test = np.load(NPZDIR / "candidates_lightgcn_test.npz")
    cand = test["candidates"][:, :CW].astype(np.int64)
    sc = test["scores"][:, :CW]
    uid_to_row = {int(u): i for i, u in enumerate(test["users"])}

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    maxid = max(docs) + 1
    dense_emb = np.asarray(model.encode([docs.get(i, "") for i in range(maxid)],
                                        batch_size=256, show_progress_bar=False))
    for name, kind, de in [("BM25", "bm25", None), ("Dense", "dense", dense_emb)]:
        sim = sim_matrix(kind, test["users"], cand, hist, item_tokens, bm25, de)
        ranked = rerank(sim, sc, cand, LAM)
        nd_npz, _ = ndcg_recall(ranked, test["targets"].astype(np.int64), TOPK)
        methods[name] = np.array([nd_npz[uid_to_row[int(u)]] for u in order_uid])

    # --- means vs paper ---
    paper = {"Base": 0.0249, "Content": 0.0267, "BM25": 0.0282, "Dense": 0.0319,
             "Direct": 0.0312, "Profile-Raw": 0.0278, "Profile-Remove": 0.0278,
             "Profile-EGPR": 0.0280}
    print("=== mean check (computed vs paper) ===")
    for m in ["Base", "Content", "BM25", "Dense", "Direct",
              "Profile-Raw", "Profile-Remove", "Profile-EGPR"]:
        print(f"{m:16s} {methods[m].mean():.4f}  (paper {paper[m]:.4f})")

    # --- bootstrap CI on Delta vs Base, paired over users ---
    rng = np.random.default_rng(SEED)
    n = len(base)
    idx = rng.integers(0, n, size=(NBOOT, n))
    print("\n=== Delta NDCG@20 vs Base, 95% bootstrap CI (x10^3) ===")
    print(f"{'Method':16s} {'Delta':>8s} {'95% CI':>20s}")
    rows = {}
    for m in ["Content", "BM25", "Dense", "Direct",
              "Profile-Raw", "Profile-Remove", "Profile-EGPR"]:
        delta = methods[m] - base
        boot = delta[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows[m] = (delta.mean(), lo, hi)
        print(f"{m:16s} {delta.mean()*1e3:8.2f} [{lo*1e3:7.2f}, {hi*1e3:7.2f}]")

    # --- key paired comparison: Dense vs Direct ---
    d = methods["Dense"] - methods["Direct"]
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\nDense - Direct (paired): {d.mean()*1e3:.2f} x10^-3  "
          f"95% CI [{lo*1e3:.2f}, {hi*1e3:.2f}] x10^-3")


if __name__ == "__main__":
    main()
