#!/usr/bin/env python3
"""Predictive ALIS: learn a per-user interface selector from cheap, inference-time
features (no LLM call, no ground truth), and compare it against the retrospective
domain-level static selector and a per-user oracle.

Outputs: a per-domain results table (CSV + LaTeX fragment) with mean NDCG@20,
regret vs oracle, LLM-call rate (cost proxy), and selection accuracy.

All inputs are existing result artifacts; no new API calls are made.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "egpr_profile_repair"

FREE = {"Base", "Content", "Dense"}  # zero-API interfaces
PAID = {"Direct", "Raw", "Remove", "EGPR"}  # require an LLM call


# ----------------------------- IO helpers -----------------------------------
def read_per_user(path: Path) -> Dict[int, float]:
    """uid -> NDCG@20 from a per_user_*.csv file."""
    out: Dict[int, float] = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            out[int(r["uid"])] = float(r["NDCG@20"])
    return out


def read_targets(path: Path) -> List[Tuple[int, int]]:
    """Ordered (uid, target) list from a per_user_*.csv file."""
    rows: List[Tuple[int, int]] = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append((int(r["uid"]), int(r["target"])))
    return rows


def ndcg_single(top_row: np.ndarray, tgt: int) -> float:
    pos = np.where(top_row == tgt)[0]
    if len(pos) == 0:
        return 0.0
    return 1.0 / math.log2(int(pos[0]) + 2)


def content_per_user(top_npy: Path, order: List[Tuple[int, int]]) -> Dict[int, float]:
    top = np.load(top_npy)
    out: Dict[int, float] = {}
    for i, (uid, tgt) in enumerate(order):
        out[uid] = ndcg_single(top[i], tgt)
    return out


def history_lengths(train_csv: Path) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    with train_csv.open() as f:
        for r in csv.DictReader(f):
            uid = int(r["uid"])
            counts[uid] = counts.get(uid, 0) + 1
    return counts


def base_features(npz_path: Path) -> Dict[int, Dict[str, float]]:
    """Per-user base-confidence features from candidate scores."""
    z = np.load(npz_path, allow_pickle=True)
    users = z["users"].astype(int)
    scores = z["scores"].astype(float)
    cands = z["candidates"].astype(int)
    feats: Dict[int, Dict[str, float]] = {}
    base_top20: Dict[int, np.ndarray] = {}
    for i, uid in enumerate(users):
        s = scores[i]
        order = np.argsort(-s)
        ss = s[order]
        p = np.exp(ss - ss.max())
        p = p / (p.sum() + 1e-12)
        ent = float(-(p * np.log(p + 1e-12)).sum())
        margin = float(ss[0] - ss[1]) if len(ss) > 1 else 0.0
        feats[int(uid)] = {
            "base_top1": float(ss[0]),
            "base_margin": margin,
            "base_entropy": ent,
            "base_std": float(s.std()),
            "n_cand": float(len(s)),
        }
        base_top20[int(uid)] = cands[i][order][:20]
    return feats, base_top20


# ----------------------------- experiment -----------------------------------
def run_domain(name: str,
               iface_ndcg: Dict[str, Dict[int, float]],
               feats: Dict[int, np.ndarray],
               feat_names: List[str],
               seed: int = 42) -> Dict[str, object]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    interfaces = list(iface_ndcg.keys())
    uids = sorted(set.intersection(*[set(d.keys()) for d in iface_ndcg.values()]) & set(feats.keys()))
    uids = np.array(uids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uids))
    half = len(uids) // 2
    val_ids = uids[perm[:half]]
    test_ids = uids[perm[half:]]

    def mat(ids):
        return np.array([[iface_ndcg[m][u] for m in interfaces] for u in ids])

    Yv, Yt = mat(val_ids), mat(test_ids)
    Xv = np.array([feats[u] for u in val_ids])
    Xt = np.array([feats[u] for u in test_ids])

    # Per-user best interface labels (ties -> cheapest interface by index order
    # with FREE first so ties never spend API budget).
    order_pref = sorted(range(len(interfaces)),
                        key=lambda j: (interfaces[j] not in FREE, j))
    def best_label(row):
        m = row.max()
        for j in order_pref:
            if row[j] >= m - 1e-12:
                return j
        return int(row.argmax())
    yv = np.array([best_label(Yv[i]) for i in range(len(val_ids))])
    yt = np.array([best_label(Yt[i]) for i in range(len(test_ids))])

    # --- methods ---
    results: Dict[str, Dict[str, float]] = {}

    def summarize(choice_idx: np.ndarray, label: str):
        picked = Yt[np.arange(len(test_ids)), choice_idx]
        oracle = Yt.max(axis=1)
        regret = float((oracle - picked).mean())
        llm_rate = float(np.mean([interfaces[c] in PAID for c in choice_idx]))
        acc = float(np.mean(choice_idx == yt))
        results[label] = {
            "NDCG": float(picked.mean()),
            "Regret": regret,
            "LLMrate": llm_rate,
            "SelAcc": acc,
        }

    n_test = len(test_ids)
    # Base-only
    bidx = interfaces.index("Base")
    summarize(np.full(n_test, bidx), "Base")

    # Static-best (validation-selected single interface = domain-level ALIS, acc-first)
    static = int(Yv.mean(axis=0).argmax())
    summarize(np.full(n_test, static), "Static-best")

    # Predictive ALIS (classifier on cheap features)
    scaler = StandardScaler().fit(Xv)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    # Guard: need >=2 classes present in labels
    if len(np.unique(yv)) >= 2:
        clf.fit(scaler.transform(Xv), yv)
        pred = clf.predict(scaler.transform(Xt)).astype(int)
    else:
        pred = np.full(n_test, int(yv[0]))
    summarize(pred, "Predictive ALIS")

    # Cost-aware predictive: route to a PAID interface only when predicted-best is
    # PAID *and* the validation-estimated gain of that interface over the best FREE
    # interface exceeds a small margin; otherwise fall back to the best FREE one.
    free_idx = [interfaces.index(m) for m in interfaces if m in FREE]
    best_free_static = free_idx[int(Yv[:, free_idx].mean(axis=0).argmax())]
    # per-class validation gain of predicted interface vs best free
    val_gain = {j: float(Yv[:, j].mean() - Yv[:, best_free_static].mean())
                for j in range(len(interfaces))}
    margin = 0.0005
    ca = []
    for c in pred:
        if interfaces[c] in PAID and val_gain[c] > margin:
            ca.append(c)
        else:
            ca.append(best_free_static)
    summarize(np.array(ca), "Cost-aware Predictive")

    # Oracle upper bound
    summarize(yt, "Oracle")

    decisive = int(np.sum(Yt.max(axis=1) - Yt.min(axis=1) > 1e-12))

    # --- Heterogeneity gap H and held-out cost-utility frontier ---
    from sklearn.ensemble import GradientBoostingRegressor
    free_cols = [interfaces.index(m) for m in interfaces if m in FREE]
    paid_cols = [interfaces.index(m) for m in interfaces if m in PAID]
    # optimistic (in-sample) heterogeneity gap on test
    H_opt = float(Yt.max(axis=1).mean() - Yt.mean(axis=0).max())
    best_free_t = Yt[:, free_cols].max(axis=1)
    # paid uplift label (validation) for a regressor
    uplift_v = Yv[:, paid_cols].max(axis=1) - Yv[:, free_cols].max(axis=1)
    reg = GradientBoostingRegressor(random_state=seed)
    reg.fit(Xv, uplift_v)
    pred_uplift = reg.predict(Xt)
    # best paid interface predicted per test user (reuse clf if multiclass else static paid)
    static_paid = paid_cols[int(Yv[:, paid_cols].mean(axis=0).argmax())]
    bf_static = free_cols[int(Yv[:, free_cols].mean(axis=0).argmax())]

    def frontier(budget: float, smart: bool) -> float:
        k = int(round(budget * n_test))
        choice = np.full(n_test, bf_static)
        if k > 0:
            if smart:
                idx = np.argsort(-pred_uplift)[:k]
            else:
                idx = rng.choice(n_test, size=k, replace=False)
            choice[idx] = static_paid
        picked = Yt[np.arange(n_test), choice]
        return float(picked.mean())

    budgets = [0.0, 0.25, 0.5, 0.75, 1.0]
    sweep = {f"{b:.2f}": {"smart": frontier(b, True), "random": frontier(b, False)}
             for b in budgets}

    return {
        "H_opt": H_opt,
        "sweep": sweep,
        "name": name,
        "interfaces": interfaces,
        "n_val": len(val_ids),
        "n_test": n_test,
        "decisive_test": decisive,
        "static": interfaces[static],
        "results": results,
        "feat_names": feat_names,
    }


# ----------------------------- domain loaders --------------------------------
def load_ml1m(seed: int = 42) -> Dict[str, object]:
    d = RES / "ml1m_2000_fair_direct_candidates"
    order = read_targets(d / "per_user_lightgcn.csv")
    iface = {
        "Base": read_per_user(d / "per_user_lightgcn.csv"),
        "Direct": read_per_user(d / "per_user_direct.csv"),
        "Raw": read_per_user(d / "per_user_profile_rerank_raw.csv"),
        "Remove": read_per_user(d / "per_user_profile_rerank_remove.csv"),
        "EGPR": read_per_user(d / "per_user_profile_rerank_egpr.csv"),
        "Content": content_per_user(d / "reranking_baselines" / "ranked_content_top20.npy", order),
        "Dense": content_per_user(d / "reranking_baselines" / "ranked_dense_top20.npy", order),
    }
    hist = history_lengths(ROOT / "data" / "ml1m" / "train.csv")
    bf, base_top20 = base_features(RES / "ml1m_seed42_deepseek_2000_expressive5" / "candidates_lightgcn_test.npz")
    ctop = np.load(d / "reranking_baselines" / "ranked_content_top20.npy")
    content_top20 = {uid: ctop[i] for i, (uid, _t) in enumerate(order)}

    feat_names = ["log_hist", "base_top1", "base_margin", "base_entropy",
                  "base_std", "base_content_overlap"]
    feats: Dict[int, np.ndarray] = {}
    for uid in iface["Base"].keys():
        if uid not in bf or uid not in hist:
            continue
        ov = 0.0
        if uid in base_top20 and uid in content_top20:
            ov = len(set(base_top20[uid].tolist()) & set(content_top20[uid].tolist())) / 20.0
        b = bf[uid]
        feats[uid] = np.array([math.log1p(hist[uid]), b["base_top1"], b["base_margin"],
                               b["base_entropy"], b["base_std"], ov])
    return run_domain("ML-1M", iface, feats, feat_names, seed=seed)


def _yelp_content_per_user(npz_dir: Path, seed: int):
    """Tune the token-overlap Content reranker lambda on val, apply on test, and
    return {uid -> NDCG@20} plus {uid -> content top-20} aligned to the test npz."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "scripts"))
    from run_reranking_baselines_strong import (
        load_item_docs, build_history, sim_matrix, rerank, ndcg_recall, tokenize,
    )
    docs = load_item_docs(ROOT / "data" / "yelp_day1" / "item_text.jsonl", "iid", ["text"])
    item_tokens = {i: tokenize(d) for i, d in docs.items()}
    hist = build_history(ROOT / "data" / "yelp_day1" / "train.csv", 50)
    val = np.load(npz_dir / "candidates_lightgcn_val.npz")
    test = np.load(npz_dir / "candidates_lightgcn_test.npz")
    grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sv = sim_matrix("overlap", val["users"], val["candidates"].astype(np.int64),
                    hist, item_tokens, None, None)
    best_lam, best = grid[0], -1.0
    for lam in grid:
        rv = rerank(sv, val["scores"], val["candidates"].astype(np.int64), lam)
        nd, _ = ndcg_recall(rv, val["targets"].astype(np.int64), 20)
        if nd.mean() > best:
            best, best_lam = nd.mean(), lam
    st = sim_matrix("overlap", test["users"], test["candidates"].astype(np.int64),
                    hist, item_tokens, None, None)
    rt = rerank(st, test["scores"], test["candidates"].astype(np.int64), best_lam)
    nd, _ = ndcg_recall(rt, test["targets"].astype(np.int64), 20)
    users = test["users"].astype(int)
    content_nd = {int(users[i]): float(nd[i]) for i in range(len(users))}
    content_top = {int(users[i]): rt[i, :20] for i in range(len(users))}
    return content_nd, content_top, best_lam, float(nd.mean())


def load_yelp(seed: int = 42) -> Dict[str, object]:
    """Yelp confirmatory panel (1000 users, top-100). Interface set is
    {Base, Content, Raw, Remove, EGPR}: Direct is below Base on Yelp and Dense is
    unreliable under partial text coverage, so both are excluded (and noted)."""
    d = RES / "yelp_seed42_deepseek_1000_expressive5"
    z = np.load(d / "candidates_lightgcn_test.npz")
    users = z["users"].astype(int)

    def col(path):  # per_user CSV is row-aligned to the test npz (verified)
        vals = [float(r["NDCG@20"]) for r in csv.DictReader(path.open())]
        return {int(users[i]): vals[i] for i in range(len(users))}

    content_nd, content_top, lam_c, c_mean = _yelp_content_per_user(d, seed)
    iface = {
        "Base": col(d / "per_user_lightgcn.csv"),
        "Content": content_nd,
        "Raw": col(d / "per_user_lightgcn_raw_profile.csv"),
        "Remove": col(d / "per_user_lightgcn_remove_repair.csv"),
        "EGPR": col(d / "per_user_lightgcn_evidence_weighted_repair.csv"),
    }
    hist = history_lengths(ROOT / "data" / "yelp_day1" / "train.csv")
    bf, base_top20 = base_features(d / "candidates_lightgcn_test.npz")
    feat_names = ["log_hist", "base_top1", "base_margin", "base_entropy",
                  "base_std", "base_content_overlap"]
    feats: Dict[int, np.ndarray] = {}
    for uid in iface["Base"].keys():
        if uid not in bf or uid not in hist:
            continue
        ov = 0.0
        if uid in base_top20 and uid in content_top:
            ov = len(set(base_top20[uid].tolist()) & set(content_top[uid].tolist())) / 20.0
        b = bf[uid]
        feats[uid] = np.array([math.log1p(hist[uid]), b["base_top1"], b["base_margin"],
                               b["base_entropy"], b["base_std"], ov])
    rep = run_domain("Yelp", iface, feats, feat_names, seed=seed)
    rep["content_lambda"] = lam_c
    rep["content_mean"] = c_mean
    return rep


def fmt_table(reports: List[Dict[str, object]]) -> str:
    lines = []
    for rep in reports:
        res = rep["results"]
        lines.append(f"% {rep['name']}: interfaces={rep['interfaces']} "
                     f"n_val={rep['n_val']} n_test={rep['n_test']} "
                     f"decisive_test={rep['decisive_test']} static={rep['static']}")
        for k, v in res.items():
            lines.append(f"%   {k:<26} NDCG={v['NDCG']:.4f} Regret={v['Regret']:.4f} "
                         f"LLMrate={v['LLMrate']:.2f} SelAcc={v['SelAcc']:.3f}")
        lines.append(f"%   H_opt(optimistic heterogeneity gap)={rep['H_opt']:.4f}")
        lines.append("%   budget frontier (NDCG): " +
                     " ".join(f"B={b}:smart={d['smart']:.4f}/rand={d['random']:.4f}"
                              for b, d in rep["sweep"].items()))
    return "\n".join(lines)


def multiseed(loader, seeds=(0, 1, 2, 3, 4)) -> Dict[str, object]:
    reps = [loader(seed=s) for s in seeds]
    base = reps[0]
    agg_results = {}
    for k in base["results"]:
        agg_results[k] = {m: float(np.mean([r["results"][k][m] for r in reps]))
                          for m in base["results"][k]}
        agg_results[k]["NDCG_std"] = float(np.std([r["results"][k]["NDCG"] for r in reps]))
    sweep = {}
    for b in base["sweep"]:
        sweep[b] = {
            "smart": float(np.mean([r["sweep"][b]["smart"] for r in reps])),
            "smart_std": float(np.std([r["sweep"][b]["smart"] for r in reps])),
            "random": float(np.mean([r["sweep"][b]["random"] for r in reps])),
        }
    base["results"] = agg_results
    base["sweep"] = sweep
    base["H_opt"] = float(np.mean([r["H_opt"] for r in reps]))
    base["n_seeds"] = len(seeds)
    return base


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    reps = []
    if which in ("all", "ml1m"):
        reps.append(multiseed(lambda seed=42: load_ml1m(seed)))
    if which in ("all", "yelp"):
        ry = multiseed(lambda seed=42: load_yelp(seed))
        print(f"% Yelp content lambda_c={ry.get('content_lambda')} "
              f"content_mean_NDCG={ry.get('content_mean'):.4f}")
        reps.append(ry)
    print(fmt_table(reps))
