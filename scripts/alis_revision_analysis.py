#!/usr/bin/env python3
"""R1 revision analyses on the per-user panels from export_revision_panels.py.
No LLM calls; numpy / pandas / scikit-learn only.

  oos        ALIS selected on one half of the users and evaluated on the disjoint
             other half (repeated random splits): held-out utility, harm, cost,
             and regret relative to the held-out-optimal interface.
  bootstrap  ALIS selection frequencies under user resampling, with and without
             confidence-aware ties.
  paired     Paired bootstrap CIs and sign-flip p-values for contested pairs.
  routing    Per-user routing with repeated nested cross-validation and several
             regularized routers, plus a replication of the original RQ6 protocol.

Usage:
  python scripts/alis_revision_analysis.py --panel-dir results/revision_r1/panels
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

FREE = {"Base", "Content", "BM25", "Dense"}
ORDER = ["Base", "Content", "BM25", "Dense", "Direct", "Raw", "Remove", "EGPR"]

PREFERENCES: Dict[str, Dict[str, float]] = {
    "accuracy_first": {"u": 1.00, "c": 0.02, "h": 0.02, "d": 0.01, "g": 0.01},
    "balanced": {"u": 1.00, "c": 0.15, "h": 0.15, "d": 0.10, "g": 0.10},
    "safety_first": {"u": 0.70, "c": 0.10, "h": 0.35, "d": 0.15, "g": 0.35},
    "cost_first": {"u": 0.60, "c": 0.45, "h": 0.10, "d": 0.10, "g": 0.05},
}

PROFILE_COST = {"ml1m": 0.44609124551503865, "yelp": 0.4691203250060023, "books": 0.5652084448694785}

WUCR = {
    "ml1m": {"Raw": 0.1226730310262529, "Remove": 0.0, "EGPR": 0.0086915026155217},
    "yelp": {"Raw": 0.08941294530858003, "Remove": 0.0, "EGPR": 0.011044725933980688},
    "books": {"Raw": 0.3479951397326853, "Remove": 0.0, "EGPR": 0.12787291027678319},
}

OVERLAP_FALLBACK = {
    "ml1m": {"Base": 1.0, "Content": 0.73655, "BM25": 0.636, "Dense": 0.643, "Direct": 0.5152,
             "Raw": 0.699425, "Remove": 0.69945, "EGPR": 0.69875},
    "yelp": {"Base": 1.0, "Content": 0.61105, "Direct": 0.6053, "Raw": 0.6277, "Remove": 0.6249,
             "EGPR": 0.62295},
    "books": {"Base": 1.0, "Content": 0.65955, "BM25": 0.359, "Dense": 0.314, "Direct": 0.65565,
              "Raw": 0.81865, "Remove": 0.91045, "EGPR": 0.8178},
}

PAIRS = [("Dense", "Direct"), ("Raw", "Content"), ("Content", "Remove"), ("Content", "Raw"),
         ("EGPR", "Raw"), ("Remove", "Raw")]

class Panel:
    """Per-user reward matrix Y (n x M), overlap matrix OV, features X."""

    def __init__(self, domain: str, df: pd.DataFrame):
        self.domain = domain
        present = [c[len("ndcg__"):] for c in df.columns if c.startswith("ndcg__")]
        self.ifaces = [m for m in ORDER if m in present] + sorted(set(present) - set(ORDER))
        Y = df[[f"ndcg__{m}" for m in self.ifaces]].to_numpy(float)
        keep = ~np.isnan(Y).any(axis=1)
        self.n_dropped = int((~keep).sum())
        df = df.loc[keep].reset_index(drop=True)
        self.uids = df["uid"].to_numpy()
        self.Y = Y[keep]
        self.n, self.M = self.Y.shape
        self.base = self.ifaces.index("Base")
        self.paid = np.array([m not in FREE for m in self.ifaces])

        fb = OVERLAP_FALLBACK.get(domain, {})
        self.overlap_source: Dict[str, str] = {}
        OV = np.zeros_like(self.Y)
        for j, m in enumerate(self.ifaces):
            col = f"overlap__{m}"
            if m == "Base":
                OV[:, j] = 1.0
                self.overlap_source[m] = "per-user"
            elif col in df and df[col].notna().mean() > 0.95:
                v = df[col].to_numpy(float)
                OV[:, j] = np.where(np.isnan(v), np.nanmean(v), v)
                self.overlap_source[m] = "per-user"
            else:
                OV[:, j] = fb.get(m, np.nan)
                self.overlap_source[m] = "constant (manuscript table)"
        self.OV = OV
        self.C = np.array([0.0 if m in FREE else (1.0 if m == "Direct" else PROFILE_COST[domain])
                           for m in self.ifaces])
        self.G = np.array([WUCR[domain].get(m, 0.0) for m in self.ifaces])
        feats = [c for c in df.columns if c.startswith("feat__")]
        self.feat_names = [c[len("feat__"):] for c in feats]
        self.X = df[feats].to_numpy(float) if feats else np.zeros((self.n, 0))
        if feats:
            self.X = np.where(np.isnan(self.X), np.nanmean(self.X, axis=0), self.X)

    def j(self, name: str) -> int:
        return self.ifaces.index(name)


def alis_scores(P: Panel, idx: np.ndarray, w: Dict[str, float], g_direct: float = 0.0):
    Y = P.Y[idx]
    U = Y.mean(axis=0)
    H = (Y < Y[:, [P.base]] - 1e-12).mean(axis=0)
    D = 1.0 - P.OV[idx].mean(axis=0)
    G = P.G.copy()
    if "Direct" in P.ifaces:
        G[P.j("Direct")] = g_direct
    Un = (U - U.min()) / max(U.max() - U.min(), 1e-12)
    Hn = H / max(H.max(), 1e-12)
    Gn = G / max(G.max(), 1e-12)
    penalty = w["c"] * P.C + w["h"] * Hn + w["d"] * D + w["g"] * Gn
    return w["u"] * Un - penalty, Un, penalty, U, H


def select_plain(P: Panel, idx: np.ndarray, w: Dict[str, float]) -> int:
    return int(np.argmax(alis_scores(P, idx, w)[0]))


def select_conf(P: Panel, idx: np.ndarray, w: Dict[str, float], z: float = 1.96) -> int:
    """Utility differences inside a paired 95% CI are treated as ties; ties are
    broken by the non-utility penalties (cost, harm, disruption, grounding)."""
    S, Un, penalty, _, _ = alis_scores(P, idx, w)
    m1 = int(np.argmax(S))
    d = P.Y[idx] - P.Y[idx][:, [m1]]
    se = d.std(axis=0, ddof=1) / np.sqrt(len(idx))
    tied = np.abs(d.mean(axis=0)) <= z * se
    tied[m1] = True
    cand = np.flatnonzero(tied)
    return int(cand[np.lexsort((-S[cand], penalty[cand]))[0]])


SELECTORS: Dict[str, Callable] = {"plain": select_plain, "confidence_aware": select_conf}


def run_oos(P: Panel, reps: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for r in range(reps):
        perm = rng.permutation(P.n)
        sel_idx, ev_idx = perm[: P.n // 2], perm[P.n // 2:]
        for pref, w in PREFERENCES.items():
            S_ev, _, _, U_ev, H_ev = alis_scores(P, ev_idx, w)
            m_star = int(np.argmax(S_ev))
            for sel_name, sel in SELECTORS.items():
                m = sel(P, sel_idx, w)
                rows.append({
                    "domain": P.domain, "rep": r, "preference": pref, "selector": sel_name,
                    "selected": P.ifaces[m], "heldout_optimal": P.ifaces[m_star],
                    "selected_is_llm": bool(P.paid[m]), "agree": m == m_star,
                    "heldout_ndcg": U_ev[m], "heldout_harm": H_ev[m], "cost": P.C[m],
                    "ndcg_regret": U_ev.max() - U_ev[m], "score_regret": S_ev[m_star] - S_ev[m],
                })
    return pd.DataFrame(rows)


def summarize_choices(df: pd.DataFrame, keys: List[str], col: str = "selected") -> pd.DataFrame:
    freq = (df.groupby(keys)[col].value_counts(normalize=True).rename("freq").reset_index()
            .pivot_table(index=keys, columns=col, values="freq", fill_value=0.0))
    freq.columns = [f"freq_{c}" for c in freq.columns]
    return freq.reset_index()


def summarize_oos(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["domain", "preference", "selector"]
    agg = df.groupby(keys).agg(
        heldout_ndcg=("heldout_ndcg", "mean"), heldout_harm=("heldout_harm", "mean"),
        cost=("cost", "mean"), llm_rate=("selected_is_llm", "mean"),
        ndcg_regret=("ndcg_regret", "mean"), ndcg_regret_sd=("ndcg_regret", "std"),
        score_regret=("score_regret", "mean"), agree_rate=("agree", "mean"),
    ).reset_index()
    return agg.merge(summarize_choices(df, keys), on=keys)


def run_bootstrap(P: Panel, B: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    full = np.arange(P.n)
    for pref, w in PREFERENCES.items():
        for sel_name, sel in SELECTORS.items():
            rows.append({"domain": P.domain, "b": -1, "preference": pref, "selector": sel_name,
                         "selected": P.ifaces[sel(P, full, w)]})
    for b in range(B):
        idx = rng.integers(0, P.n, P.n)
        for pref, w in PREFERENCES.items():
            for sel_name, sel in SELECTORS.items():
                rows.append({"domain": P.domain, "b": b, "preference": pref, "selector": sel_name,
                             "selected": P.ifaces[sel(P, idx, w)]})
    df = pd.DataFrame(rows)
    keys = ["domain", "preference", "selector"]
    full_sel = df[df.b == -1].set_index(keys)["selected"].rename("full_panel_choice")
    return summarize_choices(df[df.b >= 0], keys).merge(full_sel.reset_index(), on=keys)


def sign_flip_p(d: np.ndarray, rng: np.random.Generator, n_perm: int, chunk: int = 1000) -> float:
    nz = d[np.abs(d) > 1e-12]
    obs = abs(d.mean())
    if obs <= 0 or len(nz) == 0:
        return 1.0
    hits = 0
    for start in range(0, n_perm, chunk):
        k = min(chunk, n_perm - start)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(k, len(nz)))
        hits += int(np.sum(np.abs(signs @ nz) / len(d) >= obs - 1e-15))
    return (hits + 1) / (n_perm + 1)


def boot_ci(d: np.ndarray, rng: np.random.Generator, B: int, chunk: int = 1000):
    means = []
    for start in range(0, B, chunk):
        k = min(chunk, B - start)
        means.append(d[rng.integers(0, len(d), size=(k, len(d)))].mean(axis=1))
    lo, hi = np.percentile(np.concatenate(means), [2.5, 97.5])
    return float(lo), float(hi)


def run_paired(P: Panel, B: int, n_perm: int, rng: np.random.Generator) -> pd.DataFrame:
    pairs = [(m, "Base") for m in P.ifaces if m != "Base"]
    pairs += [(a, b) for a, b in PAIRS if a in P.ifaces and b in P.ifaces]
    rows = []
    for a, b in pairs:
        d = P.Y[:, P.j(a)] - P.Y[:, P.j(b)]
        lo, hi = boot_ci(d, rng, B)
        rows.append({"domain": P.domain, "left": a, "right": b, "users": P.n,
                     "mean_delta": d.mean(), "ci_low": lo, "ci_high": hi,
                     "signflip_p": sign_flip_p(d, rng, n_perm),
                     "wins": int((d > 1e-12).sum()), "losses": int((d < -1e-12).sum())})
    return pd.DataFrame(rows)


def run_grounding_sensitivity(P: Panel) -> pd.DataFrame:
    """Selection when Direct is assigned a non-zero grounding risk."""
    if "Direct" not in P.ifaces:
        return pd.DataFrame()
    profile_g = [P.G[P.j(m)] for m in ("Raw", "Remove", "EGPR") if m in P.ifaces]
    levels = {"zero": 0.0, "mean_profile_wucr": float(np.mean(profile_g)) if profile_g else 0.0,
              "max_profile_wucr": float(np.max(profile_g)) if profile_g else 0.0}
    full = np.arange(P.n)
    rows = []
    for lvl, g in levels.items():
        for pref, w in PREFERENCES.items():
            S = alis_scores(P, full, w, g_direct=g)[0]
            rows.append({"domain": P.domain, "direct_grounding": lvl, "value": g, "preference": pref,
                         "selected": P.ifaces[int(np.argmax(S))],
                         "direct_score": S[P.j("Direct")], "best_score": S.max()})
    return pd.DataFrame(rows)


ROUTER_GRIDS: Dict[str, list] = {
    "fixed": [None],
    "logit_label": [0.01, 0.1, 1.0, 10.0],
    "logit_l1": [0.01, 0.1, 1.0],
    "ridge_reward": [1.0, 10.0, 100.0, 1000.0],
    "gbm_reward": [50, 150],
}


def best_labels(Y: np.ndarray, ifaces: List[str]) -> np.ndarray:
    """Per-user best interface; ties go to free interfaces first."""
    order = sorted(range(len(ifaces)), key=lambda j: (ifaces[j] not in FREE, j))
    is_best = (Y >= Y.max(axis=1, keepdims=True) - 1e-12)[:, order]
    return np.array(order)[is_best.argmax(axis=1)]


def fit_predict(kind: str, hp, Xtr, Ytr, Xte, ifaces: List[str]) -> np.ndarray:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "fixed" or Xtr.shape[1] == 0:
        return np.full(len(Xte), int(Ytr.mean(axis=0).argmax()))
    if kind in ("logit_label", "logit_l1"):
        y = best_labels(Ytr, ifaces)
        classes = np.unique(y)
        if len(classes) < 2:
            return np.full(len(Xte), int(classes[0]))
        if kind == "logit_label":
            clf = LogisticRegression(C=hp, class_weight="balanced", max_iter=2000)
        else:
            clf = LogisticRegression(C=hp, penalty="l1", solver="saga", class_weight="balanced", max_iter=5000)
        return make_pipeline(StandardScaler(), clf).fit(Xtr, y).predict(Xte).astype(int)
    if kind == "ridge_reward":
        return make_pipeline(StandardScaler(), Ridge(alpha=hp)).fit(Xtr, Ytr).predict(Xte).argmax(axis=1)
    if kind == "gbm_reward":
        preds = [GradientBoostingRegressor(n_estimators=hp, max_depth=2, learning_rate=0.05, subsample=0.8,
                                           random_state=0).fit(Xtr, Ytr[:, j]).predict(Xte)
                 for j in range(Ytr.shape[1])]
        return np.column_stack(preds).argmax(axis=1)
    raise ValueError(kind)


def tune(kind: str, X, Y, ifaces, inner: int, rng: np.random.Generator):
    grid = ROUTER_GRIDS[kind]
    if len(grid) == 1:
        return grid[0]
    folds = np.array_split(rng.permutation(len(X)), inner)
    best_hp, best_v = grid[0], -np.inf
    for hp in grid:
        v = 0.0
        for k in range(inner):
            te = folds[k]
            tr = np.concatenate([folds[i] for i in range(inner) if i != k])
            ch = fit_predict(kind, hp, X[tr], Y[tr], X[te], ifaces)
            v += Y[te][np.arange(len(te)), ch].sum()
        if v > best_v:
            best_hp, best_v = hp, v
    return best_hp


def run_routing(P: Panel, cols: List[int], reps: int, folds: int, inner: int,
                boot: int, rng: np.random.Generator):
    ifaces = [P.ifaces[c] for c in cols]
    Y, X, paid = P.Y[:, cols], P.X, P.paid[cols]
    kinds = list(ROUTER_GRIDS)
    reward = {k: np.zeros((reps, P.n)) for k in kinds}
    is_paid = {k: np.zeros((reps, P.n)) for k in kinds}
    fixed_rows, hp_rows = [], []
    for r in range(reps):
        outer = np.array_split(rng.permutation(P.n), folds)
        for f in range(folds):
            te = outer[f]
            tr = np.concatenate([outer[i] for i in range(folds) if i != f])
            for kind in kinds:
                hp = tune(kind, X[tr], Y[tr], ifaces, inner, rng)
                ch = fit_predict(kind, hp, X[tr], Y[tr], X[te], ifaces)
                reward[kind][r, te] = Y[te][np.arange(len(te)), ch]
                is_paid[kind][r, te] = paid[ch]
                hp_rows.append({"domain": P.domain, "rep": r, "fold": f, "router": kind, "hp": hp})
                if kind == "fixed":
                    fixed_rows.append({"domain": P.domain, "rep": r, "fold": f, "selected": ifaces[int(ch[0])]})
    rows = []
    v_fix_insample = Y.mean(axis=0).max()
    for kind in kinds:
        delta_user = (reward[kind] - reward["fixed"]).mean(axis=0)
        lo, hi = boot_ci(delta_user, rng, boot) if kind != "fixed" else (0.0, 0.0)
        rows.append({"domain": P.domain, "policy": kind, "ndcg": reward[kind].mean(),
                     "ndcg_sd_over_reps": reward[kind].mean(axis=1).std(ddof=1) if reps > 1 else 0.0,
                     "delta_vs_fixed": delta_user.mean(), "delta_ci_low": lo, "delta_ci_high": hi,
                     "paid_call_rate": is_paid[kind].mean()})
    oracle = Y.max(axis=1)
    rows.append({"domain": P.domain, "policy": "oracle_insample", "ndcg": oracle.mean(),
                 "delta_vs_fixed": oracle.mean() - v_fix_insample,
                 "paid_call_rate": paid[best_labels(Y, ifaces)].mean()})
    fixed_df = pd.DataFrame(fixed_rows)
    fixed_freq = fixed_df["selected"].value_counts(normalize=True).rename("freq").reset_index()
    fixed_freq.insert(0, "domain", P.domain)
    return pd.DataFrame(rows), fixed_freq, pd.DataFrame(hp_rows)


def run_paper_protocol(P: Panel, cols: List[int], uids: np.ndarray) -> pd.DataFrame:
    """Original RQ6 protocol: five random validation/test halves (seeds 0-4),
    validation-selected fixed interface vs. multinomial logistic selector."""
    ifaces = [P.ifaces[c] for c in cols]
    Y, paid = P.Y[:, cols], P.paid[cols]
    order = np.argsort(uids)
    rows = []
    for seed in range(5):
        perm = order[np.random.default_rng(seed).permutation(P.n)]
        half = P.n // 2
        va, te = perm[:half], perm[half:]
        fixed = int(Y[va].mean(axis=0).argmax())
        ch = fit_predict("logit_label", 1.0, P.X[va], Y[va], P.X[te], ifaces)
        rows.append({"domain": P.domain, "seed": seed, "fixed_selected": ifaces[fixed],
                     "fixed_ndcg": Y[te, fixed].mean(), "fixed_paid": bool(paid[fixed]),
                     "router_ndcg": Y[te][np.arange(len(te)), ch].mean(), "router_paid_rate": paid[ch].mean()})
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, digits: int = 4) -> str:
    if df.empty:
        return "_(none)_"
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:.{digits}f}"
        return str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--panel-dir", type=Path, required=True)
    p.add_argument("--outdir", type=Path, default=None, help="Default: <panel-dir>/../analysis")
    p.add_argument("--domains", nargs="+", default=None)
    p.add_argument("--analyses", nargs="+", default=["oos", "bootstrap", "paired", "grounding", "routing"])
    p.add_argument("--oos-reps", type=int, default=200)
    p.add_argument("--boot", type=int, default=2000, help="Bootstrap resamples for selection frequencies.")
    p.add_argument("--paired-boot", type=int, default=10000)
    p.add_argument("--perm", type=int, default=10000)
    p.add_argument("--route-reps", type=int, default=10)
    p.add_argument("--route-folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--routing-exclude", nargs="*", default=["BM25"],
                   help="Interfaces left out of the routing pool (default matches the 7-interface RQ6 pool).")
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.outdir or args.panel_dir.parent / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    domains = args.domains or sorted(p.stem[len("panel_"):] for p in args.panel_dir.glob("panel_*.csv"))
    rng = np.random.default_rng(args.seed)
    results: Dict[str, List[pd.DataFrame]] = {}
    report = ["# ALIS R1 revision analyses", ""]

    def add(key: str, df: pd.DataFrame) -> None:
        if df is not None and not df.empty:
            results.setdefault(key, []).append(df)

    for dom in domains:
        P = Panel(dom, pd.read_csv(args.panel_dir / f"panel_{dom}.csv"))
        print(f"[{dom}] users={P.n} (dropped {P.n_dropped}) interfaces={P.ifaces} features={P.feat_names}", flush=True)
        report += [f"## Panel: {dom}", "",
                   f"- users: {P.n} (dropped for missing interfaces: {P.n_dropped})",
                   f"- interfaces: {', '.join(P.ifaces)}",
                   f"- mean NDCG@20: " + ", ".join(f"{m}={v:.4f}" for m, v in zip(P.ifaces, P.Y.mean(axis=0))),
                   f"- overlap source: " + ", ".join(f"{m}: {s}" for m, s in P.overlap_source.items()),
                   f"- routing features: {', '.join(P.feat_names) or 'none'}", ""]
        if "oos" in args.analyses:
            raw = run_oos(P, args.oos_reps, rng)
            add("oos_raw", raw)
            add("oos_summary", summarize_oos(raw))
        if "bootstrap" in args.analyses:
            add("bootstrap_selection", run_bootstrap(P, args.boot, rng))
        if "paired" in args.analyses:
            add("paired_tests", run_paired(P, args.paired_boot, args.perm, rng))
        if "grounding" in args.analyses:
            add("direct_grounding_sensitivity", run_grounding_sensitivity(P))
        if "routing" in args.analyses and P.X.shape[1] > 0:
            cols = [P.j(m) for m in P.ifaces if m not in set(args.routing_exclude)]
            summ, fixed_freq, hps = run_routing(P, cols, args.route_reps, args.route_folds,
                                                args.inner_folds, args.paired_boot, rng)
            add("routing_nested_cv", summ)
            add("routing_fixed_choice_freq", fixed_freq)
            add("routing_hyperparameters", hps)
            add("routing_paper_protocol", run_paper_protocol(P, cols, P.uids))
        print(f"[{dom}] done", flush=True)

    titles = {
        "oos_summary": "Out-of-sample ALIS (select on half, evaluate on disjoint half)",
        "bootstrap_selection": "Bootstrap selection frequencies (full_panel_choice = retrospective choice)",
        "paired_tests": "Paired per-user NDCG@20 differences",
        "direct_grounding_sensitivity": "ALIS choice when Direct is assigned non-zero grounding risk",
        "routing_nested_cv": "Per-user routing, repeated nested cross-validation",
        "routing_fixed_choice_freq": "Which interface the fold-wise best fixed policy selected",
        "routing_paper_protocol": "Original RQ6 protocol (five validation/test halves)",
    }
    for key, frames in results.items():
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(outdir / f"{key}.csv", index=False)
        if key in titles:
            report += [f"## {titles[key]}", "", md_table(df), ""]
    (outdir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"wrote {len(results)} tables and report.md to {outdir}")


if __name__ == "__main__":
    main()
