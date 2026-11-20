#!/usr/bin/env python3
"""Export matched per-user panels for the R1 revision analyses (no LLM calls).

One CSV per domain, one row per test user, aligned by uid to the base
recommender's test candidate file. For every interface m the panel holds

  ndcg__m     per-user NDCG@20 on the matched candidate set
  overlap__m  per-user Top-20 overlap with the base ranking (only when a ranked
              list for m is available; otherwise left empty)

plus cheap, LLM-free context features (feat__*) for the per-user router.

Ranked lists are preferred because they also give per-user disruption; per-user
CSVs are used as a fallback for NDCG only. Missing sources are reported and
skipped, so the script also runs on partial local copies of the results tree.

Usage (on the server):
  python scripts/export_revision_panels.py --root .
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

TOPK = 20


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--outdir", type=Path, default=None,
                   help="Default: <root>/results/revision_r1/panels")
    p.add_argument("--domains", nargs="+", default=["ml1m", "yelp", "books"])
    return p.parse_args()


# ----------------------------------------------------------------------------- helpers
def ndcg_rows(ranked: np.ndarray, targets: np.ndarray) -> np.ndarray:
    out = np.zeros(len(targets))
    for r, t in enumerate(targets):
        pos = np.flatnonzero(ranked[r, :TOPK] == int(t))
        if len(pos):
            out[r] = 1.0 / math.log2(int(pos[0]) + 2)
    return out


def overlap_rows(ranked: np.ndarray, base_top: np.ndarray) -> np.ndarray:
    return np.array([len(set(ranked[r, :TOPK].tolist()) & set(base_top[r].tolist())) / TOPK
                     for r in range(len(ranked))])


def align(src_users: np.ndarray, values: np.ndarray, ref_users: np.ndarray) -> np.ndarray:
    pos = {int(u): i for i, u in enumerate(src_users)}
    out = np.full((len(ref_users),) + values.shape[1:], np.nan if values.dtype.kind == "f" else -1,
                  dtype=values.dtype if values.dtype.kind == "f" else np.int64)
    for i, u in enumerate(ref_users):
        j = pos.get(int(u))
        if j is not None:
            out[i] = values[j]
    return out


def read_per_user_csv(path: Path) -> Tuple[Optional[np.ndarray], np.ndarray]:
    uids, ndcg = [], []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        has_uid = "uid" in (reader.fieldnames or [])
        for row in reader:
            ndcg.append(float(row["NDCG@20"]))
            if has_uid:
                uids.append(int(row["uid"]))
    return (np.array(uids) if uids else None), np.array(ndcg)


def history_lengths(train_csv: Path) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    with train_csv.open(newline="") as fh:
        for row in csv.DictReader(fh):
            u = int(row["uid"])
            counts[u] = counts.get(u, 0) + 1
    return counts


class Panel:
    def __init__(self, name: str, users: np.ndarray, targets: Optional[np.ndarray],
                 base_top: Optional[np.ndarray]):
        self.name = name
        self.users = users.astype(np.int64)
        self.targets = targets
        self.base_top = base_top
        self.cols: Dict[str, np.ndarray] = {}
        self.ranked: Dict[str, np.ndarray] = {}
        self.log: List[str] = []

    def note(self, msg: str) -> None:
        print(f"[{self.name}] {msg}", flush=True)
        self.log.append(msg)

    def add_ranked(self, iface: str, ranked: np.ndarray, src_users: Optional[np.ndarray], src: Path) -> bool:
        if iface in self.ranked:
            return True
        if src_users is None:
            if len(ranked) != len(self.users):
                self.note(f"skip {iface}: {src.name} has {len(ranked)} rows, panel has {len(self.users)} and no uid array")
                return False
            src_users = self.users
        ranked = align(np.asarray(src_users), ranked.astype(np.int64), self.users)
        if self.targets is None:
            self.note(f"skip ranked {iface}: no targets available")
            return False
        valid = ranked[:, 0] >= 0
        nd = np.full(len(self.users), np.nan)
        nd[valid] = ndcg_rows(ranked[valid], self.targets[valid])
        self.cols[f"ndcg__{iface}"] = nd
        if self.base_top is not None:
            ov = np.full(len(self.users), np.nan)
            ov[valid] = overlap_rows(ranked[valid], self.base_top[valid])
            self.cols[f"overlap__{iface}"] = ov
        self.ranked[iface] = ranked
        self.note(f"{iface}: ranked list from {src} (users matched {int(valid.sum())}/{len(valid)}), "
                  f"NDCG@20={np.nanmean(nd):.4f}")
        return True

    def add_ranked_file(self, iface: str, path: Path, order: Optional[Path] = None) -> bool:
        if not path.exists():
            return False
        if path.suffix == ".npz":
            z = np.load(path)
            return self.add_ranked(iface, z["ranked"], z["users"], path)
        src_users = None
        if order is not None and order.exists():
            src_users = read_per_user_csv(order)[0] if order.suffix == ".csv" else np.load(order)["users"]
        return self.add_ranked(iface, np.load(path), src_users, path)

    def add_csv(self, iface: str, path: Path) -> bool:
        if f"ndcg__{iface}" in self.cols or not path.exists():
            return f"ndcg__{iface}" in self.cols
        uids, nd = read_per_user_csv(path)
        if uids is None:
            if len(nd) != len(self.users):
                self.note(f"skip {iface}: {path.name} lacks uid and row count differs")
                return False
            uids = self.users
        self.cols[f"ndcg__{iface}"] = align(uids, nd, self.users)
        self.note(f"{iface}: per-user NDCG only from {path} (no ranking -> no per-user overlap), "
                  f"NDCG@20={np.nanmean(self.cols[f'ndcg__{iface}']):.4f}")
        return True

    def check_against_csv(self, iface: str, path: Path) -> None:
        col = self.cols.get(f"ndcg__{iface}")
        if col is None or not path.exists():
            return
        uids, nd = read_per_user_csv(path)
        ref = align(uids if uids is not None else self.users, nd, self.users)
        ok = ~np.isnan(ref) & ~np.isnan(col)
        mism = int(np.sum(np.abs(ref[ok] - col[ok]) > 1e-9))
        self.note(f"check {iface} vs {path.name}: {mism} mismatching users of {int(ok.sum())}")

    def add_features(self, scores: Optional[np.ndarray], train_csv: Path, content_iface: str = "Content") -> None:
        hist = history_lengths(train_csv) if train_csv.exists() else {}
        if hist:
            self.cols["feat__log_hist"] = np.array([math.log1p(hist.get(int(u), 0)) for u in self.users])
        if scores is not None:
            s = -np.sort(-scores.astype(np.float64), axis=1)
            p = np.exp(s - s[:, :1])
            p = p / p.sum(axis=1, keepdims=True)
            self.cols["feat__base_top1"] = s[:, 0]
            self.cols["feat__base_margin"] = s[:, 0] - s[:, 1]
            self.cols["feat__base_entropy"] = -(p * np.log(p + 1e-12)).sum(axis=1)
            self.cols["feat__base_std"] = scores.std(axis=1)
        if self.base_top is not None and content_iface in self.ranked:
            self.cols["feat__base_content_overlap"] = self.cols[f"overlap__{content_iface}"]

    def write(self, outdir: Path) -> Path:
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / f"panel_{self.name}.csv"
        names = ["uid", "target"] + sorted(self.cols, key=lambda c: (not c.startswith("ndcg__"),
                                                                     not c.startswith("overlap__"), c))
        with path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(names)
            for i, u in enumerate(self.users):
                t = "" if self.targets is None else int(self.targets[i])
                row = [int(u), t]
                for c in names[2:]:
                    v = self.cols[c][i]
                    row.append("" if (isinstance(v, float) and math.isnan(v)) else f"{v:.6g}")
                w.writerow(row)
        (outdir / f"panel_{self.name}.log").write_text("\n".join(self.log) + "\n", encoding="utf-8")
        print(f"[{self.name}] wrote {path} ({len(self.users)} users, "
              f"{sum(c.startswith('ndcg__') for c in self.cols)} interfaces)", flush=True)
        return path


def panel_from_npz(name: str, npz: Path, n_cand: int) -> Tuple[Panel, Optional[np.ndarray]]:
    z = np.load(npz)
    cands = z["candidates"][:, :n_cand].astype(np.int64)
    scores = z["scores"][:, :n_cand].astype(np.float64)
    panel = Panel(name, z["users"], z["targets"].astype(np.int64), cands[:, :TOPK])
    panel.add_ranked("Base", cands[:, :TOPK], z["users"], npz)
    return panel, scores


def panel_from_csv(name: str, csv_path: Path) -> Panel:
    uids, nd = read_per_user_csv(csv_path)
    targets = None
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if "target" in (reader.fieldnames or []):
            targets = np.array([int(r["target"]) for r in reader])
    users = uids if uids is not None else np.arange(len(nd))
    panel = Panel(name, users, targets, None)
    panel.note(f"WARNING: no candidate file; panel users taken from {csv_path.name}"
               + ("" if uids is not None else " (row index used as uid)"))
    return panel


# ----------------------------------------------------------------------------- domains
def build_ml1m(root: Path) -> Panel:
    res = root / "results" / "egpr_profile_repair"
    fair = res / "ml1m_2000_fair_direct_candidates"
    direct_run = root / "results" / "llm_selective" / "ml1m_seed42_deepseek_2000_h20"
    profile_run = res / "ml1m_seed42_deepseek_2000_expressive5"
    ref_npz = direct_run / "candidates_lightgcn_test.npz"
    if not ref_npz.exists():
        ref_npz = profile_run / "candidates_lightgcn_test.npz"
    scores = None
    if ref_npz.exists():
        panel, scores = panel_from_npz("ml1m", ref_npz, 50)
    else:
        panel = panel_from_csv("ml1m", fair / "per_user_lightgcn.csv")

    panel.add_ranked_file("Direct", direct_run / "reranked_lightgcn_original_top20.npy", ref_npz)
    for iface, key in [("Raw", "raw"), ("Remove", "remove"), ("EGPR", "egpr")]:
        panel.add_ranked_file(iface, fair / f"reranked_profile_rerank_{key}_top20.npy", ref_npz)
    rb = fair / "reranking_baselines"
    panel.add_ranked_file("Content", rb / "ranked_content_top20.npy", profile_run / "candidates_lightgcn_test.npz")
    for strong in (root / "results" / "revision_r1" / "ml1m_strong",
                   root / "results" / "ml1m_2000_fair_direct_candidates" / "reranking_baselines_strong"):
        for iface, kind in [("Content", "overlap"), ("BM25", "bm25"), ("Dense", "dense")]:
            panel.add_ranked_file(iface, strong / f"per_user_ranked_{kind}.npz")
    panel.add_ranked_file("Dense", rb / "ranked_dense_top20.npy", fair / "per_user_lightgcn.csv")

    csvs = {"Base": "per_user_lightgcn.csv", "Direct": "per_user_direct.csv",
            "Raw": "per_user_profile_rerank_raw.csv", "Remove": "per_user_profile_rerank_remove.csv",
            "EGPR": "per_user_profile_rerank_egpr.csv"}
    for iface, fn in csvs.items():
        if iface in panel.ranked:
            panel.check_against_csv(iface, fair / fn)
        else:
            panel.add_csv(iface, fair / fn)
    panel.add_features(scores, root / "data" / "ml1m" / "train.csv")
    return panel


def build_fair_domain(root: Path, name: str, cand_run: Path, fair_dirs: List[Path], direct_dir: Path,
                      content_dir: Path, strong_dirs: List[Path], train_csv: Path,
                      csv_dir: Path, csv_names: Dict[str, str]) -> Panel:
    ref_npz = cand_run / "candidates_lightgcn_test.npz"
    scores = None
    if ref_npz.exists():
        panel, scores = panel_from_npz(name, ref_npz, 100)
    else:
        panel = panel_from_csv(name, csv_dir / csv_names["Base"])

    dumps = [d / "per_user_ranked_top100.npz" for d in fair_dirs if (d / "per_user_ranked_top100.npz").exists()]
    if dumps:
        z = np.load(dumps[0])
        for iface, key in [("Direct", "DeepSeek_Direct_Rerank"), ("Raw", "Profile_Rerank_Raw"),
                           ("Remove", "Profile_Rerank_Remove"), ("EGPR", "Profile_Rerank_EGPR")]:
            if key in z:
                panel.add_ranked(iface, z[key], z["users"], dumps[0])
    else:
        panel.note("missing per_user_ranked_top100.npz -> re-run the patched fair-confirmatory analysis")
    panel.add_ranked_file("Direct", direct_dir / "direct_rerank_top20.npy", ref_npz)
    panel.add_ranked_file("Content", content_dir / "ranked_content_top20.npy", ref_npz)
    for strong in strong_dirs:
        for iface, kind in [("BM25", "bm25"), ("Dense", "dense")]:
            panel.add_ranked_file(iface, strong / f"per_user_ranked_{kind}.npz")
    for iface, fn in csv_names.items():
        if iface in panel.ranked:
            panel.check_against_csv(iface, csv_dir / fn)
        else:
            panel.add_csv(iface, csv_dir / fn)
    panel.add_features(scores, train_csv)
    return panel


def build_yelp(root: Path) -> Panel:
    res = root / "results" / "egpr_profile_repair"
    rev = root / "results" / "revision_r1"
    run = res / "yelp_seed42_deepseek_1000_expressive5"
    vs = res / "yelp_direct_vs_profile"
    return build_fair_domain(
        root, "yelp", run, [rev / "yelp_fair", vs / "yelp_1000_fair_confirmatory"],
        vs / "direct_rerank_1000_top100", vs / "reranking_baselines", [],
        root / "data" / "yelp_day1" / "train.csv",
        run, {"Base": "per_user_lightgcn.csv", "Raw": "per_user_lightgcn_raw_profile.csv",
              "Remove": "per_user_lightgcn_remove_repair.csv",
              "EGPR": "per_user_lightgcn_evidence_weighted_repair.csv"})


def build_books(root: Path) -> Panel:
    res = root / "results" / "egpr_profile_repair"
    rev = root / "results" / "revision_r1"
    vs = res / "amazon_books_direct_vs_profile"
    return build_fair_domain(
        root, "books", res / "amazon_books_seed42_lightgcn_1000",
        [rev / "books_fair", vs / "amazon_books_1000_fair_confirmatory"],
        vs / "direct_rerank_1000_top100", vs / "reranking_baselines_1000_top100",
        [rev / "books_strong", vs / "reranking_baselines_strong"],
        root / "data" / "amazon_books_subset" / "train.csv",
        res / "amazon_books_seed42_deepseek_1000_expressive5",
        {"Base": "per_user_lightgcn.csv", "Raw": "per_user_raw.csv", "Remove": "per_user_remove.csv",
         "EGPR": "per_user_weighted.csv"})


def main() -> None:
    args = parse_args()
    outdir = args.outdir or args.root / "results" / "revision_r1" / "panels"
    builders = {"ml1m": build_ml1m, "yelp": build_yelp, "books": build_books}
    for name in args.domains:
        builders[name](args.root).write(outdir)


if __name__ == "__main__":
    main()
