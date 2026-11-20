#!/usr/bin/env python3
"""Strong non-LLM reranking baselines for the ALIS interface-selection pool.

This script adds reviewer-requested strong, cheap, non-LLM comparison baselines
on top of the existing token-overlap ``Content`` reranker. It keeps the exact
mixing structure of the paper's content interface (Eq. content):

    s(u, i) = (1 - lambda_c) * base_norm(u, i) + lambda_c * sim(u, i)

and only replaces the similarity term ``sim`` with a stronger signal:

  * ``overlap`` : Jaccard token overlap (reproduces the paper's Content row).
  * ``bm25``    : Okapi BM25 of the candidate description given the user's
                  history tokens as the query (global IDF over the item corpus).
  * ``dense``   : cosine similarity between mean-pooled Sentence-BERT embeddings
                  of the user's history items and each candidate item
                  (requires ``sentence-transformers`` + a local/downloadable
                  model; skipped automatically if unavailable).

lambda_c is tuned on the validation split (max NDCG@20) and frozen for test,
exactly as for the original content baseline.

Outputs (schema consumed by ``run_adaptive_interface_selector.py``):
  * reranking_baselines_summary.csv      (Method, NDCG@20, Recall@20)
  * reranking_baselines_reliability.csv  (Method, HarmRate, GainHarmRatio)
  * reranking_baselines_disruption.csv   (Method, Top20OverlapVsBase,
                                          MeanAbsRankShiftVsBase)

Method names emitted: LightGCN, Content Rerank, BM25 Rerank, Dense Rerank.

Everything except the optional ``dense`` baseline runs fully offline with only
numpy/pandas, so it can be executed without API keys, GPU, or internet.

Example (Amazon Books, top-100 candidates):
  python scripts/run_reranking_baselines_strong.py \
    --run-dir results/egpr_profile_repair/amazon_books_seed42_lightgcn_1000 \
    --item-text data/amazon_books_subset/item_metadata.jsonl \
    --train data/amazon_books_subset/train.csv \
    --text-fields title categories description \
    --outdir results/egpr_profile_repair/amazon_books_direct_vs_profile/reranking_baselines_strong \
    --baselines overlap bm25
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "it", "for", "on",
    "with", "as", "by", "at", "this", "that", "from", "be", "are", "was", "you",
    "your", "i", "its", "but", "not", "they", "he", "she", "his", "her", "we",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Dir with candidates_lightgcn_{val,test}.npz (users,targets,candidates,scores).")
    p.add_argument("--item-text", type=Path, required=True,
                   help="JSONL with one object per item; must contain the item id field and text fields.")
    p.add_argument("--train", type=Path, required=True,
                   help="CSV of interactions (uid,iid,timestamp,...) used to build per-user history.")
    p.add_argument("--id-field", default="iid", help="Item id field name inside --item-text (default: iid).")
    p.add_argument("--text-fields", nargs="+", default=["title", "categories", "description"],
                   help="Fields concatenated into each item's content document.")
    p.add_argument("--hist-len", type=int, default=50, help="Max recent history items per user (default: 50).")
    p.add_argument("--topk", type=int, default=20, help="Cutoff k for NDCG@k / Recall@k (default: 20).")
    p.add_argument("--lambda-grid", type=float, nargs="+",
                   default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                   help="lambda_c grid tuned on validation.")
    p.add_argument("--baselines", nargs="+", default=["overlap", "bm25"],
                   choices=["overlap", "bm25", "dense"], help="Which similarity baselines to run.")
    p.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2",
                   help="Sentence-BERT model for the dense baseline (only if 'dense' selected).")
    p.add_argument("--outdir", type=Path, required=True)
    return p.parse_args()


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 1]


def load_item_docs(path: Path, id_field: str, text_fields: Sequence[str]) -> Dict[int, str]:
    docs: Dict[int, str] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            iid = int(obj[id_field])
            parts: List[str] = []
            for f in text_fields:
                v = obj.get(f)
                if v is None:
                    continue
                if isinstance(v, (list, tuple)):
                    parts.append(" ".join(str(x) for x in v))
                else:
                    parts.append(str(v))
            docs[iid] = " ".join(parts)
    return docs


def build_history(train: Path, hist_len: int) -> Dict[int, List[int]]:
    df = pd.read_csv(train)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    hist: Dict[int, List[int]] = defaultdict(list)
    for uid, iid in zip(df["uid"].to_numpy(), df["iid"].to_numpy()):
        hist[int(uid)].append(int(iid))
    return {u: items[-hist_len:] for u, items in hist.items()}


def minmax(row: np.ndarray) -> np.ndarray:
    lo, hi = float(row.min()), float(row.max())
    if hi - lo < 1e-12:
        return np.zeros_like(row)
    return (row - lo) / (hi - lo)


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


class BM25:
    """Okapi BM25 with IDF estimated over the full item corpus."""

    def __init__(self, docs_tokens: Mapping[int, List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tf: Dict[int, Counter] = {i: Counter(toks) for i, toks in docs_tokens.items()}
        self.doc_len: Dict[int, int] = {i: len(toks) for i, toks in docs_tokens.items()}
        n = len(docs_tokens)
        self.avgdl = (sum(self.doc_len.values()) / n) if n else 0.0
        df: Counter = Counter()
        for toks in docs_tokens.values():
            for t in set(toks):
                df[t] += 1
        self.idf = {t: math.log(1.0 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def score(self, query_tokens: Iterable[str], doc_id: int) -> float:
        tf = self.doc_tf.get(doc_id)
        if not tf:
            return 0.0
        dl = self.doc_len.get(doc_id, 0)
        denom_norm = self.k1 * (1.0 - self.b + self.b * dl / self.avgdl) if self.avgdl else self.k1
        s = 0.0
        for term in set(query_tokens):
            f = tf.get(term, 0)
            if f == 0:
                continue
            s += self.idf.get(term, 0.0) * (f * (self.k1 + 1.0)) / (f + denom_norm)
        return s


def ndcg_recall(ranked: np.ndarray, targets: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    ndcg = np.zeros(len(targets))
    recall = np.zeros(len(targets))
    for r, tgt in enumerate(targets):
        pos = np.flatnonzero(ranked[r, :k] == int(tgt))
        if len(pos):
            ndcg[r] = 1.0 / math.log2(int(pos[0]) + 2)
            recall[r] = 1.0
    return ndcg, recall


def rerank(scores_sim: np.ndarray, base_scores: np.ndarray, candidates: np.ndarray, lam: float) -> np.ndarray:
    """Return reranked candidate ids per user under mixing weight lam."""
    out = np.empty_like(candidates)
    for r in range(candidates.shape[0]):
        base_norm = minmax(base_scores[r])
        mixed = (1.0 - lam) * base_norm + lam * scores_sim[r]
        order = np.argsort(-mixed, kind="stable")
        out[r] = candidates[r][order]
    return out


def disruption(base_ranked: np.ndarray, method_ranked: np.ndarray, k: int) -> tuple[float, float]:
    overlaps, shifts = [], []
    for r in range(base_ranked.shape[0]):
        b_top = base_ranked[r, :k]
        m_top = method_ranked[r, :k]
        overlaps.append(len(set(b_top) & set(m_top)) / k)
        pos_b = {int(it): i for i, it in enumerate(base_ranked[r])}
        shift = [abs(pos_b[int(it)] - j) for j, it in enumerate(method_ranked[r]) if int(it) in pos_b]
        shifts.append(float(np.mean(shift)) if shift else 0.0)
    return float(np.mean(overlaps)), float(np.mean(shifts))


def sign_flip_pvalue(base_ndcg: np.ndarray, method_ndcg: np.ndarray, n_resamples: int = 10000,
                     seed: int = 0) -> float:
    """Two-sided paired sign-flip permutation test on per-user NDCG deltas."""
    delta = method_ndcg - base_ndcg
    nz = delta[np.abs(delta) > 1e-12]
    if len(nz) == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    obs = abs(float(nz.mean()))
    signs = rng.choice([-1.0, 1.0], size=(n_resamples, len(nz)))
    perm_means = np.abs((signs * np.abs(nz)).mean(axis=1))
    return float((perm_means >= obs - 1e-12).mean())


def reliability(base_ndcg: np.ndarray, method_ndcg: np.ndarray) -> tuple[float, float]:
    delta = method_ndcg - base_ndcg
    pos = float(delta[delta > 0].sum())
    neg = float(abs(delta[delta < 0].sum()))
    harm = float(np.mean(delta < 0))
    ghr = (pos / neg) if neg > 1e-12 else math.inf
    return harm, ghr


def sim_matrix(kind: str, users: np.ndarray, candidates: np.ndarray, hist: Mapping[int, List[int]],
               item_tokens: Mapping[int, List[str]], bm25: BM25 | None,
               dense_emb: np.ndarray | None) -> np.ndarray:
    n, m = candidates.shape
    out = np.zeros((n, m), dtype=np.float64)
    for r in range(n):
        u = int(users[r])
        h_items = hist.get(u, [])
        if kind == "dense" and dense_emb is not None:
            h_emb = dense_emb[[i for i in h_items if i < len(dense_emb)]]
            if len(h_emb) == 0:
                continue
            uvec = h_emb.mean(axis=0)
            uvec = uvec / (np.linalg.norm(uvec) + 1e-12)
            for c in range(m):
                iid = int(candidates[r, c])
                if iid < len(dense_emb):
                    cv = dense_emb[iid]
                    out[r, c] = float(uvec @ (cv / (np.linalg.norm(cv) + 1e-12)))
        else:
            h_tokens: List[str] = []
            for i in h_items:
                h_tokens.extend(item_tokens.get(i, []))
            if kind == "overlap":
                h_set = set(h_tokens)
                for c in range(m):
                    out[r, c] = jaccard(h_set, set(item_tokens.get(int(candidates[r, c]), [])))
            elif kind == "bm25" and bm25 is not None:
                for c in range(m):
                    out[r, c] = bm25.score(h_tokens, int(candidates[r, c]))
        out[r] = minmax(out[r])
    return out


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    val = np.load(args.run_dir / "candidates_lightgcn_val.npz")
    test = np.load(args.run_dir / "candidates_lightgcn_test.npz")

    docs = load_item_docs(args.item_text, args.id_field, args.text_fields)
    item_tokens = {i: tokenize(d) for i, d in docs.items()}
    hist = build_history(args.train, args.hist_len)
    bm25 = BM25(item_tokens) if "bm25" in args.baselines else None

    dense_emb = None
    if "dense" in args.baselines:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            model = SentenceTransformer(args.dense_model)
            max_id = max(docs) + 1
            texts = [docs.get(i, "") for i in range(max_id)]
            dense_emb = np.asarray(model.encode(texts, batch_size=256, show_progress_bar=True))
        except Exception as exc:  # noqa: BLE001
            print(f"[dense] skipped ({exc}); install sentence-transformers + model to enable.")
            args.baselines = [b for b in args.baselines if b != "dense"]

    summary_rows, rel_rows, dis_rows = [], [], []

    def base_eval(split):
        ranked = split["candidates"].astype(np.int64)
        ndcg, recall = ndcg_recall(ranked, split["targets"].astype(np.int64), args.topk)
        return ranked, ndcg, recall

    base_ranked_test, base_ndcg_test, base_recall_test = base_eval(test)
    summary_rows.append({"Method": "LightGCN", "NDCG@20": base_ndcg_test.mean(), "Recall@20": base_recall_test.mean()})
    rel_rows.append({"Method": "LightGCN", "HarmRate": 0.0, "GainHarmRatio": math.inf})
    dis_rows.append({"Method": "LightGCN", "Top20OverlapVsBase": 1.0, "MeanAbsRankShiftVsBase": 0.0})

    method_name = {"overlap": "Content Rerank", "bm25": "BM25 Rerank", "dense": "Dense Rerank"}
    for kind in args.baselines:
        sim_val = sim_matrix(kind, val["users"], val["candidates"].astype(np.int64), hist,
                             item_tokens, bm25, dense_emb)
        best_lam, best_val = args.lambda_grid[0], -1.0
        for lam in args.lambda_grid:
            r_val = rerank(sim_val, val["scores"], val["candidates"].astype(np.int64), lam)
            v_ndcg, _ = ndcg_recall(r_val, val["targets"].astype(np.int64), args.topk)
            if v_ndcg.mean() > best_val:
                best_val, best_lam = v_ndcg.mean(), lam

        sim_test = sim_matrix(kind, test["users"], test["candidates"].astype(np.int64), hist,
                              item_tokens, bm25, dense_emb)
        r_test = rerank(sim_test, test["scores"], test["candidates"].astype(np.int64), best_lam)
        np.savez_compressed(args.outdir / f"per_user_ranked_{kind}.npz", users=test["users"],
                            targets=test["targets"], ranked=r_test[:, :args.topk], lambda_c=best_lam)
        m_ndcg, m_recall = ndcg_recall(r_test, test["targets"].astype(np.int64), args.topk)
        harm, ghr = reliability(base_ndcg_test, m_ndcg)
        ov, shift = disruption(base_ranked_test, r_test, args.topk)
        pval = sign_flip_pvalue(base_ndcg_test, m_ndcg)
        name = method_name[kind]
        summary_rows.append({"Method": name, "NDCG@20": m_ndcg.mean(), "Recall@20": m_recall.mean(),
                             "lambda_c": best_lam, "pval_vs_base": pval})
        rel_rows.append({"Method": name, "HarmRate": harm, "GainHarmRatio": ghr})
        dis_rows.append({"Method": name, "Top20OverlapVsBase": ov, "MeanAbsRankShiftVsBase": shift})
        print(f"{name}: NDCG@{args.topk}={m_ndcg.mean():.4f} Recall@{args.topk}={m_recall.mean():.4f} "
              f"(base {base_ndcg_test.mean():.4f}) lambda_c={best_lam} HarmRate={harm:.3f} GHR={ghr:.3f} "
              f"Top{args.topk}Overlap={ov:.3f} p_vs_base={pval:.4f}")

    pd.DataFrame(summary_rows).to_csv(args.outdir / "reranking_baselines_summary.csv", index=False)
    pd.DataFrame(rel_rows).to_csv(args.outdir / "reranking_baselines_reliability.csv", index=False)
    pd.DataFrame(dis_rows).to_csv(args.outdir / "reranking_baselines_disruption.csv", index=False)
    print(f"wrote 3 CSVs to {args.outdir}")


if __name__ == "__main__":
    main()
