#!/usr/bin/env python3
"""Summarize the LLM-judge claim audit: agreement, stratum profiles, and correspondence with support scores."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["SUPPORTED", "OVERGENERALIZED", "OVERLY_SPECIFIC", "NOT_SUPPORTED", "CONTRADICTED"]
STRATA = ["retained", "downweighted", "removed"]
SHORT = {"anthropic/claude-haiku-5.5": "Claude Haiku 5.5", "openai/gpt-6-luna": "GPT-6 Luna",
         "google/gemini-3.8-flash": "Gemini 3.8 Flash"}


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    cats = sorted(set(a) | set(b))
    po = float((a.values == b.values).mean())
    pe = sum(float((a == c).mean()) * float((b == c).mean()) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def fleiss_kappa(wide: pd.DataFrame) -> float:
    cats = sorted(set(wide.values.ravel()))
    n = wide.shape[1]
    counts = np.stack([(wide.values == c).sum(axis=1) for c in cats], axis=1).astype(float)
    p_i = ((counts * (counts - 1)).sum(axis=1)) / (n * (n - 1))
    p_j = counts.sum(axis=0) / counts.sum()
    pe = float((p_j ** 2).sum())
    return (float(p_i.mean()) - pe) / (1 - pe) if pe < 1 else float("nan")


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def auc(score: np.ndarray, y: np.ndarray) -> float:
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank().values
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load(outdir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    sample = pd.read_csv(outdir / "audit_sample.csv")
    frames = []
    for p in sorted(outdir.glob("judgments_*.jsonl")):
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        df = pd.DataFrame(rows)
        df = df[df.get("label").notna()] if "label" in df else df.iloc[0:0]
        frames.append(df.drop_duplicates("key", keep="last")[["key", "model", "label", "sensitive"]])
    j = pd.concat(frames, ignore_index=True)
    models = [m for m in SHORT if m in set(j.model)] + sorted(set(j.model) - set(SHORT))
    return sample, j, models


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=ROOT / "results" / "revision_r1" / "claim_audit")
    args = ap.parse_args()
    sample, j, models = load(args.outdir)

    lab = j.pivot(index="key", columns="model", values="label")[models].dropna()
    sens = j.pivot(index="key", columns="model", values="sensitive")[models].loc[lab.index].astype(bool)
    print(f"claims with all {len(models)} judgments: {len(lab)} / {len(sample)}")

    agree_rows = []
    sup = (lab == "SUPPORTED").astype(int)
    for a, b in combinations(models, 2):
        agree_rows.append({"pair": f"{SHORT.get(a, a)} vs {SHORT.get(b, b)}",
                           "raw_agreement_5class": float((lab[a] == lab[b]).mean()),
                           "cohen_kappa_5class": cohen_kappa(lab[a], lab[b]),
                           "cohen_kappa_supported": cohen_kappa(sup[a], sup[b]),
                           "cohen_kappa_sensitive": cohen_kappa(sens[a].astype(int), sens[b].astype(int))})
    agree_rows.append({"pair": "Fleiss (all judges)",
                       "raw_agreement_5class": float((lab.nunique(axis=1) == 1).mean()),
                       "cohen_kappa_5class": fleiss_kappa(lab),
                       "cohen_kappa_supported": fleiss_kappa(sup),
                       "cohen_kappa_sensitive": fleiss_kappa(sens.astype(int))})
    agree = pd.DataFrame(agree_rows)
    agree.to_csv(args.outdir / "agreement.csv", index=False)

    def majority(row):
        vc = row.value_counts()
        return vc.index[0] if vc.iloc[0] * 2 > len(row) else "NO_MAJORITY"

    claims = sample.set_index("key").loc[lab.index].copy()
    claims["majority"] = lab.apply(majority, axis=1)
    claims["frac_supported"] = sup.mean(axis=1)
    claims["maj_supported"] = (sup.sum(axis=1) * 2 > len(models)).astype(int)
    claims["maj_sensitive"] = (sens.sum(axis=1) * 2 > len(models)).astype(int)
    claims["maj_unsupported"] = lab.isin(["NOT_SUPPORTED", "CONTRADICTED"]).sum(axis=1).mul(2).gt(len(models)).astype(int)
    claims.reset_index().to_csv(args.outdir / "claim_level_majority.csv", index=False)

    strat_rows = []
    for (dom, st), g in claims.groupby(["domain", "stratum"]):
        n = len(g)
        row = {"domain": dom, "stratum": st, "n": n, "stratum_share": float(g.stratum_size.iloc[0] / g.domain_size.iloc[0])}
        for lbl in LABELS + ["NO_MAJORITY"]:
            row[f"maj_{lbl}"] = float((g.majority == lbl).mean())
        lo, hi = wilson(g.maj_supported.sum(), n)
        row.update(maj_supported=float(g.maj_supported.mean()), sup_ci_lo=lo, sup_ci_hi=hi,
                   maj_sensitive=float(g.maj_sensitive.mean()), mean_support_weight=float(g.support_weight.mean()))
        strat_rows.append(row)
    strat = pd.DataFrame(strat_rows)
    strat["stratum"] = pd.Categorical(strat.stratum, STRATA, ordered=True)
    strat = strat.sort_values(["domain", "stratum"])
    strat.to_csv(args.outdir / "stratum_profile.csv", index=False)

    dom_rows = []
    for dom, g in claims.groupby("domain"):
        g = g.assign(w=(g.stratum_size / g.domain_size) / g.groupby("stratum")["uid"].transform("size"))
        kept = g[g.stratum != "removed"]
        removed = g[g.stratum == "removed"]
        unsup_w = (g.w * (1 - g.maj_supported)).sum()
        dom_rows.append({
            "domain": dom,
            "n": len(g),
            "pop_supported_all_claims": float((g.w * g.maj_supported).sum()),
            "pop_supported_after_remove": float((kept.w * kept.maj_supported).sum() / kept.w.sum()) if len(kept) else float("nan"),
            "pop_sensitive_all_claims": float((g.w * g.maj_sensitive).sum()),
            "removed_judged_not_supported": float(1 - removed.maj_supported.mean()) if len(removed) else float("nan"),
            "share_of_not_supported_removed": float((removed.w * (1 - removed.maj_supported)).sum() / unsup_w) if unsup_w > 0 else float("nan"),
            "auc_weight_vs_supported": auc(g.support_weight.values, g.maj_supported.values),
            "spearman_weight_vs_frac_supported": float(g.support_weight.corr(g.frac_supported, method="spearman")),
        })
    pooled = claims
    dom_rows.append({"domain": "pooled", "n": len(claims),
                     "auc_weight_vs_supported": auc(pooled.support_weight.values, pooled.maj_supported.values),
                     "spearman_weight_vs_frac_supported": float(pooled.support_weight.corr(pooled.frac_supported, method="spearman"))})
    dom = pd.DataFrame(dom_rows)
    dom.to_csv(args.outdir / "domain_summary.csv", index=False)

    sens_by_type = claims.groupby(["domain", "claim_type"]).maj_sensitive.agg(["mean", "size"]).reset_index()
    sens_by_type.to_csv(args.outdir / "sensitive_by_type.csv", index=False)

    per_model = j.groupby("model").label.value_counts(normalize=True).unstack().reindex(columns=LABELS).fillna(0)
    per_model.to_csv(args.outdir / "label_distribution_by_model.csv")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("\n== agreement ==\n", agree.round(3).to_string(index=False))
    print("\n== label distribution by model ==\n", per_model.round(3))
    print("\n== stratum profile (majority labels) ==\n", strat.round(3).to_string(index=False))
    print("\n== domain summary ==\n", dom.round(3).to_string(index=False))
    print("\n== examples of majority-sensitive claims ==")
    for _, r in claims[claims.maj_sensitive == 1].head(12).iterrows():
        print(f"  [{r.domain}/{r.stratum}/{r.majority}] {r.claim[:110]}")


if __name__ == "__main__":
    main()
