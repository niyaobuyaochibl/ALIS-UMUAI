#!/usr/bin/env python3
"""Agreement of the claim-audit judges with expert labels on LLM-AggreFact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_llm_claim_audit import LABELS, SHORT, cohen_kappa, fleiss_kappa

ROOT = Path(__file__).resolve().parents[1]


def bacc(pred: np.ndarray, gold: np.ndarray) -> float:
    return 0.5 * (float(pred[gold == 1].mean()) + float(1 - pred[gold == 0].mean()))


def stratified_boot(df: pd.DataFrame, col: str, reps: int, seed: int) -> tuple[float, float, float, float]:
    """Point estimate and 95% CI for BAcc and Cohen's kappa, resampling within dataset x label cells."""
    rng = np.random.default_rng(seed)
    cells = [g.index.values for _, g in df.groupby(["dataset", "label"])]
    gold, pred = df["label"].values, df[col].values
    pos = {k: i for i, k in enumerate(df.index)}
    b, k = [], []
    for _ in range(reps):
        idx = np.concatenate([[pos[j] for j in rng.choice(c, len(c))] for c in cells])
        b.append(bacc(pred[idx], gold[idx]))
        k.append(cohen_kappa(pd.Series(pred[idx]), pd.Series(gold[idx])))
    lo_b, hi_b = np.percentile(b, [2.5, 97.5])
    lo_k, hi_k = np.percentile(k, [2.5, 97.5])
    return lo_b, hi_b, lo_k, hi_k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=ROOT / "results" / "revision_r1" / "judge_calibration")
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()

    sample = pd.read_csv(args.outdir / "calibration_sample.csv").set_index("key")
    labels = {}
    for p in sorted(args.outdir.glob("judgments_*.jsonl")):
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        df = pd.DataFrame(rows)
        df = df[df["label"].isin(LABELS)].drop_duplicates("key", keep="last").set_index("key")
        labels[df["model"].iloc[0]] = df["label"]
    models = [m for m in SHORT if m in labels]
    lab = pd.DataFrame({m: labels[m] for m in models}).dropna()
    df = sample.loc[lab.index].copy()
    print(f"{len(df)} claims judged by all {len(models)} judges (of {len(sample)} sampled)")

    for m in models:
        df[m] = (lab[m] == "SUPPORTED").astype(int)
        df[m + "|lenient"] = lab[m].isin(["SUPPORTED", "OVERGENERALIZED"]).astype(int)
    df["majority"] = (df[models].sum(axis=1) * 2 > len(models)).astype(int)
    df["majority|lenient"] = (df[[m + "|lenient" for m in models]].sum(axis=1) * 2 > len(models)).astype(int)

    rows = []
    for col in models + ["majority"]:
        gold, pred = df["label"].values, df[col].values
        lo_b, hi_b, lo_k, hi_k = stratified_boot(df, col, args.reps, seed=2026)
        per_ds = df.groupby("dataset").apply(lambda g: bacc(g[col].values, g["label"].values))
        rows.append({
            "judge": SHORT.get(col, "Majority of three"),
            "bacc": bacc(pred, gold), "bacc_lo": lo_b, "bacc_hi": hi_b,
            "kappa": cohen_kappa(pd.Series(pred), pd.Series(gold)), "kappa_lo": lo_k, "kappa_hi": hi_k,
            "bacc_lenient": bacc(df[col + "|lenient"].values, gold),
            "recall_supported": float(pred[gold == 1].mean()),
            "recall_unsupported": float(1 - pred[gold == 0].mean()),
            "bacc_dataset_min": float(per_ds.min()), "bacc_dataset_max": float(per_ds.max()),
            "bacc_dataset_mean": float(per_ds.mean()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(args.outdir / "calibration_summary.csv", index=False)
    print("\n== agreement with expert labels ==\n" + summary.round(3).to_string(index=False))

    per_ds = pd.DataFrame({SHORT.get(c, "Majority"): df.groupby("dataset").apply(lambda g, c=c: bacc(g[c].values, g["label"].values))
                           for c in models + ["majority"]})
    per_ds.to_csv(args.outdir / "calibration_by_dataset.csv")
    print("\n== balanced accuracy by source dataset ==\n" + per_ds.round(3).to_string())

    fk5 = fleiss_kappa(lab[models])
    fk2 = fleiss_kappa(df[models])
    print(f"\n== inter-judge agreement on this benchmark == Fleiss kappa five labels {fk5:.3f}, binary {fk2:.3f}")

    rejected = lab.loc[df.index[(df["label"] == 1)]]
    dist = {SHORT[m]: rejected[m][rejected[m] != "SUPPORTED"].value_counts(normalize=True).round(3).to_dict() for m in models}
    print("\n== labels given to expert-supported claims that a judge did not accept ==")
    for k, v in dist.items():
        print(f"  {k}: {v}")

    out = {"n_claims": int(len(df)), "n_sampled": int(len(sample)), "fleiss_kappa_five": fk5, "fleiss_kappa_binary": fk2,
           "summary": summary.to_dict(orient="records"), "rejected_supported_label_share": dist}
    (args.outdir / "calibration_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
