#!/usr/bin/env python3
"""Calibrate the claim-audit judges against expert labels in LLM-AggreFact.

Usage:
  python scripts/run_judge_calibration.py --dry-run
  python scripts/run_judge_calibration.py --pilot 5
  python scripts/run_judge_calibration.py
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from aggrefact_calibration_lib import SYSTEM_DOC, build_sample, doc_prompt
from run_llm_claim_audit import DEFAULT_MODELS, ROOT, call, openrouter_key, parse


def load_benchmark(cache: Path) -> pd.DataFrame:
    if cache.exists():
        return pd.read_parquet(cache)
    from datasets import load_dataset
    df = load_dataset("lytang/LLM-AggreFact", split="test").to_pandas()
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def judge(sample: pd.DataFrame, model: str, key: str, outdir: Path, workers: int) -> None:
    path = outdir / f"judgments_{model.replace('/', '__')}.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("label"):
                done.add(rec["key"])
    todo = sample[~sample["key"].isin(done)]
    print(f"[{model}] {len(done)} cached, {len(todo)} to judge", flush=True)
    lock = threading.Lock()
    cost, n_ok, n_err = 0.0, 0, 0

    def work(row):
        rec = {"key": row.key, "model": model}
        for _ in range(2):
            try:
                resp = call(model, key, SYSTEM_DOC, doc_prompt(row.doc, row.claim))
                rec.update(parse(resp["content"]))
                rec.update(usage=resp["usage"], provider=resp["provider"], model_used=resp["model_used"])
                rec.pop("error", None)
                return rec
            except ValueError as e:
                rec["error"] = f"parse: {e}"
            except RuntimeError as e:
                rec["error"] = str(e)
                break
        return rec

    with ThreadPoolExecutor(max_workers=workers) as pool, open(path, "a", encoding="utf-8") as fh:
        futures = [pool.submit(work, row) for row in todo.itertuples(index=False)]
        for i, fut in enumerate(as_completed(futures), start=1):
            rec = fut.result()
            with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
            if rec.get("label"):
                n_ok += 1
                cost += float((rec.get("usage") or {}).get("cost") or 0.0)
            else:
                n_err += 1
            if i % 100 == 0 or i == len(futures):
                print(f"  [{model}] {i}/{len(futures)} ok={n_ok} err={n_err} cost=${cost:.4f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-label", type=int, default=50, help="Supported and unsupported claims per source dataset.")
    ap.add_argument("--max-doc-tokens", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--pilot", type=int, default=0, help="Judge only the first N sampled claims per source dataset.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--keyfile", type=Path, default=Path.home() / "Desktop" / "apikey.txt")
    ap.add_argument("--outdir", type=Path, default=ROOT / "results" / "revision_r1" / "judge_calibration")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    bench = load_benchmark(args.outdir / "llm_aggrefact_test.parquet")
    sample = build_sample(bench, args.per_label, args.seed, args.max_doc_tokens)
    sample[["key", "dataset", "label", "doc_tokens", "claim"]].to_csv(args.outdir / "calibration_sample.csv", index=False)
    print(sample.groupby(["dataset", "label"]).size().unstack().to_string())
    print(f"{len(sample)} claims; mean document ~{sample['doc_tokens'].mean():.0f} tokens; "
          f"excluded documents longer than {args.max_doc_tokens} tokens: "
          f"{int((bench['doc'].str.len() / 4 > args.max_doc_tokens).sum())} of {len(bench)}")
    if args.pilot:
        sample = sample.groupby("dataset", group_keys=False).head(args.pilot)
    if args.dry_run:
        r = sample.iloc[0]
        print("\n--- example prompt ---\n" + SYSTEM_DOC + "\n\n" + doc_prompt(r.doc, r.claim)[:1500])
        return
    key = openrouter_key(args.keyfile)
    for model in args.models:
        judge(sample, model, key, args.outdir, args.workers)


if __name__ == "__main__":
    main()
