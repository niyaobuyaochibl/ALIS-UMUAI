#!/usr/bin/env python3
"""Blind LLM-judge audit of generated profile claims (R1 comment 4).

Each sampled claim is shown to several judge models together with the exact
history text the profile generator saw, without the claim's support score or
repair outcome. Judges label it SUPPORTED / OVERGENERALIZED / OVERLY_SPECIFIC /
NOT_SUPPORTED / CONTRADICTED and flag potentially sensitive inferences.

Claims are stratified by governance outcome so that every stratum is audited:
  removed       status == unsupported (dropped by Remove, low EGPR weight)
  downweighted  kept by Remove, EGPR weight < 0.5
  retained      kept by Remove, EGPR weight >= 0.5

Usage:
  python scripts/run_llm_claim_audit.py --dry-run
  python scripts/run_llm_claim_audit.py --pilot 10
  python scripts/run_llm_claim_audit.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "egpr_profile_repair"
RUNS = {
    "ml1m": RES / "ml1m_seed42_deepseek_2000_expressive5",
    "yelp": RES / "yelp_seed42_deepseek_1000_expressive5",
    "books": RES / "amazon_books_seed42_deepseek_1000_expressive5",
}
DOMAIN_TEXT = {"ml1m": "movies (MovieLens)", "yelp": "local businesses (Yelp)", "books": "books (Amazon)"}
HISTORY_LIMIT = 20
DEFAULT_MODELS = ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "google/gemini-3.8-flash"]
LABELS = ["SUPPORTED", "OVERGENERALIZED", "OVERLY_SPECIFIC", "NOT_SUPPORTED", "CONTRADICTED"]
STRATA = ["removed", "downweighted", "retained"]

SYSTEM = (
    "You are an expert annotator auditing user profiles generated for a recommender system. "
    "You will see a user's recent interaction history and ONE preference claim that a model wrote about this user. "
    "Judge only from the history shown; do not rely on assumptions about what such users typically like."
)

INSTRUCTIONS = """Choose exactly one label for the claim:
- SUPPORTED: several items in the history clearly support the claim as stated.
- OVERGENERALIZED: the claim has some basis but states it more broadly or strongly than the history warrants (for example, it generalizes from one or two items, or asserts a strong preference when the evidence is mixed).
- OVERLY_SPECIFIC: the general direction may be supported, but the claim adds details, qualifiers, motives, or circumstances (for example, time of day, companions, reasons, sub-themes) that the history cannot show.
- NOT_SUPPORTED: nothing in the history supports the claim, and nothing contradicts it.
- CONTRADICTED: the history points against the claim.

Separately, set "sensitive" to true if the claim states or implies a sensitive personal attribute (for example religion, health or a medical condition, sexual orientation, political views, ethnicity, age or family status, financial situation), whether or not it is supported.

Return JSON only: {"label": "<one label>", "sensitive": true or false, "reason": "<at most 25 words>"}"""


# ---------------------------------------------------------------- histories
def histories_from_jsonl(domain: str) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {}
    with open(RUNS[domain] / "user_history.jsonl", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            lines = []
            for pos, it in enumerate(rec["history"][-HISTORY_LIMIT:], start=1):
                if domain == "ml1m":
                    genres = "|".join(it.get("genres") or []) or "Unknown"
                    lines.append(f"{pos}. {it['title']} | {genres}")
                else:
                    cats = ", ".join(it.get("categories") or []) or "Unknown"
                    loc = ", ".join(p for p in [it.get("city", ""), it.get("state", "")] if p)
                    lines.append(f"{pos}. {it['name']} | {cats}" + (f" | {loc}" if loc else ""))
            out[int(rec["uid"])] = lines
    return out


def book_profile_prompt(uid: int, lines: List[str]) -> str:
    head = [
        "Given this user's Amazon Books interaction history, generate 5 concise preference claims.",
        "Generate a rich but evidence-aware reading profile with explicit and latent preferences.",
        "Include genre, topic, style, theme, audience, or reading intent when supported by the history.",
        "Do not mention exact book titles.",
        "Return valid JSON only with schema {\"claims\":[{\"claim\":\"...\",\"type\":\"genre/topic/style/theme/intent\",\"confidence\":0.0}]}",
        "",
        f"User: U{uid}",
        "User history:",
    ]
    return "\n".join(head + (lines or ["None"]))


def histories_books(uids: List[int]) -> Dict[int, List[str]]:
    """Rebuild the Books history text and verify it against the profile API cache keys."""
    datadir = ROOT / "data" / "amazon_books_subset"
    train = pd.read_csv(datadir / "train.csv", usecols=["uid", "iid", "timestamp"])
    train = train[train["uid"].isin(set(uids))].sort_values(["uid", "timestamp", "iid"], kind="mergesort")
    hist = train.groupby("uid")["iid"].apply(list).to_dict()
    needed = {i for u in uids for i in hist.get(u, [])[-HISTORY_LIMIT:]}
    meta: Dict[int, tuple] = {}
    with open(datadir / "item_metadata.jsonl", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if int(row["iid"]) in needed:
                meta[int(row["iid"])] = (str(row.get("title", "Unknown Book")), [str(c) for c in row.get("categories", [])])
    out: Dict[int, List[str]] = {}
    for u in uids:
        lines = []
        for pos, iid in enumerate(hist.get(u, [])[-HISTORY_LIMIT:], start=1):
            title, cats = meta.get(iid, ("Unknown Book", []))
            lines.append(f"{pos}. {title} | {', '.join(cats[:5]) if cats else 'Unknown'}")
        out[u] = lines
    cache = RUNS["books"] / "profile_api_cache"
    if os.name == "nt":
        cache = Path("\\\\?\\" + str(cache.resolve()))
    hits = sum((cache / (hashlib.sha256(("deepseek\ndeepseek-v4-flash\n" + book_profile_prompt(u, out[u]))
                                         .encode("utf-8")).hexdigest() + ".json")).exists() for u in uids)
    print(f"[books] history reconstruction verified against profile cache: {hits}/{len(uids)} users")
    if hits < len(uids):
        raise SystemExit("Books history does not match the prompts the profiles were generated from.")
    return out


# ---------------------------------------------------------------- sampling
def stratum(row) -> str:
    if row["status"] == "unsupported":
        return "removed"
    return "downweighted" if float(row["support_weight"]) < 0.5 else "retained"


def build_sample(per_stratum: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for dom, run in RUNS.items():
        df = pd.read_json(run / "claim_support.jsonl", lines=True)
        df["domain"] = dom
        df["stratum"] = df.apply(stratum, axis=1)
        sizes = df["stratum"].value_counts().to_dict()
        for s in STRATA:
            pool = df[df["stratum"] == s]
            take = pool.iloc[rng.permutation(len(pool))[:per_stratum]].copy()
            take["stratum_size"] = sizes.get(s, 0)
            take["domain_size"] = len(df)
            frames.append(take)
    sample = pd.concat(frames, ignore_index=True)
    sample["key"] = sample["domain"] + ":" + sample["uid"].astype(str) + ":" + sample["claim_id"].astype(str)
    return sample.iloc[rng.permutation(len(sample))].reset_index(drop=True)


# ---------------------------------------------------------------- judging
def openrouter_key(keyfile: Path) -> str:
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    for line in keyfile.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.match(r"\s*openrouter\s*[:：]\s*(\S+)", line)
        if m:
            return m.group(1)
    raise SystemExit("OpenRouter key not found")


def user_prompt(domain: str, lines: List[str], claim: str) -> str:
    return (f"Domain: {DOMAIN_TEXT[domain]}\n\nInteraction history (oldest first, most recent last):\n"
            + "\n".join(lines or ["(empty)"]) + f"\n\nClaim: \"{claim}\"\n\n" + INSTRUCTIONS)


def parse(content: str) -> dict:
    m = re.search(r"\{.*\}", content or "", re.S)
    if not m:
        raise ValueError("no JSON object")
    obj = json.loads(m.group(0))
    label = str(obj.get("label", "")).strip().upper().replace(" ", "_")
    if label not in LABELS:
        raise ValueError(f"bad label {label!r}")
    sens = obj.get("sensitive", False)
    sens = sens if isinstance(sens, bool) else str(sens).strip().lower() == "true"
    return {"label": label, "sensitive": sens, "reason": str(obj.get("reason", ""))[:300]}


def call(model: str, key: str, system: str, prompt: str, retries: int = 5) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 2000,
        "reasoning": {"effort": "low"},
        "response_format": {"type": "json_object"},
        "usage": {"include": True},
    }
    last = ""
    for attempt in range(retries):
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                     "X-Title": "ALIS claim audit"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.load(r)
            if "error" in data:
                raise RuntimeError(json.dumps(data["error"])[:300])
            content = data["choices"][0]["message"].get("content") or ""
            usage = data.get("usage", {})
            return {"content": content, "usage": usage, "provider": data.get("provider"), "model_used": data.get("model")}
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:300]!r}"
            if e.code in (400, 401, 402, 403):
                break
        except Exception as e:  # noqa: BLE001
            last = str(e)[:300]
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(last)


def judge_all(sample: pd.DataFrame, histories: Dict[str, Dict[int, List[str]]], models: List[str],
              key: str, outdir: Path, workers: int) -> None:
    lock = threading.Lock()
    for model in models:
        path = outdir / f"judgments_{model.replace('/', '__')}.jsonl"
        done = set()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if rec.get("label"):
                    done.add(rec["key"])
        todo = sample[~sample["key"].isin(done)]
        print(f"[{model}] {len(done)} cached, {len(todo)} to judge", flush=True)
        cost = 0.0
        n_ok = n_err = 0

        def work(row):
            prompt = user_prompt(row.domain, histories[row.domain].get(int(row.uid), []), row.claim)
            rec = {"key": row.key, "model": model}
            for _ in range(2):
                try:
                    resp = call(model, key, SYSTEM, prompt)
                    rec.update(parse(resp["content"]))
                    rec.update(usage=resp["usage"], provider=resp["provider"], model_used=resp["model_used"])
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
                if i % 50 == 0 or i == len(futures):
                    print(f"  [{model}] {i}/{len(futures)} ok={n_ok} err={n_err} cost=${cost:.4f}", flush=True)
        if n_ok:
            print(f"[{model}] mean cost per claim ${cost / n_ok:.5f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-stratum", type=int, default=134, help="Claims per domain x stratum (default 134 -> 1,206 claims).")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--pilot", type=int, default=0, help="Judge only the first N sampled claims per domain.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--keyfile", type=Path, default=Path.home() / "Desktop" / "apikey.txt")
    ap.add_argument("--outdir", type=Path, default=ROOT / "results" / "revision_r1" / "claim_audit")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    sample = build_sample(args.per_stratum, args.seed)
    histories = {"ml1m": histories_from_jsonl("ml1m"), "yelp": histories_from_jsonl("yelp"),
                 "books": histories_books(sorted(sample.loc[sample.domain == "books", "uid"].astype(int).unique()))}
    missing = sample.apply(lambda r: int(r.uid) not in histories[r.domain], axis=1)
    if missing.any():
        print(f"dropping {int(missing.sum())} claims whose user history is unavailable")
        sample = sample[~missing].reset_index(drop=True)
    cols = ["key", "domain", "uid", "claim_id", "claim", "claim_type", "status", "support_score",
            "support_weight", "support_count", "stratum", "stratum_size", "domain_size"]
    sample[cols].to_csv(args.outdir / "audit_sample.csv", index=False)
    print(sample.groupby(["domain", "stratum"]).size().unstack())

    if args.pilot:
        sample = sample.groupby("domain", group_keys=False).head(args.pilot)
    lens = [len(user_prompt(r.domain, histories[r.domain].get(int(r.uid), []), r.claim)) + len(SYSTEM)
            for r in sample.itertuples(index=False)]
    print(f"{len(sample)} claims x {len(args.models)} models; mean prompt ~{np.mean(lens) / 4:.0f} tokens (chars/4)")
    if args.dry_run:
        r = sample.iloc[0]
        print("\n--- example prompt ---\n" + SYSTEM + "\n\n" + user_prompt(r.domain, histories[r.domain][int(r.uid)], r.claim))
        return
    judge_all(sample, histories, args.models, openrouter_key(args.keyfile), args.outdir, args.workers)


if __name__ == "__main__":
    main()
