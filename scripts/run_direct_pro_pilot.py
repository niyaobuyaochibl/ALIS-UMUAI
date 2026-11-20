#!/usr/bin/env python3
"""Frontier-model robustness check: re-run Direct LLM reranking on the ML-1M
confirmatory panel with the STRONGER deepseek-v4-pro model and compare against
the deepseek-v4-flash model used in the paper and the free Dense reranker.

The Direct prompt is byte-for-byte the same builder as the paper
(run_llm_selective_invocation_pilot.build_prompt), so flash vs pro differ only
in the model id. Responses are cached per (model, uid) for resumability.

Usage: DEEPSEEK_API_KEY=... python3 run_direct_pro_pilot.py --model deepseek-v4-pro --max-users 150
"""
from __future__ import annotations
import argparse, csv, json, math, os, re, time, urllib.request
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "results" / "egpr_profile_repair" / "ml1m_2000_fair_direct_candidates"
NPZ = ROOT / "results" / "egpr_profile_repair" / "ml1m_seed42_deepseek_2000_expressive5" / "candidates_lightgcn_test.npz"
ITEM_TEXT = ROOT / "data" / "ml1m" / "item_text.jsonl"
TRAIN = ROOT / "data" / "ml1m" / "train.csv"
CACHE = ROOT / "results" / "frontier_check" / "cache"
API_URL = "https://api.deepseek.com/chat/completions"
TOPK = 20


def load_meta():
    title, genres = {}, {}
    with ITEM_TEXT.open() as f:
        for line in f:
            o = json.loads(line)
            i = int(o["iid"]); title[i] = o.get("title", f"Movie_{i}")
            genres[i] = o.get("genres", "").replace(" ", "|") or "Unknown"
    return title, genres


def load_hist():
    rows = {}
    with TRAIN.open() as f:
        for r in csv.DictReader(f):
            rows.setdefault(int(r["uid"]), []).append((int(r["timestamp"]), int(r["iid"])))
    return {u: [i for _, i in sorted(v)] for u, v in rows.items()}


def line(code, iid, title, genres):
    return f"{code}: {title.get(iid, f'Movie_{iid}')} [genres={genres.get(iid, 'Unknown')}]"


def build_prompt(uid, hist, cands, title, genres, hlim):
    recent = hist[-hlim:] if hist else []
    L = ["Task: rerank candidate movies for one MovieLens user.",
         "Use the recent history and item metadata only. Return JSON only.",
         'Output schema: {"ranking":["C01","C02",...]} with exactly 20 unique candidate IDs.',
         f"User: U{uid}", "Recent history:"]
    L += ["None"] if not len(recent) else [line(f"H{p:02d}", i, title, genres) for p, i in enumerate(recent, 1)]
    L.append("Candidates:")
    L += [line(f"C{p:02d}", int(i), title, genres) for p, i in enumerate(cands, 1)]
    return "\n".join(L)


def call(model, prompt, key, timeout=180, retries=3):
    body = json.dumps({"model": model, "temperature": 0.0, "max_tokens": 3000,
                       "response_format": {"type": "json_object"},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(API_URL, data=body, method="POST",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=timeout) as r:
                o = json.load(r)
            o["_latency_seconds"] = time.time() - t0
            return o
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise last


def parse_ranking(text, n_cand):
    codes = re.findall(r"C(\d{1,3})", text or "")
    seen, order = set(), []
    for c in codes:
        j = int(c) - 1
        if 0 <= j < n_cand and j not in seen:
            seen.add(j); order.append(j)
    return order


def ndcg(ranked_iids, target):
    pos = np.where(np.asarray(ranked_iids[:TOPK]) == int(target))[0]
    return 1.0 / math.log2(int(pos[0]) + 2) if len(pos) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("--max-users", type=int, default=150)
    ap.add_argument("--cand-width", type=int, default=50)
    ap.add_argument("--hist-limit", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    key = os.environ["DEEPSEEK_API_KEY"]
    (CACHE / args.model).mkdir(parents=True, exist_ok=True)

    title, genres = load_meta(); hist = load_hist()
    z = np.load(NPZ)
    users = z["users"].astype(int); cands = z["candidates"].astype(int)[:, :args.cand_width]
    tgts = z["targets"].astype(int)
    n = min(args.max_users, len(users))

    from concurrent.futures import ThreadPoolExecutor

    def fetch(row):
        uid = int(users[row]); cand = cands[row]
        cf = CACHE / args.model / f"{uid}.json"
        if cf.exists():
            return row, json.loads(cf.read_text())
        prompt = build_prompt(uid, hist.get(uid, []), cand, title, genres, args.hist_limit)
        o = call(args.model, prompt, key)
        cf.write_text(json.dumps(o))
        return row, o

    results = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for row, o in ex.map(fetch, range(n)):
            results[row] = o
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{n} fetched", flush=True)

    nd, in_tok, out_tok, lat = [], [], [], []
    for row in range(n):
        cand = cands[row]
        o = results[row]
        txt = o["choices"][0]["message"]["content"]
        order = parse_ranking(txt, len(cand))
        ranked = [int(cand[j]) for j in order]
        ranked += [int(c) for c in cand if int(c) not in set(ranked)]
        nd.append(ndcg(ranked, tgts[row]))
        u = o.get("usage", {})
        in_tok.append(u.get("prompt_tokens", 0)); out_tok.append(u.get("completion_tokens", 0))
        lat.append(o.get("_latency_seconds", 0.0))

    nd = np.array(nd)
    np.save(CACHE / f"ndcg_{args.model}_{n}.npy", nd)
    np.save(CACHE / f"uids_{n}.npy", users[:n])
    print(f"\nMODEL={args.model} users={n} NDCG@20={nd.mean():.4f} "
          f"prompt_tok={np.mean(in_tok):.0f} compl_tok={np.mean(out_tok):.0f} lat={np.mean(lat):.2f}s "
          f"total_tokens={np.sum(in_tok)+np.sum(out_tok)}")


if __name__ == "__main__":
    main()
