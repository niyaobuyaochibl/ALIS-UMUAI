#!/usr/bin/env python3
"""R1 point 5: ALIS sensitivity to harm/disruption weights, interface-pool
composition (min-max normalization), and profile reuse horizon.
No LLM calls; reuses the per-user panels and scoring of alis_revision_analysis.py.

Usage:
  python scripts/alis_weight_reuse_sensitivity.py --panel-dir results/revision_r1/panels
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from alis_revision_analysis import PREFERENCES, PROFILE_COST, Panel, alis_scores

PROFILE = {"Raw", "Remove", "EGPR"}
HARM_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
DISRUPTION_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
REUSE = [1, 2, 5, 10, 20, 50]


def select(P: Panel, w: Dict[str, float], pool: np.ndarray | None = None) -> str:
    idx = np.arange(P.n)
    if pool is None:
        return P.ifaces[int(np.argmax(alis_scores(P, idx, w)[0]))]
    sub = Panel.__new__(Panel)
    sub.__dict__.update(P.__dict__)
    sub.ifaces = [P.ifaces[j] for j in pool]
    sub.Y, sub.OV = P.Y[:, pool], P.OV[:, pool]
    sub.C, sub.G, sub.paid = P.C[pool], P.G[pool], P.paid[pool]
    sub.M = len(pool)
    sub.base = sub.ifaces.index("Base")
    return sub.ifaces[int(np.argmax(alis_scores(sub, idx, w)[0]))]


def harm_disruption_sweep(P: Panel) -> pd.DataFrame:
    rows = []
    for wh, wd in itertools.product(HARM_GRID, DISRUPTION_GRID):
        w = dict(PREFERENCES["balanced"], h=wh, d=wd)
        rows.append({"domain": P.domain, "w_h": wh, "w_d": wd, "selected": select(P, w)})
    return pd.DataFrame(rows)


def pool_leave_one_out(P: Panel) -> pd.DataFrame:
    rows = []
    for pref, w in PREFERENCES.items():
        full = select(P, w)
        for drop in P.ifaces:
            if drop in ("Base", full):
                continue
            pool = np.array([j for j, m in enumerate(P.ifaces) if m != drop])
            rows.append({"domain": P.domain, "preference": pref, "full_pool_choice": full,
                         "dropped": drop, "selected": select(P, w, pool)})
    return pd.DataFrame(rows)


def reuse_horizon(P: Panel) -> pd.DataFrame:
    """Profile cost per request relative to Direct when one profile serves R requests."""
    rows = []
    base_cost = P.C.copy()
    for R in REUSE:
        P.C = np.array([PROFILE_COST[P.domain] / R if m in PROFILE else c
                        for m, c in zip(P.ifaces, base_cost)])
        for pref, w in PREFERENCES.items():
            rows.append({"domain": P.domain, "reuse_R": R,
                         "profile_cost_ratio": PROFILE_COST[P.domain] / R,
                         "preference": pref, "selected": select(P, w)})
    P.C = base_cost
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-dir", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()
    outdir = args.outdir or args.panel_dir.parent / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)

    frames: Dict[str, List[pd.DataFrame]] = {"retro": [], "hd": [], "pool": [], "reuse": []}
    for dom in ("ml1m", "yelp", "books"):
        P = Panel(dom, pd.read_csv(args.panel_dir / f"panel_{dom}.csv"))
        frames["retro"].append(pd.DataFrame(
            [{"domain": dom, "preference": p, "selected": select(P, w)} for p, w in PREFERENCES.items()]))
        frames["hd"].append(harm_disruption_sweep(P))
        frames["pool"].append(pool_leave_one_out(P))
        frames["reuse"].append(reuse_horizon(P))

    retro = pd.concat(frames["retro"], ignore_index=True)
    hd = pd.concat(frames["hd"], ignore_index=True)
    pool = pd.concat(frames["pool"], ignore_index=True)
    reuse = pd.concat(frames["reuse"], ignore_index=True)
    retro.to_csv(outdir / "weight_retro_check.csv", index=False)
    hd.to_csv(outdir / "weight_harm_disruption_sweep.csv", index=False)
    pool.to_csv(outdir / "weight_pool_leave_one_out.csv", index=False)
    reuse.to_csv(outdir / "weight_reuse_horizon.csv", index=False)

    print("== retrospective choice on panel ==")
    print(retro.pivot(index="domain", columns="preference", values="selected"))
    print(f"\n== harm x disruption sweep ({len(HARM_GRID) * len(DISRUPTION_GRID)} points, balanced otherwise) ==")
    print(hd.groupby(["domain", "selected"]).size().unstack(fill_value=0))
    print("\n== pool leave-one-out: cells whose choice changes ==")
    changed = pool[pool["selected"] != pool["full_pool_choice"]]
    print(f"{len(changed)} of {len(pool)} leave-one-out cells change")
    print(changed.to_string(index=False) if len(changed) else "(none)")
    print("\n== reuse horizon ==")
    print(reuse.pivot_table(index=["domain", "reuse_R"], columns="preference",
                            values="selected", aggfunc="first"))


if __name__ == "__main__":
    main()
