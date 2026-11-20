#!/usr/bin/env python3
"""Generate rq5_table.tex from the cross-provider LLMRank result CSVs.

Reads summary + reliability CSVs for ML-1M and Amazon Books, for Base / Content /
LLMRank(DeepSeek) / LLMRank(Moonshot), and writes a fully-populated LaTeX table.
Missing sources are rendered as \\textit{tbd} so the document still compiles.
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "egpr_profile_repair"
ML = RES / "ml1m_2000_fair_direct_candidates"
AMZ = RES / "amazon_books_direct_vs_profile"
OUT = ROOT / "results" / "egpr_profile_repair" / "rq5_table.tex"

TBD = r"\textit{tbd}"


def _rows(path: Path):
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def metric(summary: Path, reliability: Path, method_is_base: bool):
    """Return (ndcg, recall, harm) strings for the relevant row.

    method_is_base=True -> use the LightGCN row and harm '--'.
    Otherwise use the last (non-LightGCN) row of each file.
    """
    srows = _rows(summary)
    rrows = _rows(reliability)
    if not srows:
        return TBD, TBD, (r"--" if method_is_base else TBD)
    if method_is_base:
        row = srows[0]
        ndcg = f"{float(row['NDCG@20']):.4f}"
        recall = f"{float(row['Recall@20']):.4f}"
        return ndcg, recall, r"--"
    row = srows[-1]
    ndcg = f"{float(row['NDCG@20']):.4f}"
    recall = f"{float(row['Recall@20']):.4f}"
    harm = TBD
    if rrows:
        harm = f"{float(rrows[-1]['HarmRate']):.3f}"
    return ndcg, recall, harm


def main() -> None:
    ml_deepseek = ML / "reranking_baselines_deepseek_llmrank_1000"
    ml_moonshot = ML / "reranking_baselines_moonshot_llmrank_1000"
    ml_content = ML / "reranking_baselines_content_1000"

    amz_deepseek = AMZ / "reranking_baselines_llmrank_1000"
    amz_moonshot = AMZ / "reranking_baselines_moonshot_llmrank_1000"
    amz_content = AMZ / "reranking_baselines_1000_top100"

    # ML-1M rows
    ml_base = metric(ml_deepseek / "reranking_baselines_summary.csv",
                     ml_deepseek / "reranking_baselines_reliability.csv", True)
    ml_cont = metric(ml_content / "reranking_baselines_summary.csv",
                     ml_content / "reranking_baselines_reliability.csv", False)
    ml_ds = metric(ml_deepseek / "reranking_baselines_summary.csv",
                   ml_deepseek / "reranking_baselines_reliability.csv", False)
    ml_ms = metric(ml_moonshot / "reranking_baselines_summary.csv",
                   ml_moonshot / "reranking_baselines_reliability.csv", False)

    # Books rows
    amz_base = metric(amz_deepseek / "reranking_baselines_summary.csv",
                      amz_deepseek / "reranking_baselines_reliability.csv", True)
    amz_cont = metric(amz_content / "reranking_baselines_summary.csv",
                      amz_content / "reranking_baselines_reliability.csv", False)
    amz_ds = metric(amz_deepseek / "reranking_baselines_summary.csv",
                    amz_deepseek / "reranking_baselines_reliability.csv", False)
    amz_ms = metric(amz_moonshot / "reranking_baselines_summary.csv",
                    amz_moonshot / "reranking_baselines_reliability.csv", False)

    def line(method, m):
        return f" & {method:<19}& {m[0]} & {m[1]} & {m[2]} \\\\"

    body = []
    body.append(r"\begin{table}[!htbp]")
    body.append(r"\centering")
    body.append(r"\caption{RQ5: external listwise reranker (LLMRank) under two independent LLM "
                r"providers, against \textsc{Base} and the zero-API \textsc{Content} reranker, on "
                r"matched users and candidate sets. Under both providers the LLM reranker does not "
                r"hurt accuracy and the free \textsc{Content} reranker stays competitive with it; "
                r"which provider gives the larger gain is domain-dependent.}")
    body.append(r"\label{tab:crossprovider}")
    body.append(r"\small")
    body.append(r"\begin{tabular}{llrrr}")
    body.append(r"\toprule")
    body.append(r"Dataset & Method & NDCG@20 & Recall@20 & HarmRate $\downarrow$ \\")
    body.append(r"\midrule")
    body.append(r"\multirow{4}{*}{ML-1M}")
    body.append(line("Base", ml_base))
    body.append(line("Content", ml_cont))
    body.append(line("LLMRank (DeepSeek)", ml_ds))
    body.append(line("LLMRank (Moonshot)", ml_ms))
    body.append(r"\midrule")
    body.append(r"\multirow{4}{*}{Books}")
    body.append(line("Base", amz_base))
    body.append(line("Content", amz_cont))
    body.append(line("LLMRank (DeepSeek)", amz_ds))
    body.append(line("LLMRank (Moonshot)", amz_ms))
    body.append(r"\bottomrule")
    body.append(r"\end{tabular}")
    body.append(r"\end{table}")
    OUT.write_text("\n".join(body) + "\n")

    print("Wrote", OUT)
    print("ML-1M  base=", ml_base, "content=", ml_cont, "ds=", ml_ds, "ms=", ml_ms)
    print("Books  base=", amz_base, "content=", amz_cont, "ds=", amz_ds, "ms=", amz_ms)


if __name__ == "__main__":
    main()
