#!/usr/bin/env bash
# R1 revision: export per-user rankings and run the selector / routing analyses.
# No LLM API calls. Run from the repository root:
#   bash scripts/run_revision_r1.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RES=$ROOT/results/egpr_profile_repair
OUT=$ROOT/results/revision_r1
mkdir -p "$OUT"
cd "$ROOT"

# 1. Re-run the fair confirmatory analyses so that they also save per-user
#    rankings (per_user_ranked_top{50,100}.npz). Profile scores are recomputed
#    from cached claims; Direct rankings are read from disk.
python scripts/analyze_yelp_1000_fair_confirmatory.py --outdir "$OUT/yelp_fair" 2>&1 | tee "$OUT/log_yelp_fair.txt"
python scripts/analyze_amazon_books_fair_confirmatory.py --outdir "$OUT/books_fair" 2>&1 | tee "$OUT/log_books_fair.txt"

# 2. Strong non-LLM rerankers with per-user rankings (per_user_ranked_{bm25,dense}.npz).
python scripts/run_reranking_baselines_strong.py \
  --run-dir "$ROOT/results/llm_selective/ml1m_seed42_deepseek_2000_h20" \
  --item-text data/ml1m/item_text.jsonl --train data/ml1m/train.csv \
  --text-fields title genres --baselines overlap bm25 dense \
  --outdir "$OUT/ml1m_strong" \
  2>&1 | tee "$OUT/log_ml1m_strong.txt"

python scripts/run_reranking_baselines_strong.py \
  --run-dir "$RES/amazon_books_seed42_lightgcn_1000" \
  --item-text data/amazon_books_subset/item_metadata.jsonl --train data/amazon_books_subset/train.csv \
  --text-fields title categories description --baselines overlap bm25 dense \
  --outdir "$OUT/books_strong" \
  2>&1 | tee "$OUT/log_books_strong.txt"

# 3. Assemble matched per-user panels and run the analyses.
python scripts/export_revision_panels.py --root "$ROOT" 2>&1 | tee "$OUT/log_export.txt"
python scripts/alis_revision_analysis.py --panel-dir "$OUT/panels" --outdir "$OUT/analysis" \
  2>&1 | tee "$OUT/log_analysis.txt"

echo "Done. Copy $OUT back to the local machine (panels/, analysis/, log_*.txt)."
