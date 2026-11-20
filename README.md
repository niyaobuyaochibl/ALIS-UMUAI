# ALIS: Adaptive Selection of LLM Knowledge Interfaces for User Modeling in Recommender Systems

Code, generated user-model artifacts, logs and result tables for

> Yunan Zhang, Jingjing Fan, Yanxiao Liu. *ALIS: Adaptive Selection of LLM Knowledge Interfaces
> for User Modeling in Recommender Systems.* User Modeling and User-Adapted Interaction, 2026.

ALIS decides, per domain and deployment preference, whether an LLM-derived user model is worth
paying for and through which interface: Direct LLM reranking, a cached LLM profile (Raw, Remove
repair, or evidence-governed repair, EGPR), or one of three free non-LLM rerankers (Content, BM25,
Dense). The experiments use MovieLens-1M, Yelp and Amazon Books.

## Contents

```
scripts/            experiment, analysis and data-preparation scripts
  amazon_upstream/  the two steps that build the Amazon Books split from the public review dump
src/temporal_popularity/   LightGCN/SASRec training, splits, evaluation (used by the scripts)
configs/formal/     base-recommender configurations (LightGCN, ML-1M and Yelp)
results/            all outputs used in the paper (see the mapping below)
  egpr_profile_repair/     profile, Direct and reranker runs, ALIS selection
  revision_r1/             per-user panels, selection stability, routing, claim audit, judge calibration
  llm_selective/           ML-1M Direct reranking run (2,000 users)
  frontier_check/          deepseek-v4-flash vs deepseek-v4-pro check (Section 8)
  tallrec_baseline/        LLaMA-7B + Alpaca-LoRA diagnostic
  egpr_profile_repair_c1/  Kimi profile-generator diagnostic
  formal/                  base-recommender configs and manifests (checkpoints not included)
  stdout_tables/           printed output of the two scripts that write no files (Tables 9 and 14)
review_packages/    the two packages seen by the reviewers, unchanged
```

Each experiment run directory contains `run_manifest.json` with the model, prompt variant, sample
sizes and arguments of that run.

## Installation

The experiments ran with Python 3.12; the re-runs described below used Python 3.13.

```bash
pip install -r requirements.txt
```

Run all commands from the repository root. Scripts add `src/` and `scripts/` to the import path
themselves.

## Reproducing the paper from the released files

The following need neither raw data nor API access and run on a CPU. Each regenerates files that
are already in `results/`, so redirect `--outdir` (where available) to compare.

| Paper | Command | Output |
|---|---|---|
| Tables 3, 4; Figs. 3, 4 | `python scripts/run_adaptive_interface_selector.py` | `results/egpr_profile_repair/adaptive_interface_selector/` |
| Tables 5, 6, 15 | `python scripts/alis_revision_analysis.py --panel-dir results/revision_r1/panels --outdir <dir>` (about 30 min) | `results/revision_r1/analysis/` |
| Table 7 | `python scripts/alis_weight_reuse_sensitivity.py --panel-dir results/revision_r1/panels --outdir <dir>` | `results/revision_r1/analysis/weight_*.csv` |
| Table 12 | `python scripts/analyze_llm_claim_audit.py` | `results/revision_r1/claim_audit/` |
| Judge calibration (Section 6.4) | `python scripts/analyze_judge_calibration.py` | `results/revision_r1/judge_calibration/` |
| Table 13 | `python scripts/finalize_rq5_table.py` | `results/egpr_profile_repair/rq5_table.tex` |

We re-ran these commands on a clean copy of this repository. The outputs are identical to the
released files, except for one row of Table 15 noted under [Known differences](#known-differences),
and `finalize_rq5_table.py` reproduces the values of Table 13. With the prepared ML-1M and Yelp
data in `data/`, `predictive_alis.py` reproduces Table 14 exactly.

## Where each result comes from

| Paper | Files |
|---|---|
| Table 1 (preference vectors) | `PREFERENCE_PROFILES` in `scripts/run_adaptive_interface_selector.py` |
| Table 2 (settings) | `configs/formal/`, `results/formal/*/seed42/config.json`, `run_manifest.json` of each run |
| Table 3, Fig. 3 | `adaptive_interface_selector/alis_selected_interfaces.csv`, `alis_sensitivity_grid.csv` |
| Table 4 | `adaptive_interface_selector/alis_sensitivity_grid.csv` |
| Table 5 | `revision_r1/analysis/oos_summary.csv`, `oos_raw.csv`, `bootstrap_selection.csv` |
| Table 6 | `revision_r1/analysis/paired_tests.csv` |
| Table 7 | `revision_r1/analysis/weight_reuse_horizon.csv` (also `weight_harm_disruption_sweep.csv`, `weight_pool_leave_one_out.csv`) |
| Table 8, Fig. 4 | `adaptive_interface_selector/candidate_interface_metrics.csv`, which `run_adaptive_interface_selector.py` assembles from: ML-1M `egpr_profile_repair/ml1m_2000_fair_direct_candidates/fair_direct_candidate_summary.csv`, `rank_disruption/`, `reranking_baselines/` and `results/ml1m_2000_fair_direct_candidates/reranking_baselines_strong/`; Yelp `egpr_profile_repair/yelp_direct_vs_profile/yelp_1000_fair_confirmatory/` and `reranking_baselines/`; Amazon Books `egpr_profile_repair/amazon_books_direct_vs_profile/amazon_books_1000_fair_confirmatory/`, `reranking_baselines_1000_top100/` and `reranking_baselines_strong/` |
| Table 9 | `scripts/compute_ci_ml1m.py`; printed output in `stdout_tables/compute_ci_ml1m_output.txt` |
| Table 10 | logged token counts and latencies: `profile_cost_trace.csv` (profile runs), `direct_rerank_cost_trace.csv` (Yelp, Books), `llm_selective/ml1m_seed42_deepseek_2000_h20/cost_trace_lightgcn_original.csv` (ML-1M Direct) |
| Table 11 | `table1_profile_faithfulness.csv` in `ml1m_seed42_deepseek_2000_expressive5/`, `yelp_seed42_deepseek_1000_expressive5/`, `amazon_books_seed42_deepseek_1000_expressive5/` |
| Table 12 | `revision_r1/claim_audit/stratum_profile.csv`, `agreement.csv`, `domain_summary.csv` |
| Section 6.5 (SASRec) | `egpr_profile_repair/ml1m_seed42_sasrec_deepseek_2000_expressive5/sasrec_profile_paired_stats.csv` |
| Table 13 | `reranking_baselines_{content_1000,deepseek_llmrank_1000,moonshot_llmrank_1000}/` (ML-1M) and `reranking_baselines_{1000_top100,llmrank_1000,moonshot_llmrank_1000}/` (Books) |
| Table 14, Fig. 5 | `scripts/predictive_alis.py`; printed output in `stdout_tables/predictive_alis_output.txt` |
| Table 15 | `revision_r1/analysis/routing_nested_cv.csv`, `routing_hyperparameters.csv` |
| Section 8, stronger model | `frontier_check/frontier_summary.md`, `frontier_check/cache/` |

Paths in the second column are relative to `results/` or `results/egpr_profile_repair/`.

## Generated user models

For each domain the profile runs contain

- `raw_profiles.jsonl`: the LLM-generated profile of each user, as a list of claims;
- `claim_support.jsonl`: each claim with its type, confidence, support count, support score,
  EGPR weight and status;
- `repaired_profiles.jsonl`: the profile after Remove repair and after EGPR;
- `table1_profile_faithfulness.csv` ... `table4_lambda_validation.csv`: run-level summaries.

Profile prompts are defined in the profile scripts (`run_egpr_profile_repair_pilot.py`,
`run_yelp_profile_rerank_pilot.py`, `run_amazon_books_profile_rerank_pilot.py`); the Direct
reranking prompt in `run_llm_selective_invocation_pilot.py` and the Direct scripts; the claim-audit
judge prompt in `run_llm_claim_audit.py`.

## What is not included

- **Raw user histories and the prepared splits** (`data/`). The datasets are redistributed only by
  their providers. Instructions to rebuild them are below.
- **`user_history.jsonl`** of the profile runs: the recent interactions shown to the profile
  generator, which are rows of the raw datasets.
- **The `supporting_items` field of `claim_support.jsonl`**: the history items matched to each
  claim. All other fields are released.
- **LLM request caches** (`*_api_cache/`, `profile_api_cache/`): every cached prompt contains a
  user's history. The exception is `frontier_check/cache/`, which holds responses only, with the
  model's reasoning trace removed because it restates the history; the script reads only the final
  ranking, so the check can be recomputed from it.
- **Model checkpoints and embedding matrices** (LightGCN, SASRec, LoRA weights). They are learned
  from the user interactions and can be retrained with the commands below.
- **LLM-AggreFact claims and documents**: `calibration_sample.csv` identifies each sampled claim by
  `<dataset>:<row of the test split>`; the benchmark is available from its authors
  (`lytang/LLM-AggreFact` on Hugging Face) under their access conditions.

## Rebuilding the data and re-running the experiments

Download the public datasets and place them as follows.

| Dataset | Files | Location |
|---|---|---|
| MovieLens-1M (GroupLens) | `ratings.dat`, `movies.dat` | `data/raw/ml-1m/` |
| Yelp Open Dataset | `yelp_academic_dataset_review.json`, `yelp_academic_dataset_business.json` | `data/raw/yelp/` |
| Amazon review data (2014), Books | `reviews_Books_5.json.gz`, `meta_Books.json.gz` | `data/raw/amazon-books/` |

```bash
# ML-1M: 10-core filtering and leave-one-out split, then item text
python scripts/run_ml1m_minipilot.py          # writes data/ml1m before its pilot training starts
python scripts/build_item_text.py ml1m

# Yelp
python scripts/run_yelp_day1_drift.py
python scripts/build_item_text.py yelp

# Amazon Books (the subset uses a fixed seed)
python scripts/amazon_upstream/preprocess_amazon.py --raw_dir data/raw/amazon-books --output_dir data/raw/amazon-books-processed
python scripts/amazon_upstream/create_amazon_subset.py --input data/raw/amazon-books-processed --output data/raw/amazon-books-subset
python scripts/prepare_amazon_books_subset.py

# Base recommenders
python scripts/prepare_formal_runs.py
python scripts/run_formal_seed.py results/formal/ml1m/seed42/config.json
python scripts/run_formal_seed.py results/formal/yelp/seed42/config.json
python scripts/run_formal_sasrec_audit.py results/formal/ml1m/seed42/config.json --outdir results/formal/backbone_robustness/ml1m/seed42/sasrec
python scripts/train_amazon_books_lightgcn_candidates.py
```

We checked the ML-1M steps against the original server copy: starting from the GroupLens
`ml-1m.zip`, the splits, `mappings.json` and `item_text.jsonl` are identical. The Yelp and Amazon
Books steps could not be checked this way because we no longer hold those dumps. Yelp item text
exists only for the 20,106 items listed in `configs/yelp_item_text_iids.txt`, the items whose
business record the Yelp profile run looked up; this is why the paper reports partial text
coverage for Yelp.

LLM runs call OpenAI-compatible chat APIs. Keys are read from the environment
(`DEEPSEEK_API_KEY`, `MOONSHOT_API_KEY`, `OPENROUTER_API_KEY`) and are not stored in this repository.
The arguments of each run are in its `run_manifest.json`; `scripts/run_ml1m_confirmatory_2000.sh`
and `scripts/run_revision_r1.sh` chain the ML-1M runs and the revision analyses. LLM outputs vary
between calls and over time even at temperature 0, so re-generated profiles and rankings will
differ from the released ones; the released files are the ones the paper reports.

## Known differences

- **ML-1M BM25 and Dense in Tables 8 and 9.** These rows come from
  `results/ml1m_2000_fair_direct_candidates/reranking_baselines_strong/` (NDCG@20 0.0282 and 0.0319).
  The same rerankers were later re-run with per-user rankings for the revision panels
  (`results/revision_r1/ml1m_strong/`, 0.0284 and 0.0320), and these are the values the released code
  reproduces, also in `stdout_tables/compute_ci_ml1m_output.txt` (Dense vs. Base
  \+7.18 [+2.66, +11.81] instead of +7.00 [+2.40, +11.70]; Dense vs. Direct +0.84 [−2.93, +4.68]
  instead of +0.66 [−3.08, +4.43], all ×10⁻³). No conclusion changes. The selection analyses
  (Tables 3–7, 15) use the revision panels.
- **ℓ1-logistic router in Table 15.** With scikit-learn 1.9 this row differs from the released
  `routing_nested_cv.csv` by up to 0.7×10⁻³ NDCG@20 and 0.05 in paid-call rate, because the ℓ1
  solver path changed between versions; all other routers and all other analyses reproduce exactly.
- **Server paths.** The experiments ran under `/root/temporal_popularity_pilot`; in this release the
  hard-coded root and raw-data paths point to the repository instead. The scripts are otherwise
  unchanged; the reviewed versions are in `review_packages/`. In addition, `predictive_alis.py` and
  `run_adaptive_interface_selector.py` are newer than the reviewed versions: they add the BM25 and
  Dense rerankers reported in the final paper.

## Licence

Code: MIT (see `LICENSE`). Generated artifacts and results: CC BY 4.0. The artifacts are derived
from MovieLens-1M, the Amazon review data and the Yelp Open Dataset; their use is also subject to
those datasets' terms.

## Citation

See `CITATION.cff`.
