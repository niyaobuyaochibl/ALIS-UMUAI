# Yelp Direct Rerank vs Profile Rerank Decision

Dataset: Yelp. Baseline: LightGCN seed42. Same 300 test users from the Yelp expressive profile-rerank pilot. Direct rerank uses DeepSeek over LightGCN top-50. Profile rerank uses DeepSeek-generated expressive profiles over LightGCN top-100 with local profile-item scoring.

## Same-Sample Results

| Method | NDCG@20 | Recall@20 | HarmRate | GHR | TestCostUSD | CostVsDirect |
| --- | --- | --- | --- | --- | --- | --- |
| LightGCN | 0.011630 | 0.026667 | 0.000000 | - | 0.000000 | 0.000000 |
| DeepSeek Direct Rerank | 0.007004 | 0.016667 | 0.013333 | 0.303181 | 0.103597 | 1.000000 |
| Profile Rerank Expressive Raw Profile | 0.011096 | 0.023333 | 0.006667 | 0.573482 | 0.043334 | 0.418292 |
| Profile Rerank Expressive Remove Repair | 0.011096 | 0.023333 | 0.006667 | 0.573482 | 0.043334 | 0.418292 |
| Profile Rerank Expressive EGPR | 0.011856 | 0.026667 | 0.003333 | 1.471721 | 0.043334 | 0.418292 |

## Profile Faithfulness

| Method | Claims | UCR | WeightedUCR | EvidenceCoverage | ProfileDriftScore |
| --- | --- | --- | --- | --- | --- |
| Raw Profile | 2995 | 0.090484 | 0.090484 | 0.830950 | 0.657739 |
| Remove Repair | 2724 | 0.000000 | 0.000000 | 0.830950 | 0.655330 |
| Evidence-Weighted Repair | 2995 | 0.090484 | 0.011198 | 0.830950 | 0.644376 |

## Cost Notes

- Direct rerank test cost: 0.103597 USD for 300 users.
- Profile generation cost for the same 300 test users: 0.043334 USD (0.418x direct).
- Full profile run cost including validation users: 0.086473 USD (0.835x direct), but profiles are reusable for local reranking and auditing.

## Interpretation

- Yelp differs from ML-1M: raw expressive profile reranking slightly hurts LightGCN, while evidence-weighted EGPR recovers utility and ends slightly above the base.
- Direct DeepSeek reranking is clearly harmful on this Yelp sample: lower NDCG/Recall than LightGCN and the weakest GHR.
- Evidence-weighted profile reranking is the best Yelp setting among the tested methods: highest NDCG, lowest harm, and substantially lower test-user API cost than direct reranking.
- This supports the revised paper framing: profile-before-ranking is a cheaper and safer interface than direct LLM candidate reranking, and EGPR is useful as a safety layer when expressive profiles drift.

## Artifacts

- `results/egpr_profile_repair/yelp_seed42_deepseek_300_expressive5/yelp_profile_rerank_report.md`
- `results/egpr_profile_repair/yelp_direct_vs_profile/direct_rerank_300/yelp_direct_rerank_report.md`
- `results/egpr_profile_repair/yelp_direct_vs_profile/yelp_direct_vs_profile_summary.csv`
- `scripts/run_yelp_profile_rerank_pilot.py`
- `scripts/run_yelp_direct_rerank_pilot.py`
