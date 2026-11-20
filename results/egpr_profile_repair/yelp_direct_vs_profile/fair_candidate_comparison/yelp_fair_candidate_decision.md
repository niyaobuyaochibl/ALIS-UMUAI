# Yelp Fair Candidate-Set Comparison

Same 300 Yelp test users. Direct rerank and profile rerank are compared under matched LightGCN candidate-set sizes. Profile generation is unchanged across candidate sizes because profiles are generated once from user history and reused locally.

## Main Fair Comparison

| CandidateSet | Method | NDCG@20 | Recall@20 | HarmRate | GHR | TestCostUSD | CostVsDirect | ParseFailures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | LightGCN | 0.011630 | 0.026667 | 0.000000 | - | 0.000000 | 0.000000 | 0 |
| 50 | DeepSeek Direct Rerank top-50 | 0.007004 | 0.016667 | 0.013333 | 0.303181 | 0.103597 | 1.000000 | 0 |
| 50 | Profile Rerank EGPR top-50 | 0.011856 | 0.026667 | 0.003333 | 1.471721 | 0.043334 | 0.418292 | 0 |
| 100 | LightGCN | 0.011630 | 0.026667 | 0.000000 | - | 0.000000 | 0.000000 | 0 |
| 100 | DeepSeek Direct Rerank top-100 | 0.008604 | 0.020000 | 0.013333 | 0.593152 | 0.182703 | 1.000000 | 0 |
| 100 | Profile Rerank EGPR top-100 | 0.011856 | 0.026667 | 0.003333 | 1.471721 | 0.043334 | 0.237182 | 0 |

## Full Method Table

| CandidateSet | Method | NDCG@20 | Recall@20 | HarmRate | GHR | TestCostUSD | CostVsDirect | ParseFailures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | LightGCN | 0.011630 | 0.026667 | 0.000000 | - | 0.000000 | 0.000000 | 0 |
| 50 | DeepSeek Direct Rerank top-50 | 0.007004 | 0.016667 | 0.013333 | 0.303181 | 0.103597 | 1.000000 | 0 |
| 50 | Profile Rerank Raw top-50 | 0.011056 | 0.023333 | 0.006667 | 0.541589 | 0.043334 | 0.418292 | 0 |
| 50 | Profile Rerank Remove top-50 | 0.011159 | 0.023333 | 0.006667 | 0.623274 | 0.043334 | 0.418292 | 0 |
| 50 | Profile Rerank EGPR top-50 | 0.011856 | 0.026667 | 0.003333 | 1.471721 | 0.043334 | 0.418292 | 0 |
| 100 | LightGCN | 0.011630 | 0.026667 | 0.000000 | - | 0.000000 | 0.000000 | 0 |
| 100 | DeepSeek Direct Rerank top-100 | 0.008604 | 0.020000 | 0.013333 | 0.593152 | 0.182703 | 1.000000 | 0 |
| 100 | Profile Rerank Raw top-100 | 0.011096 | 0.023333 | 0.006667 | 0.573482 | 0.043334 | 0.237182 | 0 |
| 100 | Profile Rerank Remove top-100 | 0.011096 | 0.023333 | 0.006667 | 0.573482 | 0.043334 | 0.237182 | 0 |
| 100 | Profile Rerank EGPR top-100 | 0.011856 | 0.026667 | 0.003333 | 1.471721 | 0.043334 | 0.237182 | 0 |

## Interpretation

- Under matched top-50 candidates, Profile Rerank EGPR remains above LightGCN and far above Direct DeepSeek rerank.
- Under matched top-100 candidates, Direct DeepSeek improves over its top-50 version but remains below LightGCN and below Profile Rerank EGPR.
- Profile reranking is cheaper in both matched settings because the LLM sees only user history once: 0.418x direct cost for top-50 and 0.237x direct cost for top-100.
- This removes the candidate-set fairness concern for the Yelp result.

## Cost Notes

- Profile test-user generation cost: 0.043334 USD.
- Direct top-50 rerank cost: 0.103597 USD.
- Direct top-100 rerank cost: 0.182703 USD.
- Full profile run cost including validation users: 0.086473 USD.

## Artifacts

- `results/egpr_profile_repair/yelp_direct_vs_profile/fair_candidate_comparison/yelp_fair_candidate_summary.csv`
- `results/egpr_profile_repair/yelp_direct_vs_profile/direct_rerank_300/yelp_direct_rerank_report.md`
- `results/egpr_profile_repair/yelp_direct_vs_profile/direct_rerank_300_top100/yelp_direct_rerank_report.md`
- `results/egpr_profile_repair/yelp_direct_vs_profile/profile_candidate_sizes/profile_candidate_size_report.md`
