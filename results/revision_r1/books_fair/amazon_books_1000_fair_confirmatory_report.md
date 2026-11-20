# Amazon Books 1000 Fair Confirmatory Analysis

Profile run: `<repo>\results\egpr_profile_repair\amazon_books_seed42_deepseek_1000_expressive5`.
Direct rerank and profile rerank are evaluated on matched LightGCN top-50 and top-100 candidate sets.
Profile generation cost is reusable and fixed across candidate sizes: `0.168051` USD.

## Fair Accuracy / Reliability / Cost

| CandidateSet | Method | NDCG@20 | Recall@20 | DeltaVsBase | HarmRate | GHR | TestCostUSD | CostVsDirect |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | LightGCN | 0.004263 | 0.011000 | 0.000000 | 0.000000 | - | 0.000000 | 0.000000 |
| 50 | DeepSeek Direct Rerank | 0.003245 | 0.009000 | -0.001018 | 0.009000 | 0.474602 | 0.165065 | 1.000000 |
| 50 | Profile Rerank Raw | 0.004664 | 0.012000 | 0.000400 | 0.004000 | 15.691483 | 0.168051 | 1.018089 |
| 50 | Profile Rerank Remove | 0.004634 | 0.012000 | 0.000371 | 0.002000 | 6.009568 | 0.168051 | 1.018089 |
| 50 | Profile Rerank EGPR | 0.004259 | 0.011000 | -0.000005 | 0.001000 | 0.000000 | 0.168051 | 1.018089 |
| 100 | LightGCN | 0.004263 | 0.011000 | 0.000000 | 0.000000 | - | 0.000000 | 0.000000 |
| 100 | DeepSeek Direct Rerank | 0.003946 | 0.012000 | -0.000317 | 0.007000 | 0.846079 | 0.297325 | 1.000000 |
| 100 | Profile Rerank Raw | 0.004864 | 0.012000 | 0.000600 | 0.005000 | 2.872237 | 0.168051 | 0.565208 |
| 100 | Profile Rerank Remove | 0.004975 | 0.013000 | 0.000712 | 0.002000 | 10.608022 | 0.168051 | 0.565208 |
| 100 | Profile Rerank EGPR | 0.004812 | 0.011000 | 0.000549 | 0.005000 | 1.962263 | 0.168051 | 0.565208 |

## Profile Lambda Selection

| CandidateSet | Method | SelectedLambda | ValidationNDCG@20 |
| --- | --- | --- | --- |
| 50 | Profile Rerank Raw | 0.100000 | 0.004726 |
| 50 | Profile Rerank Remove | 0.300000 | 0.004820 |
| 50 | Profile Rerank EGPR | 0.050000 | 0.004737 |
| 100 | Profile Rerank Raw | 0.500000 | 0.005060 |
| 100 | Profile Rerank Remove | 0.300000 | 0.004820 |
| 100 | Profile Rerank EGPR | 0.500000 | 0.005001 |

## Paired NDCG Tests

| CandidateSet | Comparison | Metric | Users | MeanDelta | BootstrapCI95Low | BootstrapCI95High | SignFlipPValueTwoSided | Wins | Ties | Losses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | Direct vs Base | NDCG@20 | 1000 | -0.001018 | -0.002981 | 0.000824 | 0.341658 | 3 | 988 | 9 |
| 50 | Profile Raw vs Base | NDCG@20 | 1000 | 0.000400 | -0.000029 | 0.001028 | 0.234765 | 3 | 993 | 4 |
| 50 | Profile Remove vs Base | NDCG@20 | 1000 | 0.000371 | -0.000079 | 0.001084 | 0.329670 | 3 | 995 | 2 |
| 50 | Profile EGPR vs Base | NDCG@20 | 1000 | -0.000005 | -0.000015 | 0.000000 | 1.000000 | 0 | 999 | 1 |
| 50 | Profile Raw vs Direct | NDCG@20 | 1000 | 0.001419 | -0.000576 | 0.003577 | 0.189810 | 10 | 987 | 3 |
| 50 | Profile Remove vs Direct | NDCG@20 | 1000 | 0.001390 | -0.000459 | 0.003549 | 0.181818 | 9 | 987 | 4 |
| 50 | Profile EGPR vs Direct | NDCG@20 | 1000 | 0.001014 | -0.000649 | 0.003032 | 0.321678 | 9 | 988 | 3 |
| 50 | Profile EGPR vs Raw | NDCG@20 | 1000 | -0.000405 | -0.001067 | 0.000021 | 0.251748 | 3 | 994 | 3 |
| 50 | Profile Remove vs Raw | NDCG@20 | 1000 | -0.000029 | -0.000261 | 0.000202 | 0.898102 | 5 | 992 | 3 |
| 100 | Direct vs Base | NDCG@20 | 1000 | -0.000317 | -0.003324 | 0.001989 | 0.839161 | 7 | 986 | 7 |
| 100 | Profile Raw vs Base | NDCG@20 | 1000 | 0.000600 | -0.000352 | 0.001702 | 0.208791 | 5 | 990 | 5 |
| 100 | Profile Remove vs Base | NDCG@20 | 1000 | 0.000712 | 0.000009 | 0.001713 | 0.156843 | 4 | 994 | 2 |
| 100 | Profile EGPR vs Base | NDCG@20 | 1000 | 0.000549 | -0.000716 | 0.001768 | 0.381618 | 6 | 989 | 5 |
| 100 | Profile Raw vs Direct | NDCG@20 | 1000 | 0.000917 | -0.001449 | 0.003550 | 0.545455 | 8 | 984 | 8 |
| 100 | Profile Remove vs Direct | NDCG@20 | 1000 | 0.001029 | -0.001392 | 0.003595 | 0.511489 | 9 | 984 | 7 |
| 100 | Profile EGPR vs Direct | NDCG@20 | 1000 | 0.000866 | -0.001624 | 0.003958 | 0.579421 | 7 | 985 | 8 |
| 100 | Profile EGPR vs Raw | NDCG@20 | 1000 | -0.000051 | -0.000688 | 0.000424 | 0.930070 | 3 | 995 | 2 |
| 100 | Profile Remove vs Raw | NDCG@20 | 1000 | 0.000111 | -0.000278 | 0.000663 | 0.917083 | 3 | 993 | 4 |

## Rank Disruption

| CandidateSet | Method | Users | NDCG@20 | Recall@20 | WinsVsBase | TiesVsBase | LossesVsBase | Top20OverlapVsBase | Top20JaccardVsBase | MeanAbsRankShiftVsBase | HitUsers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | LightGCN | 1000 | 0.004263 | 0.011000 | 0 | 1000 | 0 | 1.000000 | 1.000000 | 0.000000 | 11 |
| 50 | DeepSeek Direct Rerank | 1000 | 0.003245 | 0.009000 | 3 | 988 | 9 | 0.754250 | 0.659693 | 4.063116 | 9 |
| 50 | Profile Rerank Raw | 1000 | 0.004664 | 0.012000 | 3 | 993 | 4 | 0.966950 | 0.937947 | 0.637141 | 12 |
| 50 | Profile Rerank Remove | 1000 | 0.004634 | 0.012000 | 3 | 995 | 2 | 0.914100 | 0.847783 | 1.533540 | 12 |
| 50 | Profile Rerank EGPR | 1000 | 0.004259 | 0.011000 | 0 | 999 | 1 | 0.985100 | 0.971758 | 0.336402 | 11 |
| 100 | LightGCN | 1000 | 0.004263 | 0.011000 | 0 | 1000 | 0 | 1.000000 | 1.000000 | 0.000000 | 11 |
| 100 | DeepSeek Direct Rerank | 1000 | 0.003946 | 0.012000 | 7 | 986 | 7 | 0.655650 | 0.580003 | 4.239199 | 12 |
| 100 | Profile Rerank Raw | 1000 | 0.004864 | 0.012000 | 5 | 990 | 5 | 0.818650 | 0.703297 | 2.728521 | 12 |
| 100 | Profile Rerank Remove | 1000 | 0.004975 | 0.013000 | 4 | 994 | 2 | 0.910450 | 0.841983 | 1.524135 | 13 |
| 100 | Profile Rerank EGPR | 1000 | 0.004812 | 0.011000 | 6 | 989 | 5 | 0.817800 | 0.702329 | 2.772396 | 11 |

## Affected-Hit Subset

| CandidateSet | Subset | Method | Users | NDCG@20 | Recall@20 | MeanDeltaVsBase | WinsVsBase | TiesVsBase | LossesVsBase |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | any_method_hit | LightGCN | 14 | 0.304531 | 0.785714 | 0.000000 | 0 | 14 | 0 |
| 50 | any_method_hit | DeepSeek Direct Rerank | 14 | 0.231784 | 0.642857 | -0.072747 | 3 | 2 | 9 |
| 50 | any_method_hit | Profile Rerank Raw | 14 | 0.333116 | 0.857143 | 0.028585 | 3 | 7 | 4 |
| 50 | any_method_hit | Profile Rerank Remove | 14 | 0.331035 | 0.857143 | 0.026504 | 3 | 9 | 2 |
| 50 | any_method_hit | Profile Rerank EGPR | 14 | 0.304185 | 0.785714 | -0.000346 | 0 | 13 | 1 |
| 100 | any_method_hit | LightGCN | 18 | 0.236857 | 0.611111 | 0.000000 | 0 | 18 | 0 |
| 100 | any_method_hit | DeepSeek Direct Rerank | 18 | 0.219237 | 0.666667 | -0.017620 | 7 | 4 | 7 |
| 100 | any_method_hit | Profile Rerank Raw | 18 | 0.270201 | 0.666667 | 0.033344 | 5 | 8 | 5 |
| 100 | any_method_hit | Profile Rerank Remove | 18 | 0.276394 | 0.722222 | 0.039536 | 4 | 12 | 2 |
| 100 | any_method_hit | Profile Rerank EGPR | 18 | 0.267360 | 0.611111 | 0.030502 | 6 | 7 | 5 |

## Artifacts

- `amazon_books_1000_fair_summary.csv`
- `amazon_books_1000_paired_stats.csv`
- `amazon_books_1000_rank_disruption.csv`
- `amazon_books_1000_affected_subset.csv`
- `amazon_books_1000_profile_lambdas.csv`
- `run_manifest.json`
