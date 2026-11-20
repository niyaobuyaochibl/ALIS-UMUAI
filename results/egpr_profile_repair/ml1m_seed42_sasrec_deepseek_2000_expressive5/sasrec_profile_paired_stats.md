# ML-1M SASRec Profile Rerank Paired Statistics

Paired bootstrap confidence intervals and paired sign-flip tests over the same 2000 test users.

## NDCG@20

| Comparison | Metric | Users | MeanDelta | BootstrapCI95Low | BootstrapCI95High | SignFlipPValueTwoSided | Wins | Ties | Losses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Raw vs SASRec | NDCG@20 | 2000 | 0.004716 | 0.001827 | 0.007590 | 0.001310 | 89 | 1866 | 45 |
| Remove vs SASRec | NDCG@20 | 2000 | 0.004716 | 0.001871 | 0.007539 | 0.001080 | 89 | 1866 | 45 |
| EGPR vs SASRec | NDCG@20 | 2000 | 0.004758 | 0.001814 | 0.007663 | 0.001490 | 91 | 1863 | 46 |
| EGPR vs Raw | NDCG@20 | 2000 | 0.000042 | -0.000528 | 0.000629 | 0.829282 | 18 | 1965 | 17 |
| Remove vs Raw | NDCG@20 | 2000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 2000 | 0 |

## Interpretation

- Profile reranking has positive mean deltas over SASRec for Raw, Remove, and Evidence-Weighted variants.
- The confidence intervals for the profile-vs-base comparisons are above zero or touch zero at the lower endpoint, giving stronger backbone evidence than the earlier 500-user pilots.
- Differences among Raw, Remove, and Evidence-Weighted repair remain tiny on ML-1M SASRec.
