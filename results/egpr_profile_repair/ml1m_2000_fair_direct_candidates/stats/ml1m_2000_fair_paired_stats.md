# ML-1M 2000 Fair Candidate Paired Statistics

Paired bootstrap 95% CIs and paired sign-flip randomization tests on per-user NDCG@20 deltas. Positive mean delta means A > B.

| Comparison | Users | MeanDeltaNDCG@20 | BootstrapCI95 | SignFlipP | Wins | Ties | Losses |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Direct vs Base | 2000 | 0.006340 | [0.001304, 0.011344] | 0.013899 | 118 | 1810 | 72 |
| Profile Raw vs Base | 2000 | 0.002950 | [-0.000258, 0.006174] | 0.070493 | 89 | 1841 | 70 |
| Profile EGPR vs Base | 2000 | 0.003129 | [-0.000156, 0.006349] | 0.048795 | 88 | 1841 | 71 |
| Profile EGPR vs Direct | 2000 | -0.003211 | [-0.007250, 0.000907] | 0.123188 | 70 | 1820 | 110 |
| Direct vs Profile EGPR | 2000 | 0.003211 | [-0.000962, 0.007296] | 0.127887 | 110 | 1820 | 70 |
| Profile EGPR vs Raw | 2000 | 0.000179 | [-0.000075, 0.000616] | 0.536246 | 17 | 1964 | 19 |

## Interpretation

- This is the fair top-50 direct-candidate setting: direct and profile methods share the same candidate sets.
- Use this table for confirmatory claims; do not reuse the earlier 500-user same-prefix pilot as final evidence.
