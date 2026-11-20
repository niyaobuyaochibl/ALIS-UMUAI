# Frontier / model-strength robustness check (ML-1M)

Direct LLM reranking re-run on a 500-user ML-1M subset (top-50 candidates,
history-limit 20, identical prompt builder) with two DeepSeek models exposed by
the chat API, compared against the free Sentence-BERT Dense reranker on the same
users. Both API models are reasoning-enabled (hidden chain-of-thought), so they
behave differently from the non-reasoning model in the main table; this is a
fresh robustness check, not a reproduction of the main-table Direct number.

| Interface | NDCG@20 | prompt tok | completion tok | latency/call |
|-----------|---------|-----------|----------------|--------------|
| deepseek-v4-flash Direct | 0.0291 | 1638 | 2265 | 19.3 s |
| deepseek-v4-pro Direct (stronger) | 0.0243 | 1638 | 2998 | 26.7 s |
| free Dense reranker | 0.0291 | 0 | 0 | ~ms (CPU) |

Paired two-sided sign-flip tests (500 users, 56 decisive, 10k resamples):

- pro  - dense : -0.0047  p = 0.29  (not significant)
- pro  - flash : -0.0048  p = 0.32  (not significant)
- flash- dense : +0.0001  p = 0.99  (statistically identical)

Reading: the stronger, more expensive v4-pro reasoning model does NOT beat the
zero-cost Dense reranker (point estimate slightly lower, difference not
significant) while spending ~30% more output tokens and ~40% more latency; the
cheaper flash model is statistically indistinguishable from free Dense. The
negative result is robust to LLM capability rather than an artifact of using a
weak model. Caveat: per-user signal is sparse (56/500 decisive), so the honest
claim is statistical indistinguishability, not that the stronger model is worse.
