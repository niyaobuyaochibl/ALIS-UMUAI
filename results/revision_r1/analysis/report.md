# ALIS R1 revision analyses

## Panel: books

- users: 1000 (dropped for missing interfaces: 0)
- interfaces: Base, Content, BM25, Dense, Direct, Raw, Remove, EGPR
- mean NDCG@20: Base=0.0043, Content=0.0053, BM25=0.0040, Dense=0.0046, Direct=0.0039, Raw=0.0049, Remove=0.0050, EGPR=0.0048
- overlap source: Base: per-user, Content: per-user, BM25: per-user, Dense: per-user, Direct: per-user, Raw: per-user, Remove: per-user, EGPR: per-user
- routing features: base_content_overlap, base_entropy, base_margin, base_std, base_top1, log_hist

## Panel: ml1m

- users: 2000 (dropped for missing interfaces: 0)
- interfaces: Base, Content, BM25, Dense, Direct, Raw, Remove, EGPR
- mean NDCG@20: Base=0.0249, Content=0.0267, BM25=0.0284, Dense=0.0320, Direct=0.0312, Raw=0.0278, Remove=0.0278, EGPR=0.0280
- overlap source: Base: per-user, Content: per-user, BM25: per-user, Dense: per-user, Direct: per-user, Raw: per-user, Remove: per-user, EGPR: per-user
- routing features: base_content_overlap, base_entropy, base_margin, base_std, base_top1, log_hist

## Panel: yelp

- users: 1000 (dropped for missing interfaces: 0)
- interfaces: Base, Content, Direct, Raw, Remove, EGPR
- mean NDCG@20: Base=0.0155, Content=0.0164, Direct=0.0152, Raw=0.0166, Remove=0.0160, EGPR=0.0152
- overlap source: Base: per-user, Content: per-user, Direct: per-user, Raw: per-user, Remove: per-user, EGPR: per-user
- routing features: base_content_overlap, base_entropy, base_margin, base_std, base_top1, log_hist

## Out-of-sample ALIS (select on half, evaluate on disjoint half)

| domain | preference | selector | heldout_ndcg | heldout_harm | cost | llm_rate | ndcg_regret | ndcg_regret_sd | score_regret | agree_rate | freq_BM25 | freq_Base | freq_Content | freq_Dense | freq_Direct | freq_EGPR | freq_Raw | freq_Remove |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| books | accuracy_first | confidence_aware | 0.0043 | 0.0002 | 0.0028 | 0.0050 | 0.0017 | 0.0010 | 0.5917 | 0.0300 | 0.0000 | 0.9750 | 0.0150 | 0.0050 | 0.0000 | 0.0000 | 0.0000 | 0.0050 |
| books | accuracy_first | plain | 0.0038 | 0.0060 | 0.2326 | 0.3500 | 0.0022 | 0.0013 | 0.7212 | 0.0450 | 0.0650 | 0.0250 | 0.3450 | 0.2150 | 0.0800 | 0.0500 | 0.0400 | 0.1800 |
| books | balanced | confidence_aware | 0.0043 | 0.0001 | 0.0028 | 0.0050 | 0.0017 | 0.0010 | 0.4790 | 0.0650 | 0.0000 | 0.9800 | 0.0100 | 0.0050 | 0.0000 | 0.0000 | 0.0000 | 0.0050 |
| books | balanced | plain | 0.0039 | 0.0056 | 0.1928 | 0.2950 | 0.0021 | 0.0013 | 0.7046 | 0.0850 | 0.0750 | 0.0600 | 0.4000 | 0.1700 | 0.0600 | 0.0500 | 0.0000 | 0.1850 |
| books | cost_first | confidence_aware | 0.0043 | 0.0001 | 0.0000 | 0.0000 | 0.0017 | 0.0010 | 0.2450 | 0.1900 | 0.0000 | 0.9950 | 0.0000 | 0.0050 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| books | cost_first | plain | 0.0040 | 0.0049 | 0.0085 | 0.0150 | 0.0020 | 0.0013 | 0.3579 | 0.1750 | 0.0950 | 0.2300 | 0.4800 | 0.1800 | 0.0000 | 0.0050 | 0.0000 | 0.0100 |
| books | safety_first | confidence_aware | 0.0043 | 0.0000 | 0.0057 | 0.0100 | 0.0017 | 0.0009 | 0.2192 | 0.1950 | 0.0000 | 0.9900 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0100 |
| books | safety_first | plain | 0.0041 | 0.0045 | 0.1687 | 0.2600 | 0.0019 | 0.0013 | 0.4590 | 0.0650 | 0.0500 | 0.1850 | 0.4250 | 0.0800 | 0.0500 | 0.0100 | 0.0000 | 0.2000 |
| ml1m | accuracy_first | confidence_aware | 0.0272 | 0.0233 | 0.0000 | 0.0000 | 0.0050 | 0.0043 | 0.5532 | 0.2150 | 0.0050 | 0.3650 | 0.1500 | 0.4800 | 0.0000 | 0.0000 | nan | 0.0000 |
| ml1m | accuracy_first | plain | 0.0306 | 0.0348 | 0.2945 | 0.3000 | 0.0016 | 0.0017 | 0.2143 | 0.3050 | 0.0050 | 0.0000 | 0.0000 | 0.6950 | 0.2900 | 0.0100 | nan | 0.0000 |
| ml1m | balanced | confidence_aware | 0.0272 | 0.0238 | 0.0000 | 0.0000 | 0.0050 | 0.0043 | 0.4744 | 0.2550 | 0.0050 | 0.3500 | 0.1700 | 0.4750 | 0.0000 | 0.0000 | nan | 0.0000 |
| ml1m | balanced | plain | 0.0308 | 0.0345 | 0.1495 | 0.1550 | 0.0014 | 0.0019 | 0.1630 | 0.5550 | 0.0200 | 0.0000 | 0.0000 | 0.8250 | 0.1450 | 0.0050 | nan | 0.0050 |
| ml1m | cost_first | confidence_aware | 0.0267 | 0.0230 | 0.0000 | 0.0000 | 0.0055 | 0.0041 | 0.3011 | 0.2900 | 0.0000 | 0.3750 | 0.2250 | 0.4000 | 0.0000 | 0.0000 | nan | 0.0000 |
| ml1m | cost_first | plain | 0.0315 | 0.0338 | 0.0000 | 0.0000 | 0.0008 | 0.0014 | 0.0244 | 0.8600 | 0.0200 | 0.0000 | 0.0000 | 0.9800 | 0.0000 | 0.0000 | nan | 0.0000 |
| ml1m | safety_first | confidence_aware | 0.0272 | 0.0239 | 0.0089 | 0.0200 | 0.0050 | 0.0043 | 0.2560 | 0.2700 | 0.0000 | 0.3500 | 0.1550 | 0.4750 | 0.0000 | 0.0000 | nan | 0.0200 |
| ml1m | safety_first | plain | 0.0309 | 0.0342 | 0.1395 | 0.1450 | 0.0013 | 0.0019 | 0.1072 | 0.6050 | 0.0050 | 0.0050 | 0.0000 | 0.8450 | 0.1350 | 0.0050 | nan | 0.0050 |
| yelp | accuracy_first | confidence_aware | 0.0157 | 0.0005 | 0.0000 | 0.0000 | 0.0022 | 0.0014 | 0.5685 | 0.1200 | nan | 0.9800 | 0.0200 | nan | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| yelp | accuracy_first | plain | 0.0148 | 0.0186 | 0.3439 | 0.5350 | 0.0031 | 0.0017 | 0.7763 | 0.0050 | nan | 0.1200 | 0.3450 | nan | 0.1750 | 0.0100 | 0.3150 | 0.0350 |
| yelp | balanced | confidence_aware | 0.0157 | 0.0005 | 0.0000 | 0.0000 | 0.0022 | 0.0014 | 0.3764 | 0.1700 | nan | 0.9800 | 0.0200 | nan | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| yelp | balanced | plain | 0.0147 | 0.0164 | 0.2717 | 0.4150 | 0.0033 | 0.0016 | 0.7922 | 0.0000 | nan | 0.2400 | 0.3450 | nan | 0.1450 | 0.0100 | 0.1650 | 0.0950 |
| yelp | cost_first | confidence_aware | 0.0157 | 0.0003 | 0.0000 | 0.0000 | 0.0022 | 0.0014 | 0.1380 | 0.3750 | nan | 0.9900 | 0.0100 | nan | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| yelp | cost_first | plain | 0.0152 | 0.0137 | 0.0801 | 0.1650 | 0.0027 | 0.0013 | 0.3332 | 0.0350 | nan | 0.3900 | 0.4450 | nan | 0.0050 | 0.0050 | 0.1200 | 0.0350 |
| yelp | safety_first | confidence_aware | 0.0156 | 0.0006 | 0.0100 | 0.0100 | 0.0023 | 0.0016 | 0.0974 | 0.5200 | nan | 0.9800 | 0.0100 | nan | 0.0100 | 0.0000 | 0.0000 | 0.0000 |
| yelp | safety_first | plain | 0.0147 | 0.0103 | 0.1839 | 0.2450 | 0.0032 | 0.0015 | 0.4405 | 0.1100 | nan | 0.5450 | 0.2100 | nan | 0.1300 | 0.0100 | 0.0000 | 0.1050 |

## Bootstrap selection frequencies (full_panel_choice = retrospective choice)

| domain | preference | selector | freq_BM25 | freq_Base | freq_Content | freq_Dense | freq_Direct | freq_EGPR | freq_Raw | freq_Remove | full_panel_choice |
|---|---|---|---|---|---|---|---|---|---|---|---|
| books | accuracy_first | confidence_aware | 0.0020 | 0.7815 | 0.1675 | 0.0050 | 0.0000 | 0.0010 | 0.0000 | 0.0430 | Base |
| books | accuracy_first | plain | 0.0920 | 0.0040 | 0.3660 | 0.2190 | 0.0665 | 0.0805 | 0.0290 | 0.1430 | Content |
| books | balanced | confidence_aware | 0.0015 | 0.7765 | 0.1630 | 0.0045 | 0.0000 | 0.0010 | 0.0000 | 0.0535 | Base |
| books | balanced | plain | 0.0840 | 0.0325 | 0.4220 | 0.1970 | 0.0440 | 0.0390 | 0.0010 | 0.1805 | Content |
| books | cost_first | confidence_aware | 0.0035 | 0.8810 | 0.1070 | 0.0085 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Base |
| books | cost_first | plain | 0.0870 | 0.1565 | 0.5205 | 0.2165 | 0.0015 | 0.0030 | 0.0000 | 0.0150 | Content |
| books | safety_first | confidence_aware | 0.0005 | 0.7325 | 0.0620 | 0.0025 | 0.0025 | 0.0010 | 0.0000 | 0.1990 | Base |
| books | safety_first | plain | 0.0555 | 0.1440 | 0.4000 | 0.1075 | 0.0315 | 0.0015 | 0.0000 | 0.2600 | Content |
| ml1m | accuracy_first | confidence_aware | 0.0190 | 0.0960 | 0.1350 | 0.7445 | 0.0055 | 0.0000 | nan | 0.0000 | Dense |
| ml1m | accuracy_first | plain | 0.0455 | 0.0000 | 0.0010 | 0.6585 | 0.2900 | 0.0040 | nan | 0.0010 | Dense |
| ml1m | balanced | confidence_aware | 0.0185 | 0.0945 | 0.1575 | 0.7240 | 0.0055 | 0.0000 | nan | 0.0000 | Dense |
| ml1m | balanced | plain | 0.0505 | 0.0000 | 0.0010 | 0.8145 | 0.1300 | 0.0035 | nan | 0.0005 | Dense |
| ml1m | cost_first | confidence_aware | 0.0145 | 0.1140 | 0.1950 | 0.6760 | 0.0005 | 0.0000 | nan | 0.0000 | Dense |
| ml1m | cost_first | plain | 0.0625 | 0.0010 | 0.0035 | 0.9320 | 0.0005 | 0.0005 | nan | 0.0000 | Dense |
| ml1m | safety_first | confidence_aware | 0.0160 | 0.0925 | 0.1415 | 0.7235 | 0.0055 | 0.0000 | nan | 0.0210 | Dense |
| ml1m | safety_first | plain | 0.0450 | 0.0060 | 0.0020 | 0.8170 | 0.1240 | 0.0020 | nan | 0.0040 | Dense |
| yelp | accuracy_first | confidence_aware | nan | 0.8650 | 0.1220 | nan | 0.0020 | 0.0010 | 0.0015 | 0.0085 | Base |
| yelp | accuracy_first | plain | nan | 0.1395 | 0.3270 | nan | 0.1480 | 0.0080 | 0.3575 | 0.0200 | Raw |
| yelp | balanced | confidence_aware | nan | 0.8675 | 0.1205 | nan | 0.0020 | 0.0010 | 0.0015 | 0.0075 | Base |
| yelp | balanced | plain | nan | 0.2330 | 0.3445 | nan | 0.1265 | 0.0085 | 0.2375 | 0.0500 | Content |
| yelp | cost_first | confidence_aware | nan | 0.8860 | 0.1045 | nan | 0.0005 | 0.0010 | 0.0015 | 0.0065 | Base |
| yelp | cost_first | plain | nan | 0.3695 | 0.4595 | nan | 0.0160 | 0.0035 | 0.1345 | 0.0170 | Content |
| yelp | safety_first | confidence_aware | nan | 0.8945 | 0.0505 | nan | 0.0230 | 0.0000 | 0.0000 | 0.0320 | Base |
| yelp | safety_first | plain | nan | 0.5295 | 0.2185 | nan | 0.1235 | 0.0045 | 0.0005 | 0.1235 | Content |

## Paired per-user NDCG@20 differences

| domain | left | right | users | mean_delta | ci_low | ci_high | signflip_p | wins | losses |
|---|---|---|---|---|---|---|---|---|---|
| books | Content | Base | 1000 | 0.0011 | -0.0006 | 0.0032 | 0.3354 | 6 | 4 |
| books | BM25 | Base | 1000 | -0.0003 | -0.0031 | 0.0027 | 0.8588 | 6 | 7 |
| books | Dense | Base | 1000 | 0.0003 | -0.0028 | 0.0036 | 0.8776 | 7 | 8 |
| books | Direct | Base | 1000 | -0.0003 | -0.0032 | 0.0020 | 0.8485 | 7 | 7 |
| books | Raw | Base | 1000 | 0.0006 | -0.0004 | 0.0017 | 0.2325 | 5 | 5 |
| books | Remove | Base | 1000 | 0.0007 | -0.0000 | 0.0016 | 0.1622 | 4 | 2 |
| books | EGPR | Base | 1000 | 0.0005 | -0.0006 | 0.0018 | 0.3878 | 6 | 5 |
| books | Dense | Direct | 1000 | 0.0006 | -0.0023 | 0.0038 | 0.7169 | 7 | 9 |
| books | Raw | Content | 1000 | -0.0005 | -0.0025 | 0.0011 | 0.6996 | 5 | 5 |
| books | Content | Remove | 1000 | 0.0004 | -0.0013 | 0.0024 | 0.7691 | 5 | 6 |
| books | Content | Raw | 1000 | 0.0005 | -0.0011 | 0.0024 | 0.6972 | 5 | 5 |
| books | EGPR | Raw | 1000 | -0.0001 | -0.0007 | 0.0004 | 0.9393 | 3 | 2 |
| books | Remove | Raw | 1000 | 0.0001 | -0.0003 | 0.0007 | 0.9099 | 3 | 4 |
| ml1m | Content | Base | 2000 | 0.0019 | -0.0014 | 0.0052 | 0.2670 | 84 | 72 |
| ml1m | BM25 | Base | 2000 | 0.0036 | -0.0011 | 0.0082 | 0.1382 | 98 | 82 |
| ml1m | Dense | Base | 2000 | 0.0072 | 0.0027 | 0.0119 | 0.0027 | 108 | 68 |
| ml1m | Direct | Base | 2000 | 0.0063 | 0.0012 | 0.0116 | 0.0131 | 118 | 72 |
| ml1m | Raw | Base | 2000 | 0.0029 | -0.0002 | 0.0061 | 0.0662 | 89 | 70 |
| ml1m | Remove | Base | 2000 | 0.0029 | -0.0002 | 0.0061 | 0.0693 | 89 | 70 |
| ml1m | EGPR | Base | 2000 | 0.0031 | -0.0001 | 0.0064 | 0.0578 | 88 | 71 |
| ml1m | Dense | Direct | 2000 | 0.0008 | -0.0031 | 0.0046 | 0.6787 | 88 | 86 |
| ml1m | Raw | Content | 2000 | 0.0011 | -0.0022 | 0.0044 | 0.5193 | 85 | 71 |
| ml1m | Content | Remove | 2000 | -0.0011 | -0.0044 | 0.0022 | 0.5121 | 71 | 85 |
| ml1m | Content | Raw | 2000 | -0.0011 | -0.0044 | 0.0022 | 0.5150 | 71 | 85 |
| ml1m | EGPR | Raw | 2000 | 0.0002 | -0.0001 | 0.0006 | 0.5409 | 17 | 19 |
| ml1m | Remove | Raw | 2000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 0 |
| yelp | Content | Base | 1000 | 0.0008 | -0.0024 | 0.0043 | 0.6406 | 20 | 21 |
| yelp | Direct | Base | 1000 | -0.0003 | -0.0046 | 0.0038 | 0.8903 | 22 | 16 |
| yelp | Raw | Base | 1000 | 0.0010 | -0.0026 | 0.0049 | 0.5922 | 21 | 19 |
| yelp | Remove | Base | 1000 | 0.0005 | -0.0029 | 0.0040 | 0.7942 | 21 | 19 |
| yelp | EGPR | Base | 1000 | -0.0003 | -0.0038 | 0.0033 | 0.8805 | 21 | 20 |
| yelp | Raw | Content | 1000 | 0.0002 | -0.0033 | 0.0040 | 0.9176 | 17 | 23 |
| yelp | Content | Remove | 1000 | 0.0004 | -0.0028 | 0.0035 | 0.8287 | 23 | 17 |
| yelp | Content | Raw | 1000 | -0.0002 | -0.0040 | 0.0033 | 0.9182 | 23 | 17 |
| yelp | EGPR | Raw | 1000 | -0.0013 | -0.0030 | 0.0001 | 0.1057 | 10 | 12 |
| yelp | Remove | Raw | 1000 | -0.0006 | -0.0017 | 0.0001 | 0.3623 | 3 | 5 |

## ALIS choice when Direct is assigned non-zero grounding risk

| domain | direct_grounding | value | preference | selected | direct_score | best_score |
|---|---|---|---|---|---|---|
| books | zero | 0.0000 | accuracy_first | Content | -0.0409 | 0.9866 |
| books | zero | 0.0000 | balanced | Content | -0.3157 | 0.8910 |
| books | zero | 0.0000 | safety_first | Content | -0.4579 | 0.4739 |
| books | zero | 0.0000 | cost_first | Content | -0.5719 | 0.5160 |
| books | mean_profile_wucr | 0.1586 | accuracy_first | Content | -0.0455 | 0.9866 |
| books | mean_profile_wucr | 0.1586 | balanced | Content | -0.3613 | 0.8910 |
| books | mean_profile_wucr | 0.1586 | safety_first | Content | -0.6174 | 0.4739 |
| books | mean_profile_wucr | 0.1586 | cost_first | Content | -0.5947 | 0.5160 |
| books | max_profile_wucr | 0.3480 | accuracy_first | Content | -0.0509 | 0.9866 |
| books | max_profile_wucr | 0.3480 | balanced | Content | -0.4157 | 0.8910 |
| books | max_profile_wucr | 0.3480 | safety_first | Content | -0.8079 | 0.4739 |
| books | max_profile_wucr | 0.3480 | cost_first | Content | -0.6219 | 0.5160 |
| ml1m | zero | 0.0000 | accuracy_first | Dense | 0.8409 | 0.9798 |
| ml1m | zero | 0.0000 | balanced | Dense | 0.5531 | 0.8399 |
| ml1m | zero | 0.0000 | safety_first | Dense | 0.1382 | 0.3562 |
| ml1m | zero | 0.0000 | cost_first | Dense | -0.0563 | 0.4814 |
| ml1m | mean_profile_wucr | 0.0438 | accuracy_first | Dense | 0.8373 | 0.9798 |
| ml1m | mean_profile_wucr | 0.0438 | balanced | Dense | 0.5174 | 0.8399 |
| ml1m | mean_profile_wucr | 0.0438 | safety_first | Dense | 0.0133 | 0.3562 |
| ml1m | mean_profile_wucr | 0.0438 | cost_first | Dense | -0.0742 | 0.4814 |
| ml1m | max_profile_wucr | 0.1227 | accuracy_first | Dense | 0.8309 | 0.9798 |
| ml1m | max_profile_wucr | 0.1227 | balanced | Dense | 0.4531 | 0.8399 |
| ml1m | max_profile_wucr | 0.1227 | safety_first | Dense | -0.2118 | 0.3562 |
| ml1m | max_profile_wucr | 0.1227 | cost_first | Dense | -0.1063 | 0.4814 |
| yelp | zero | 0.0000 | accuracy_first | Raw | -0.0392 | 0.9588 |
| yelp | zero | 0.0000 | balanced | Content | -0.3038 | 0.6644 |
| yelp | zero | 0.0000 | safety_first | Content | -0.4259 | 0.1890 |
| yelp | zero | 0.0000 | cost_first | Content | -0.5657 | 0.3731 |
| yelp | mean_profile_wucr | 0.0335 | accuracy_first | Raw | -0.0429 | 0.9588 |
| yelp | mean_profile_wucr | 0.0335 | balanced | Content | -0.3412 | 0.6644 |
| yelp | mean_profile_wucr | 0.0335 | safety_first | Content | -0.5569 | 0.1890 |
| yelp | mean_profile_wucr | 0.0335 | cost_first | Content | -0.5844 | 0.3731 |
| yelp | max_profile_wucr | 0.0894 | accuracy_first | Raw | -0.0492 | 0.9588 |
| yelp | max_profile_wucr | 0.0894 | balanced | Content | -0.4038 | 0.6644 |
| yelp | max_profile_wucr | 0.0894 | safety_first | Content | -0.7759 | 0.1890 |
| yelp | max_profile_wucr | 0.0894 | cost_first | Content | -0.6157 | 0.3731 |

## Per-user routing, repeated nested cross-validation

| domain | policy | ndcg | ndcg_sd_over_reps | delta_vs_fixed | delta_ci_low | delta_ci_high | paid_call_rate |
|---|---|---|---|---|---|---|---|
| books | fixed | 0.0038 | 0.0010 | 0.0000 | 0.0000 | 0.0000 | 0.2200 |
| books | logit_label | 0.0043 | 0.0007 | 0.0005 | -0.0007 | 0.0016 | 0.5217 |
| books | logit_l1 | 0.0045 | 0.0006 | 0.0007 | -0.0001 | 0.0016 | 0.5648 |
| books | ridge_reward | 0.0033 | 0.0010 | -0.0006 | -0.0014 | 0.0000 | 0.3378 |
| books | gbm_reward | 0.0050 | 0.0010 | 0.0012 | -0.0001 | 0.0026 | 0.5748 |
| books | oracle_insample | 0.0097 | nan | 0.0044 | nan | nan | 0.0070 |
| ml1m | fixed | 0.0305 | 0.0008 | 0.0000 | 0.0000 | 0.0000 | 0.2600 |
| ml1m | logit_label | 0.0278 | 0.0005 | -0.0027 | -0.0059 | 0.0006 | 0.5199 |
| ml1m | logit_l1 | 0.0277 | 0.0007 | -0.0029 | -0.0061 | 0.0003 | 0.5891 |
| ml1m | ridge_reward | 0.0302 | 0.0011 | -0.0003 | -0.0020 | 0.0013 | 0.3550 |
| ml1m | gbm_reward | 0.0293 | 0.0006 | -0.0012 | -0.0031 | 0.0008 | 0.4245 |
| ml1m | oracle_insample | 0.0535 | nan | 0.0215 | nan | nan | 0.0390 |
| yelp | fixed | 0.0146 | 0.0008 | 0.0000 | 0.0000 | 0.0000 | 0.6400 |
| yelp | logit_label | 0.0154 | 0.0007 | 0.0008 | -0.0010 | 0.0026 | 0.5132 |
| yelp | logit_l1 | 0.0152 | 0.0006 | 0.0006 | -0.0006 | 0.0018 | 0.6204 |
| yelp | ridge_reward | 0.0161 | 0.0006 | 0.0015 | -0.0000 | 0.0031 | 0.5301 |
| yelp | gbm_reward | 0.0148 | 0.0009 | 0.0002 | -0.0016 | 0.0019 | 0.6128 |
| yelp | oracle_insample | 0.0273 | nan | 0.0108 | nan | nan | 0.0270 |

## Which interface the fold-wise best fixed policy selected

| domain | selected | freq |
|---|---|---|
| books | Content | 0.6800 |
| books | Remove | 0.1600 |
| books | Dense | 0.1000 |
| books | EGPR | 0.0200 |
| books | Direct | 0.0200 |
| books | Raw | 0.0200 |
| ml1m | Dense | 0.7400 |
| ml1m | Direct | 0.2600 |
| yelp | Raw | 0.6200 |
| yelp | Content | 0.3000 |
| yelp | Base | 0.0600 |
| yelp | Direct | 0.0200 |

## Original RQ6 protocol (five validation/test halves)

| domain | seed | fixed_selected | fixed_ndcg | fixed_paid | router_ndcg | router_paid_rate |
|---|---|---|---|---|---|---|
| books | 0 | Remove | 0.0051 | True | 0.0042 | 0.2920 |
| books | 1 | Dense | 0.0019 | False | 0.0027 | 0.4080 |
| books | 2 | Remove | 0.0042 | True | 0.0030 | 0.2920 |
| books | 3 | Content | 0.0044 | False | 0.0066 | 0.4740 |
| books | 4 | Direct | 0.0061 | True | 0.0065 | 0.2580 |
| ml1m | 0 | Raw | 0.0267 | True | 0.0284 | 0.5410 |
| ml1m | 1 | Direct | 0.0356 | True | 0.0301 | 0.4950 |
| ml1m | 2 | Dense | 0.0287 | False | 0.0272 | 0.3940 |
| ml1m | 3 | Dense | 0.0335 | False | 0.0277 | 0.4530 |
| ml1m | 4 | Dense | 0.0311 | False | 0.0286 | 0.4420 |
| yelp | 0 | Raw | 0.0116 | True | 0.0112 | 0.5200 |
| yelp | 1 | Raw | 0.0172 | True | 0.0173 | 0.5200 |
| yelp | 2 | Raw | 0.0134 | True | 0.0130 | 0.4260 |
| yelp | 3 | Direct | 0.0152 | True | 0.0175 | 0.3620 |
| yelp | 4 | Direct | 0.0171 | True | 0.0195 | 0.3440 |
