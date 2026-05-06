# MVP Training Dataset Quality Report

Generated: 2026-05-01T15:02:48.839573+00:00

Status: **READY**

## Data Summary

| Source | Fish | Non-fish | Total |
| --- | --- | --- | --- |
| Telegram reviewed seed | 245 | 13 | 258 |
| External (GBIF/iNat, stage_b) | 979 | 0 | 979 |
| External (stage_a no_fish, iNat birds) | 0 | 50 | 50 |
| External (stage_a lure, Wikimedia CC) | 0 | 60 | 60 |
| **Total** | **1224** | **123** | **1347** |

Fish:non-fish imbalance ratio: **10.0:1**

## Split Counts

| Split | Fish | Non-fish | Total |
| --- | --- | --- | --- |
| train | 856 | 87 | 943 |
| val | 183 | 14 | 197 |
| test | 185 | 22 | 207 |

## Training Gates: ALL PASSED

Ready for `train_stage_a.py` or equivalent structural baseline training.

⚠️ **Note:** This dataset is marked EXPERIMENTAL. Do not use for production decisions.