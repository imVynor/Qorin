# Runtime Estimates: 500M to 5B Token Budgets

Estimates based on measured median sustained throughput (5,054.4 tokens/sec), including mandatory 105-min train / 15-min cooldown cycles, validation, and atomic checkpointing.

| Budget | Tokens | Pure GPU Train | Cooldown Cycles (1:45/15) | Val & Ckpt Overhead | Total Wall-Clock (Expected) | Total Days |
|:---:|---:|---:|:---:|---:|---:|---:|
| **500M** | 500,000,000 | 27.48 hrs | 15 cycles (3.75 hrs) | 0.19 hrs | **31.42 hrs** | **1.31 days** |
| **1B** | 1,000,000,000 | 54.96 hrs | 31 cycles (7.75 hrs) | 0.39 hrs | **63.09 hrs** | **2.63 days** |
| **2B** | 2,000,000,000 | 109.92 hrs | 62 cycles (15.5 hrs) | 0.76 hrs | **126.18 hrs** | **5.26 days** |
| **3B** | 3,000,000,000 | 164.87 hrs | 94 cycles (23.5 hrs) | 1.15 hrs | **189.52 hrs** | **7.9 days** |
| **4B** | 4,000,000,000 | 219.83 hrs | 125 cycles (31.25 hrs) | 1.54 hrs | **252.61 hrs** | **10.53 days** |
| **5B (Full)** | 5,000,000,000 | 274.79 hrs | 157 cycles (39.25 hrs) | 1.91 hrs | **315.95 hrs** | **13.16 days** |

## Achievable Tokens in Fixed Windows
- **6 Hours**: ~93M tokens
- **12 Hours**: ~186M tokens
- **24 Hours**: ~371M tokens
- **48 Hours**: ~742M tokens
- **72 Hours (3 Days)**: ~1.11B tokens
