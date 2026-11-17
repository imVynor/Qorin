# Phase 1B.5: Download and Streaming Throughput Log

## Session Performance Overview

Streaming was executed using the authenticated Hugging Face Hub client over persistent HTTPS connections with automatic exponential backoff.

| Domain | Raw Bytes Examined | Elapsed (s) | Measured Throughput (KB/s) | Docs Accepted | Tokens Added | Checkpoint Frequency |
|---|---:|---:|---:|---:|---:|---|
| **Quality Web** | 13,014,996 | 21.42 | 593.4 | 1,200 | 1,243,851 | Every 250 docs |
| **Educational** | 8,308,318 | 15.64 | 518.8 | 1,000 | 709,640 | Every 250 docs |
| **Source Code** | 33,132,231 | 12.69 | 2550.1 | 1,000 | 1,516,577 | Every 250 docs |
| **Wikipedia** | 44,414,893 | 30.04 | 1443.8 | 1,000 | 4,819,454 | Every 250 docs |
| **Mathematics** | 16,370,938 | 17.4 | 919.1 | 1,000 | 1,724,596 | Every 250 docs |
| **Books / Prose** | 20,528,921 | 35.45 | 565.5 | 75 | 1,248,515 | Checkpointed |
| **Curated Dialogue** | 12,271,344 | 15.68 | 764.2 | 1,000 | 1,190,320 | Every 250 docs |
| **Session Total** | **148,041,641** | **148.32** | **974.7 avg** | **6,275** | **12,452,953** | — |

*Note: UltraChat encountered a transient remote socket close (`peer closed connection without sending complete message body`) which was automatically recovered by the built-in HTTP retry handler on retry 1/5 without data corruption.*
