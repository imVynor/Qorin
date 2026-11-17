# Sustained Training Throughput Benchmark

## Measured Steady-State Performance
- **Microbatch Size**: 1 (2,048 tokens)
- **Gradient Accumulation**: 16 steps
- **Effective Batch Size**: 32,768 tokens per optimizer update
- **Mean Sustained Throughput**: **5,124.5 tokens/sec**
- **Median Sustained Throughput**: **5,054.4 tokens/sec**
- **Conservative Throughput (25th Percentile)**: **5,017.2 tokens/sec**
- **Seconds per Optimizer Update**: 6.404 seconds
- **GPU Utilization**: 100% during active steps
- **Average GPU Temperature**: 86.5°C (Max: 87.0°C)
- **Average Power Draw**: 89.3 W (TDP cap: 115W)
- **CPU Utilization**: 8.0%
