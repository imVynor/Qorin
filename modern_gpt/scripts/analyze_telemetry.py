import json
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

metrics_path = REPO_ROOT / "logs/pretrain_100m/training_metrics.jsonl"
thermal_path = REPO_ROOT / "logs/sessions/thermal_telemetry.jsonl"

print("=" * 60)
print("GPU TELEMETRY & THROUGHPUT AUDIT")
print("=" * 60)

temps = []
powers = []
vram_res = []
vram_alloc = []
tok_speeds = []
step_times = []

with open(metrics_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        rec = json.loads(line)
        if rec.get("gpu_temp_c") is not None:
            temps.append(rec["gpu_temp_c"])
        if rec.get("gpu_power_w") is not None:
            powers.append(rec["gpu_power_w"])
        if rec.get("mem_res_mb") is not None:
            vram_res.append(rec["mem_res_mb"])
        if rec.get("mem_alloc_mb") is not None:
            vram_alloc.append(rec["mem_alloc_mb"])
        if rec.get("tok_per_sec") is not None:
            tok_speeds.append(rec["tok_per_sec"])
        if rec.get("step_time_sec") is not None:
            step_times.append(rec["step_time_sec"])

print(f"Total steps logged in metrics: {len(temps)}")
print(f"Temperature (°C):")
print(f"  Min:  {np.min(temps):.1f}°C")
print(f"  Max:  {np.max(temps):.1f}°C")
print(f"  Mean: {np.mean(temps):.1f}°C")
print(f"  Steps >= 78°C (Warning threshold): {sum(1 for t in temps if t >= 78.0)}")
print(f"  Steps >= 82°C (Emergency pause threshold): {sum(1 for t in temps if t >= 82.0)}")

print(f"\nVRAM Utilization (MB):")
print(f"  Allocated Peak: {np.max(vram_alloc):.1f} MB (Mean: {np.mean(vram_alloc):.1f} MB)")
print(f"  Reserved Peak:  {np.max(vram_res):.1f} MB (Mean: {np.mean(vram_res):.1f} MB)")
print(f"  Utilization of 8,123 MB: {np.max(vram_res) / 8123.0 * 100:.1f}%")
print(f"  Headroom: {8123.0 - np.max(vram_res):.1f} MB safe margin")

print(f"\nThroughput & Timing:")
print(f"  Average Throughput: {np.mean(tok_speeds):,.1f} tokens/sec")
print(f"  Max Throughput:     {np.max(tok_speeds):,.1f} tokens/sec")
print(f"  Average Step Time:  {np.mean(step_times):.3f} sec/step")
print(f"  Total Duration (50 steps): {np.sum(step_times):.1f} seconds ({np.sum(step_times)/60.0:.2f} min)")

# Check thermal daemon logs
if thermal_path.exists():
    daemon_temps = []
    throttles = []
    with open(thermal_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            d_rec = json.loads(line)
            if d_rec.get("gpu_temp_c"):
                daemon_temps.append(d_rec["gpu_temp_c"])
            if d_rec.get("throttle_reasons"):
                throttles.append(d_rec["throttle_reasons"])
    print(f"\nThermal Daemon Telemetry:")
    print(f"  Daemon readings count: {len(daemon_temps)}")
    if daemon_temps:
        print(f"  Daemon Max Temp: {np.max(daemon_temps)}°C")
    print(f"  Active Throttling Instances: {len(throttles)}")

