"""ModernGPT-S 100M: Hardware Telemetry and Thermal Watchdog.

Monitors NVIDIA GPU status in real time:
- Core temperature (°C)
- GPU utilization (%)
- Memory utilization (%)
- Allocated VRAM and Total VRAM (MB)
- Power draw (W)
- Throttling flags

Provides:
- Background monitoring thread writing JSONL telemetry.
- Dynamic thermal guardrails (warning threshold 90°C, pause/checkpoint threshold 95°C).
- Blocking cooldown loop waiting for GPU temperature to settle below 70°C.
"""

import datetime
import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("ModernGPT.GPUMonitor")


class GPUMonitor:
    def __init__(
        self,
        log_file: Optional[Path] = None,
        poll_interval_seconds: float = 10.0,
        warn_temp_celsius: float = 90.0,
        max_temp_celsius: float = 95.0,
        emergency_cooldown_temp: float = 70.0,
    ):
        self.log_file = log_file
        self.poll_interval = poll_interval_seconds
        self.warn_temp = warn_temp_celsius
        self.max_temp = max_temp_celsius
        self.cooldown_temp = emergency_cooldown_temp

        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_metrics: Dict[str, Any] = {}
        self._thermal_history: list = []

    def poll_once(self) -> Dict[str, Any]:
        """Poll nvidia-smi synchronously and return structured metric dict."""
        cmd = [
            "nvidia-smi",
            "--query-gpu=temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.current.graphics,clocks.current.memory,clocks_throttle_reasons.active,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.hw_thermal_slowdown",
            "--format=csv,noheader,nounits",
        ]
        try:
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
                check=True,
            )
            line = res.stdout.strip()
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                metrics = {
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "gpu_temp_c": float(parts[0]),
                    "gpu_util_pct": float(parts[1]),
                    "mem_util_pct": float(parts[2]),
                    "mem_used_mb": float(parts[3]),
                    "mem_total_mb": float(parts[4]),
                    "power_draw_w": float(parts[5]),
                    "clock_graphics_mhz": float(parts[6]) if len(parts) > 6 else None,
                    "clock_memory_mhz": float(parts[7]) if len(parts) > 7 else None,
                    "throttle_active": parts[8] if len(parts) > 8 else None,
                    "sw_thermal_throttle": parts[9] if len(parts) > 9 else None,
                    "hw_thermal_throttle": parts[10] if len(parts) > 10 else None,
                }
                with self._lock:
                    self._latest_metrics = metrics
                    self._thermal_history.append((time.time(), metrics["gpu_temp_c"]))
                    if len(self._thermal_history) > 1000:
                        self._thermal_history.pop(0)
                return metrics
        except Exception as e:
            logger.debug(f"GPU poll failed: {e}")

        with self._lock:
            return dict(self._latest_metrics)

    def _worker(self):
        while not self._stop_event.is_set():
            metrics = self.poll_once()
            if metrics and self.log_file:
                try:
                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(metrics) + "\n")
                except Exception:
                    pass

            # Log warning if threshold breached
            temp = metrics.get("gpu_temp_c", 0.0)
            if temp >= self.max_temp:
                logger.warning(
                    f"⚠️ CRITICAL THERMAL LIMIT REACHED: {temp}°C >= {self.max_temp}°C! Initiating thermal throttle/pause."
                )
            elif temp >= self.warn_temp:
                logger.warning(f"⚠️ THERMAL WARNING: {temp}°C >= {self.warn_temp}°C. Nearing safety threshold.")

            self._stop_event.wait(self.poll_interval)

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="GPUMonitorThread")
        self._thread.start()
        logger.info(f"GPU Monitor background daemon started (poll interval: {self.poll_interval}s).")

    def stop(self):
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=3.0)
            self._thread = None
            logger.info("GPU Monitor stopped.")

    def get_latest(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._latest_metrics)

    def is_overheating(self) -> Tuple[bool, float]:
        """Returns (is_overheating, current_temp)."""
        metrics = self.poll_once()
        temp = metrics.get("gpu_temp_c", 0.0)
        return (temp >= self.max_temp, temp)

    def wait_for_cooldown(
        self,
        target_temp: Optional[float] = None,
        poll_interval: float = 10.0,
        min_cooldown_seconds: float = 0.0,
        timeout_seconds: float = 3600.0,
    ) -> float:
        """Block execution until mandatory cooldown window elapses AND GPU temperature drops to or below target_temp."""
        target = target_temp or self.cooldown_temp
        logger.info(
            f"Entering cooling break (mandatory duration: {min_cooldown_seconds/60.0:.1f} min, target: <= {target}°C)..."
        )
        t_start = time.time()

        while True:
            elapsed = time.time() - t_start
            metrics = self.poll_once()
            current_temp = metrics.get("gpu_temp_c", 100.0)
            logger.info(
                f"[Cooling Break] Elapsed: {elapsed:.0f}s / {min_cooldown_seconds:.0f}s | "
                f"Current Temp: {current_temp}°C | Target: <={target}°C"
            )

            # Both conditions must be satisfied: mandatory rest duration has elapsed AND temperature is safe
            if elapsed >= min_cooldown_seconds and current_temp <= target:
                logger.info(f"✓ Cooling break complete in {elapsed/60.0:.1f} min (GPU at {current_temp}°C <= {target}°C).")
                return current_temp

            if elapsed >= timeout_seconds:
                logger.warning(f"Maximum cooldown timeout of {timeout_seconds}s reached. Resuming.")
                return current_temp

            time.sleep(poll_interval)


if __name__ == "__main__":
    monitor = GPUMonitor()
    print("Testing single GPU poll:")
    m = monitor.poll_once()
    print(json.dumps(m, indent=2))

