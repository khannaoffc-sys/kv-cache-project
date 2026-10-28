"""
power_logger_nvml.py — Background GPU power sampler using pynvml / nvidia-ml-py.

Reads GPU power via NVML every 0.1-0.2 s on a background thread.
The NVML import is deferred so this module can be imported safely on any machine;
a RuntimeError is raised only when start() is called on a non-NVML host.

Records idle GPU power at start() so callers can subtract it if desired.

Interface (identical to power_logger.PowerLogger and power_logger_fake.PowerLogger):
    logger = PowerLogger(interval=0.1, gpu_index=0)
    logger.start()
    # ... run inference ...
    logger.stop()
    avg_w, dur_s, energy_j = logger.summary()
    logger.save_raw("results/power_nvml.csv")

Additional attributes (after stop()):
    logger.idle_power_w    — mean power sampled before workload (W)
    logger.mean_power_w    — same as summary()[0]

POWER_BACKEND check:
    If pynvml is installed but nvmlDeviceGetPowerUsage returns N/A for a device
    (common on virtualised GPU instances), start() will print a clear warning and
    fall back to recording NaN samples. Summary will return (nan, dur, nan).
    Callers should check power_source in the CSV.
"""

import csv
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

# pynvml is imported lazily in start() to allow safe module import anywhere.
_pynvml_module = None  # cached after first successful import


class PowerLogger:
    """Samples GPU power from NVML on a background thread.

    Parameters
    ----------
    interval : float
        Sampling interval in seconds (default 0.1 s → 10 Hz).
        Must be in the range [0.05, 1.0].
    gpu_index : int
        Index of the GPU to monitor (default 0).
    idle_samples : int
        Number of samples to collect for idle baseline before signalling ready.
        Set to 0 to skip idle measurement.
    """

    def __init__(
        self,
        interval: float = 0.1,
        gpu_index: int = 0,
        idle_samples: int = 5,
    ) -> None:
        if not (0.05 <= interval <= 1.0):
            raise ValueError(f"interval must be in [0.05, 1.0], got {interval}")
        self._interval = interval
        self._gpu_index = gpu_index
        self._idle_samples = idle_samples

        self._samples: List[Tuple[float, float]] = []  # (timestamp_s, watts)
        self._idle_watts: List[float] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._start_time: Optional[float] = None
        self._stop_time: Optional[float] = None
        self._active_start: Optional[float] = None  # after idle baseline
        self._active_stop: Optional[float] = None   # taken before join
        self.idle_power_w: float = float("nan")

    # ──────────────────────────────────────────────────────────────────────────
    # Public API (matches power_logger.PowerLogger exactly)
    # ──────────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start background NVML power sampling.

        Raises
        ------
        RuntimeError
            If pynvml is not installed or NVML init fails.
        RuntimeError
            If the logger is already running.
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("PowerLogger (NVML) is already running; call stop() first.")

        # Deferred import — safe to import this module anywhere
        global _pynvml_module
        if _pynvml_module is None:
            try:
                import pynvml as _pynvml  # type: ignore[import]
                _pynvml_module = _pynvml
            except ImportError as exc:
                raise RuntimeError(
                    "pynvml is not installed. Install it with: pip install nvidia-ml-py\n"
                    f"Original error: {exc}"
                ) from exc

        try:
            _pynvml_module.nvmlInit()
        except Exception as exc:
            raise RuntimeError(
                f"NVML initialisation failed — no NVIDIA GPU available or driver error.\n"
                f"Original error: {exc}"
            ) from exc

        self._samples.clear()
        self._idle_watts.clear()
        self._stop_event.clear()
        self._ready_event.clear()
        self._start_time = time.perf_counter()
        self._active_start = None
        self._active_stop = None
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

        # Wait until idle measurement is complete (or 2 s timeout)
        if self._idle_samples > 0:
            self._ready_event.wait(timeout=max(2.0, self._interval * self._idle_samples * 3))

    def stop(self) -> None:
        """Stop background sampling and wait for the thread to finish."""
        if self._thread is None:
            return
        self._active_stop = time.perf_counter()
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        self._stop_time = time.perf_counter()
        self._thread = None
        try:
            assert _pynvml_module is not None
            _pynvml_module.nvmlShutdown()
        except Exception:
            pass

    def summary(self) -> Tuple[float, float, float]:
        """Return (avg_power_watts, duration_s, energy_joules).

        Returns (nan, duration, nan) if no valid samples were collected
        (e.g., NVML returned N/A for this device).

        Raises
        ------
        RuntimeError
            If start() and stop() have not both been called.
        """
        if self._start_time is None or self._stop_time is None:
            raise RuntimeError(
                "Call start() followed by stop() before reading summary()."
            )
        t_a = self._active_start if self._active_start is not None else self._start_time
        t_b = self._active_stop if self._active_stop is not None else self._stop_time
        duration = t_b - t_a
        if not self._samples:
            return float("nan"), duration, float("nan")

        watts = [w for _, w in self._samples if not (w != w)]  # filter NaN
        if not watts:
            return float("nan"), duration, float("nan")

        avg_power = sum(watts) / len(watts)
        energy = avg_power * duration
        return avg_power, duration, energy

    def save_raw(self, path: "str | Path") -> None:
        """Write raw samples to a CSV file.

        Columns: timestamp_s, power_watts
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["timestamp_s", "power_watts"])
            writer.writerows(self._samples)
        print(f"[PowerLogger/nvml] Saved {len(self._samples)} samples → {path}")

    # ──────────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────────

    def _sample_loop(self) -> None:
        """Background thread: poll NVML every self._interval seconds."""
        try:
            handle = _pynvml_module.nvmlDeviceGetHandleByIndex(self._gpu_index)
        except Exception as exc:
            print(f"[PowerLogger/nvml] Cannot get device handle: {exc}")
            self._ready_event.set()
            return

        # -- Idle baseline --
        idle_collected = 0
        idle_na_warned = False
        while idle_collected < self._idle_samples and not self._stop_event.is_set():
            pw = self._read_power_mw(handle, idle_na_warned)
            if pw is not None:
                self._idle_watts.append(pw / 1000.0)
                idle_collected += 1
            else:
                idle_na_warned = True
            self._stop_event.wait(timeout=self._interval)

        if self._idle_watts:
            self.idle_power_w = sum(self._idle_watts) / len(self._idle_watts)
        else:
            self.idle_power_w = float("nan")
        self._ready_event.set()

        self._active_start = time.perf_counter()
        # -- Active sampling --
        na_warned = idle_na_warned
        while not self._stop_event.is_set():
            pw = self._read_power_mw(handle, na_warned)
            ts = time.perf_counter() - self._start_time
            if pw is not None:
                self._samples.append((ts, pw / 1000.0))
            else:
                self._samples.append((ts, float("nan")))
                na_warned = True
            self._stop_event.wait(timeout=self._interval)

    def _read_power_mw(self, handle, already_warned: bool) -> Optional[int]:
        """Read power in mW from NVML, returning None if N/A."""
        try:
            return _pynvml_module.nvmlDeviceGetPowerUsage(handle)  # mW
        except _pynvml_module.NVMLError as exc:
            if not already_warned:
                print(
                    f"\n[PowerLogger/nvml] WARNING: nvmlDeviceGetPowerUsage returned N/A "
                    f"({exc}). This is common on virtualised GPUs (Kaggle T4, Colab). "
                    "Energy columns will be blank in the CSV. "
                    "Consider using --power_backend fake or a bare-metal GPU."
                )
            return None
