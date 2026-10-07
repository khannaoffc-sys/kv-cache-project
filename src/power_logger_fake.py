"""
power_logger_fake.py -- Drop-in replacement for PowerLogger using synthetic data.

Generates plausible fake board-power samples (uniform random 5-15 W) on the
same background-thread pattern as the real PowerLogger, so the interface is
100% identical and switching to real hardware is a one-line import change.

Interface
---------
    from power_logger_fake import PowerLogger

    logger = PowerLogger(interval=0.1)
    logger.start()
    # ... run inference ...
    logger.stop()

    avg_w, dur_s, energy_j = logger.summary()
    logger.save_raw("results/power_samples_fake.csv")
"""

import csv
import random
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple


class PowerLogger:
    """Fake power logger with the identical interface to the real PowerLogger.

    Instead of reading jtop, samples are drawn from Uniform(POWER_MIN, POWER_MAX)
    to simulate a small ARM/SoC board under moderate inference load.

    Parameters
    ----------
    interval : float
        Sampling interval in seconds (default 0.1 s -> 10 Hz).
        Must be in [0.05, 1.0].
    power_min : float
        Lower bound of the fake power range in watts (default 5.0 W).
    power_max : float
        Upper bound of the fake power range in watts (default 15.0 W).
    """

    # Plausible idle-to-peak power range for a Jetson Orin Nano / RPi 5
    POWER_MIN: float = 5.0
    POWER_MAX: float = 15.0

    def __init__(
        self,
        interval: float = 0.1,
        power_min: float = POWER_MIN,
        power_max: float = POWER_MAX,
    ) -> None:
        if not (0.05 <= interval <= 1.0):
            raise ValueError(f"interval must be in [0.05, 1.0], got {interval}")
        if power_min >= power_max:
            raise ValueError(
                f"power_min ({power_min}) must be less than power_max ({power_max})"
            )
        self._interval: float = interval
        self._power_min: float = power_min
        self._power_max: float = power_max
        self._samples: List[Tuple[float, float]] = []  # (timestamp_s, watts)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._start_time: Optional[float] = None
        self._stop_time: Optional[float] = None

    # -------------------------------------------------------------------------
    # Public API  (mirrors power_logger.PowerLogger exactly)
    # -------------------------------------------------------------------------

    def start(self) -> None:
        """Start background fake-power sampling.

        Raises
        ------
        RuntimeError
            If the logger is already running.
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError(
                "PowerLogger (fake) is already running; call stop() first."
            )
        self._samples.clear()
        self._stop_event.clear()
        self._start_time = time.perf_counter()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop background sampling and wait for the thread to finish."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        self._stop_time = time.perf_counter()
        self._thread = None

    def summary(self) -> Tuple[float, float, float]:
        """Return (avg_power_watts, duration_s, energy_joules).

        Returns
        -------
        avg_power_watts : float
        duration_s      : float
        energy_joules   : float

        Raises
        ------
        RuntimeError
            If start() / stop() have not both been called.
        """
        if self._start_time is None or self._stop_time is None:
            raise RuntimeError(
                "Call start() followed by stop() before reading summary()."
            )
        if not self._samples:
            return 0.0, self._stop_time - self._start_time, 0.0

        watts = [w for _, w in self._samples]
        avg_power = sum(watts) / len(watts)
        duration = self._stop_time - self._start_time
        energy = avg_power * duration
        return avg_power, duration, energy

    def save_raw(self, path: "str | Path") -> None:
        """Write raw samples to a CSV file.

        Columns: ``timestamp_s``, ``power_watts``.

        Parameters
        ----------
        path : str or Path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["timestamp_s", "power_watts"])
            writer.writerows(self._samples)
        print(
            f"[PowerLogger/fake] Saved {len(self._samples)} samples -> {path}"
        )

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _sample_loop(self) -> None:
        """Background thread: generate synthetic power readings."""
        while not self._stop_event.is_set():
            power_w = random.uniform(self._power_min, self._power_max)
            ts = time.perf_counter() - self._start_time
            self._samples.append((ts, power_w))
            self._stop_event.wait(timeout=self._interval)
