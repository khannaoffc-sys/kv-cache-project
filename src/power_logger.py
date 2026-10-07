"""
power_logger.py — Background power sampler for NVIDIA Jetson boards.

Uses jtop (jetson-stats) to read total board power every ~0.1–0.2 s on a
background thread.  The jtop import is intentionally deferred so this module
can be imported safely on any machine; a RuntimeError is raised only when
start() is actually called on a non-Jetson / non-jtop host.

Typical usage
─────────────
    from power_logger import PowerLogger
    import time

    logger = PowerLogger(interval=0.1)
    logger.start()
    # ... run inference ...
    logger.stop()

    avg_w, dur_s, energy_j = logger.summary()
    print(f"Average power : {avg_w:.2f} W")
    print(f"Duration      : {dur_s:.3f} s")
    print(f"Energy        : {energy_j:.3f} J")

    logger.save_raw("results/power_samples.csv")
"""

import csv
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple


class PowerLogger:
    """Samples total board power from jtop on a background thread.

    Parameters
    ----------
    interval : float
        Sampling interval in seconds (default 0.1 s → 10 Hz).
        Must be in the range [0.05, 1.0].
    """

    def __init__(self, interval: float = 0.1) -> None:
        if not (0.05 <= interval <= 1.0):
            raise ValueError(f"interval must be in [0.05, 1.0], got {interval}")
        self._interval: float = interval
        self._samples: List[Tuple[float, float]] = []  # (timestamp, watts)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._start_time: Optional[float] = None
        self._stop_time: Optional[float] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start background sampling.

        Raises
        ------
        RuntimeError
            If jtop / jetson-stats is not installed (non-Jetson host).
        RuntimeError
            If the logger is already running.
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("PowerLogger is already running; call stop() first.")

        # Deferred import — safe to import this module anywhere
        try:
            from jtop import jtop as _jtop  # noqa: F401 — just validate presence
        except ImportError as exc:
            raise RuntimeError(
                "jtop (jetson-stats) is not installed or not available on this host. "
                "Install it on a Jetson device with: sudo pip3 install jetson-stats\n"
                f"Original error: {exc}"
            ) from exc

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
            Mean sampled power in watts.
        duration_s : float
            Wall-clock duration between start() and stop() in seconds.
        energy_joules : float
            Estimated energy (avg_power × duration) in joules.

        Raises
        ------
        RuntimeError
            If the logger has not been started and stopped yet.
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

    def save_raw(self, path: str | Path) -> None:
        """Write raw samples to a CSV file.

        The CSV has two columns: ``timestamp_s`` (seconds since start) and
        ``power_watts``.

        Parameters
        ----------
        path : str or Path
            Destination file path.  Parent directories are created if needed.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["timestamp_s", "power_watts"])
            writer.writerows(self._samples)
        print(f"[PowerLogger] Saved {len(self._samples)} samples → {path}")

    # ─────────────────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────────────────

    def _sample_loop(self) -> None:
        """Background thread: open jtop and poll every self._interval seconds."""
        try:
            from jtop import jtop
        except ImportError:
            return  # Already guarded in start(); this is a safety net

        try:
            with jtop() as jetson:
                while not self._stop_event.is_set():
                    try:
                        power_w = jetson.power["tot"]["power"] / 1_000.0  # mW → W
                    except (KeyError, TypeError):
                        # Jetson firmware version differences — key may vary
                        try:
                            power_w = jetson.power["tot"]["cur"] / 1_000.0
                        except (KeyError, TypeError):
                            power_w = float("nan")

                    ts = time.perf_counter() - self._start_time
                    self._samples.append((ts, power_w))
                    self._stop_event.wait(timeout=self._interval)
        except Exception as exc:  # noqa: BLE001
            print(f"[PowerLogger] Sampling thread error: {exc}")
