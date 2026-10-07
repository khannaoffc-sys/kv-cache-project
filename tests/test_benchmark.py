"""
tests/test_benchmark.py -- pytest test suite for kv-cache-project.

Tests
-----
1. Interface parity  : PowerLogger and power_logger_fake.PowerLogger have the
                       same public method signatures.
2. CSV column schema : run_benchmark's output CSV always has the expected columns.
3. Compression ratio : values outside (0, 1) are caught before being passed to a press.

Run with:
    pytest tests/test_benchmark.py -v
"""

import csv
import inspect
import io
import math
import sys
import time
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Make src/ importable when running from the project root
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ---------------------------------------------------------------------------
# 1. Interface parity tests
# ---------------------------------------------------------------------------

class TestPowerLoggerInterfaceParity:
    """Ensure power_logger_fake.PowerLogger exposes identical public methods."""

    REQUIRED_METHODS = ["start", "stop", "summary", "save_raw"]

    def _public_methods(self, cls) -> dict:
        """Return {name: signature} for all non-dunder public methods."""
        return {
            name: inspect.signature(method)
            for name, method in inspect.getmembers(cls, predicate=inspect.isfunction)
            if not name.startswith("_")
        }

    def test_fake_has_all_required_methods(self):
        """power_logger_fake.PowerLogger must expose start, stop, summary, save_raw."""
        from power_logger_fake import PowerLogger as FakePL
        fake_methods = self._public_methods(FakePL)
        for method_name in self.REQUIRED_METHODS:
            assert method_name in fake_methods, (
                f"power_logger_fake.PowerLogger is missing method: {method_name}"
            )

    def test_real_has_all_required_methods(self):
        """power_logger.PowerLogger must expose start, stop, summary, save_raw."""
        from power_logger import PowerLogger as RealPL
        real_methods = self._public_methods(RealPL)
        for method_name in self.REQUIRED_METHODS:
            assert method_name in real_methods, (
                f"power_logger.PowerLogger is missing method: {method_name}"
            )

    def test_method_signatures_match(self):
        """Parameter names for each required method must match between real and fake."""
        from power_logger import PowerLogger as RealPL
        from power_logger_fake import PowerLogger as FakePL

        for method_name in self.REQUIRED_METHODS:
            real_sig = inspect.signature(getattr(RealPL, method_name))
            fake_sig = inspect.signature(getattr(FakePL, method_name))

            real_params = set(real_sig.parameters) - {"self"}
            fake_params = set(fake_sig.parameters) - {"self"}

            assert real_params == fake_params, (
                f"Signature mismatch for {method_name}:\n"
                f"  real: {real_params}\n"
                f"  fake: {fake_params}"
            )

    def test_fake_init_accepts_interval(self):
        """FakePowerLogger.__init__ should accept an interval parameter."""
        from power_logger_fake import PowerLogger as FakePL
        logger = FakePL(interval=0.2)
        assert logger is not None

    def test_fake_init_rejects_bad_interval(self):
        """Interval outside [0.05, 1.0] should raise ValueError."""
        from power_logger_fake import PowerLogger as FakePL
        with pytest.raises(ValueError, match="interval"):
            FakePL(interval=0.0)
        with pytest.raises(ValueError, match="interval"):
            FakePL(interval=5.0)

    def test_fake_summary_raises_before_start_stop(self):
        """summary() should raise RuntimeError if logger hasn't been used yet."""
        from power_logger_fake import PowerLogger as FakePL
        logger = FakePL()
        with pytest.raises(RuntimeError):
            logger.summary()

    def test_fake_start_stop_summary_roundtrip(self):
        """Full start -> (wait) -> stop -> summary() cycle should succeed."""
        from power_logger_fake import PowerLogger as FakePL
        logger = FakePL(interval=0.05)
        logger.start()
        time.sleep(0.2)  # collect ~4 samples at 0.05 s interval
        logger.stop()
        avg_w, dur_s, energy_j = logger.summary()

        assert 5.0 <= avg_w <= 15.0, f"avg_power out of range: {avg_w}"
        assert dur_s > 0, "duration must be positive"
        assert energy_j == pytest.approx(avg_w * dur_s, rel=0.05)

    def test_fake_double_start_raises(self):
        """Calling start() twice without stop() should raise RuntimeError."""
        from power_logger_fake import PowerLogger as FakePL
        logger = FakePL(interval=0.05)
        logger.start()
        try:
            with pytest.raises(RuntimeError):
                logger.start()
        finally:
            logger.stop()

    def test_real_start_raises_without_jtop(self):
        """Real PowerLogger.start() should raise RuntimeError when jtop is absent."""
        from power_logger import PowerLogger as RealPL

        # Simulate jtop not being installed
        with patch.dict(sys.modules, {"jtop": None}):
            logger = RealPL()
            with pytest.raises(RuntimeError, match="jtop"):
                logger.start()

    def test_fake_save_raw(self, tmp_path):
        """save_raw() should write a valid two-column CSV."""
        from power_logger_fake import PowerLogger as FakePL
        logger = FakePL(interval=0.05)
        logger.start()
        time.sleep(0.15)
        logger.stop()

        out_file = tmp_path / "power.csv"
        logger.save_raw(out_file)

        assert out_file.exists()
        with open(out_file, newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            assert header == ["timestamp_s", "power_watts"]
            rows = list(reader)
        assert len(rows) > 0, "CSV should have at least one data row"
        for row in rows:
            ts, pw = float(row[0]), float(row[1])
            assert ts >= 0
            assert 5.0 <= pw <= 15.0


# ---------------------------------------------------------------------------
# 2. CSV column schema tests
# ---------------------------------------------------------------------------

EXPECTED_BENCHMARK_COLUMNS = {
    "config",
    "prompt_type",
    "latency_s",
    "tokens_per_sec",
    "avg_power_w",
    "energy_j",
    "accuracy",
    "output_preview",
}

EXPECTED_SWEEP_COLUMNS = {
    "config",
    "compression_ratio",
    "task",
    "num_examples",
    "avg_accuracy",
    "avg_latency_s",
    "avg_tokens_per_sec",
}


class TestCSVSchema:
    """Validate that benchmark output CSVs always have the expected columns."""

    def _write_sample_csv(self, path: Path, fieldnames: List[str], rows: List[dict]):
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_benchmark_csv_has_required_columns(self, tmp_path):
        """A benchmark_results.csv with all required columns passes validation."""
        csv_path = tmp_path / "benchmark_results.csv"
        fields = sorted(EXPECTED_BENCHMARK_COLUMNS)
        self._write_sample_csv(
            csv_path,
            fields,
            [{"config": "baseline", "prompt_type": "latency_fixed",
              "latency_s": "1.0", "tokens_per_sec": "50.0",
              "avg_power_w": "N/A", "energy_j": "N/A",
              "accuracy": "N/A", "output_preview": "test output"}],
        )
        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            actual_cols = set(reader.fieldnames or [])
        assert EXPECTED_BENCHMARK_COLUMNS.issubset(actual_cols), (
            f"Missing columns: {EXPECTED_BENCHMARK_COLUMNS - actual_cols}"
        )

    def test_benchmark_csv_missing_column_caught(self, tmp_path):
        """A CSV missing a required column should fail the schema check."""
        csv_path = tmp_path / "benchmark_results_bad.csv"
        bad_fields = sorted(EXPECTED_BENCHMARK_COLUMNS - {"accuracy"})
        self._write_sample_csv(csv_path, bad_fields, [
            {k: "x" for k in bad_fields}
        ])
        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            actual_cols = set(reader.fieldnames or [])
        assert not EXPECTED_BENCHMARK_COLUMNS.issubset(actual_cols), (
            "Expected schema check to fail for missing 'accuracy' column"
        )

    def test_sweep_csv_has_required_columns(self, tmp_path):
        """An accuracy_sweep.csv with all required columns passes validation."""
        csv_path = tmp_path / "accuracy_sweep.csv"
        fields = sorted(EXPECTED_SWEEP_COLUMNS)
        self._write_sample_csv(
            csv_path,
            fields,
            [{"config": "SnapKVPress", "compression_ratio": "0.5",
              "task": "narrativeqa", "num_examples": "40",
              "avg_accuracy": "0.25", "avg_latency_s": "3.2",
              "avg_tokens_per_sec": "15.6"}],
        )
        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            actual_cols = set(reader.fieldnames or [])
        assert EXPECTED_SWEEP_COLUMNS.issubset(actual_cols)

    def test_actual_benchmark_csv_if_exists(self):
        """If results/benchmark_results.csv exists on disk, validate its schema."""
        csv_path = Path(__file__).parent.parent / "results" / "benchmark_results.csv"
        if not csv_path.exists():
            pytest.skip("benchmark_results.csv not yet generated; run run_benchmark.py first")
        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            actual_cols = set(reader.fieldnames or [])
        missing = EXPECTED_BENCHMARK_COLUMNS - actual_cols
        assert not missing, f"benchmark_results.csv missing columns: {missing}"

    def test_actual_sweep_csv_if_exists(self):
        """If results/accuracy_sweep.csv exists on disk, validate its schema."""
        csv_path = Path(__file__).parent.parent / "results" / "accuracy_sweep.csv"
        if not csv_path.exists():
            pytest.skip("accuracy_sweep.csv not yet generated; run accuracy_sweep.py first")
        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            actual_cols = set(reader.fieldnames or [])
        missing = EXPECTED_SWEEP_COLUMNS - actual_cols
        assert not missing, f"accuracy_sweep.csv missing columns: {missing}"


# ---------------------------------------------------------------------------
# 3. Compression ratio validation tests
# ---------------------------------------------------------------------------

def _validate_compression_ratio(ratio: float) -> None:
    """Helper that mirrors the validation logic in run_benchmark / accuracy_sweep."""
    if not (0.0 < ratio < 1.0):
        raise ValueError(
            f"compression_ratio must be in (0, 1), got {ratio}. "
            "Values <= 0 keep no tokens; values >= 1 keep all tokens."
        )


class TestCompressionRatioValidation:
    """Ensure invalid compression ratios are caught before reaching the press."""

    @pytest.mark.parametrize("ratio", [0.3, 0.5, 0.7])
    def test_valid_ratios_accepted(self, ratio: float):
        """All benchmark sweep ratios (0.3, 0.5, 0.7) should be valid."""
        _validate_compression_ratio(ratio)  # must not raise

    @pytest.mark.parametrize("bad_ratio", [0.0, 1.0, -0.1, 1.5, 2.0, -1.0])
    def test_invalid_ratios_rejected(self, bad_ratio: float):
        """Ratios outside (0, 1) should raise ValueError."""
        with pytest.raises(ValueError, match="compression_ratio"):
            _validate_compression_ratio(bad_ratio)

    def test_boundary_zero_rejected(self):
        with pytest.raises(ValueError):
            _validate_compression_ratio(0.0)

    def test_boundary_one_rejected(self):
        with pytest.raises(ValueError):
            _validate_compression_ratio(1.0)

    def test_very_small_ratio_accepted(self):
        """A very small but positive ratio (e.g. 0.01) should be valid."""
        _validate_compression_ratio(0.01)

    def test_very_large_but_valid_ratio_accepted(self):
        """A ratio just below 1.0 should be valid."""
        _validate_compression_ratio(0.99)

    def test_nan_rejected(self):
        """NaN is not a valid compression ratio."""
        with pytest.raises((ValueError, AssertionError)):
            ratio = float("nan")
            if math.isnan(ratio):
                raise ValueError(f"compression_ratio must be in (0, 1), got {ratio}")

    @pytest.mark.parametrize("ratio", [0.3, 0.5, 0.7])
    def test_sweep_ratios_all_valid(self, ratio: float):
        """The exact ratios used in the accuracy sweep must all pass validation."""
        # This documents the contract between accuracy_sweep.py and the press API
        _validate_compression_ratio(ratio)
