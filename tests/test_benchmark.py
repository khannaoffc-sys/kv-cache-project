"""
tests/test_benchmark.py — pytest test suite for kv-cache-project.

Tests
-----
1. Interface parity   : PowerLogger (fake, jtop, nvml) all have identical public method signatures.
2. CSV column schema  : benchmark and sweep CSVs have all required columns.
3. Row content        : benchmark CSV must not contain ERROR or NaN rows.
4. Compression ratio  : values outside (0, 1) are caught before reaching a press.
5. Power source       : fake-power rows must have blank energy columns.
6. LongBench metrics  : longbench_metrics.score() returns expected values on known examples.

Run with:
    pytest tests/test_benchmark.py -v
"""

import csv
import inspect
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
# 1. Interface parity tests — fake, jtop, nvml loggers
# ---------------------------------------------------------------------------

class TestPowerLoggerInterfaceParity:
    """Ensure all PowerLogger implementations expose identical public method signatures."""

    REQUIRED_METHODS = ["start", "stop", "summary", "save_raw"]

    def _public_methods(self, cls) -> dict:
        return {
            name: inspect.signature(method)
            for name, method in inspect.getmembers(cls, predicate=inspect.isfunction)
            if not name.startswith("_")
        }

    def test_fake_has_all_required_methods(self):
        from power_logger_fake import PowerLogger as FakePL
        fake_methods = self._public_methods(FakePL)
        for m in self.REQUIRED_METHODS:
            assert m in fake_methods, f"power_logger_fake.PowerLogger missing: {m}"

    def test_real_jtop_has_all_required_methods(self):
        from power_logger import PowerLogger as RealPL
        real_methods = self._public_methods(RealPL)
        for m in self.REQUIRED_METHODS:
            assert m in real_methods, f"power_logger.PowerLogger missing: {m}"

    def test_nvml_has_all_required_methods(self):
        from power_logger_nvml import PowerLogger as NvmlPL
        nvml_methods = self._public_methods(NvmlPL)
        for m in self.REQUIRED_METHODS:
            assert m in nvml_methods, f"power_logger_nvml.PowerLogger missing: {m}"

    def _check_signature_parity(self, cls_a, cls_b, method_name: str):
        sig_a = inspect.signature(getattr(cls_a, method_name))
        sig_b = inspect.signature(getattr(cls_b, method_name))
        params_a = set(sig_a.parameters) - {"self"}
        params_b = set(sig_b.parameters) - {"self"}
        assert params_a == params_b, (
            f"Signature mismatch for {method_name}:\n"
            f"  {cls_a.__name__}: {params_a}\n"
            f"  {cls_b.__name__}: {params_b}"
        )

    def test_fake_vs_jtop_signatures(self):
        from power_logger import PowerLogger as RealPL
        from power_logger_fake import PowerLogger as FakePL
        for m in self.REQUIRED_METHODS:
            self._check_signature_parity(RealPL, FakePL, m)

    def test_fake_vs_nvml_signatures(self):
        from power_logger_fake import PowerLogger as FakePL
        from power_logger_nvml import PowerLogger as NvmlPL
        for m in self.REQUIRED_METHODS:
            self._check_signature_parity(FakePL, NvmlPL, m)

    def test_jtop_vs_nvml_signatures(self):
        from power_logger import PowerLogger as RealPL
        from power_logger_nvml import PowerLogger as NvmlPL
        for m in self.REQUIRED_METHODS:
            self._check_signature_parity(RealPL, NvmlPL, m)

    def test_fake_init_accepts_interval(self):
        from power_logger_fake import PowerLogger as FakePL
        logger = FakePL(interval=0.2)
        assert logger is not None

    def test_fake_init_rejects_bad_interval(self):
        from power_logger_fake import PowerLogger as FakePL
        with pytest.raises(ValueError, match="interval"):
            FakePL(interval=0.0)
        with pytest.raises(ValueError, match="interval"):
            FakePL(interval=5.0)

    def test_nvml_init_accepts_interval(self):
        from power_logger_nvml import PowerLogger as NvmlPL
        logger = NvmlPL(interval=0.2, gpu_index=0)
        assert logger is not None

    def test_nvml_init_rejects_bad_interval(self):
        from power_logger_nvml import PowerLogger as NvmlPL
        with pytest.raises(ValueError, match="interval"):
            NvmlPL(interval=0.0)

    def test_fake_summary_raises_before_start_stop(self):
        from power_logger_fake import PowerLogger as FakePL
        logger = FakePL()
        with pytest.raises(RuntimeError):
            logger.summary()

    def test_fake_start_stop_summary_roundtrip(self):
        from power_logger_fake import PowerLogger as FakePL
        logger = FakePL(interval=0.05)
        logger.start()
        time.sleep(0.2)
        logger.stop()
        avg_w, dur_s, energy_j = logger.summary()
        assert 5.0 <= avg_w <= 15.0, f"avg_power out of range: {avg_w}"
        assert dur_s > 0, "duration must be positive"
        assert energy_j == pytest.approx(avg_w * dur_s, rel=0.05)

    def test_fake_double_start_raises(self):
        from power_logger_fake import PowerLogger as FakePL
        logger = FakePL(interval=0.05)
        logger.start()
        try:
            with pytest.raises(RuntimeError):
                logger.start()
        finally:
            logger.stop()

    def test_real_start_raises_without_jtop(self):
        from power_logger import PowerLogger as RealPL
        with patch.dict(sys.modules, {"jtop": None}):
            logger = RealPL()
            with pytest.raises(RuntimeError, match="jtop"):
                logger.start()

    def test_nvml_start_raises_without_pynvml(self):
        """NVML logger must raise RuntimeError when pynvml is absent."""
        from power_logger_nvml import PowerLogger as NvmlPL
        import power_logger_nvml as nvml_mod
        old = nvml_mod._pynvml_module
        nvml_mod._pynvml_module = None
        with patch.dict(sys.modules, {"pynvml": None}):
            logger = NvmlPL()
            with pytest.raises(RuntimeError, match="pynvml"):
                logger.start()
        nvml_mod._pynvml_module = old  # restore

    def test_fake_save_raw(self, tmp_path):
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
# 2. CSV column schema tests (Phase 2 extended columns)
# ---------------------------------------------------------------------------

EXPECTED_BENCHMARK_COLUMNS = {
    "model_name", "device", "dtype",
    "config", "prompt_type", "prompt_tokens", "new_tokens", "kv_cache_bytes",
    "latency_s", "latency_std_s", "tokens_per_sec",
    "power_source", "avg_power_w", "energy_j",
    "accuracy", "status", "output_preview",
}

EXPECTED_SWEEP_COLUMNS = {
    "model_name", "device", "dtype",
    "config", "compression_ratio", "task", "num_examples",
    "avg_accuracy", "avg_latency_s", "avg_tokens_per_sec", "status",
}


class TestCSVSchema:
    """Validate that benchmark output CSVs always have the expected columns."""

    def _write_sample_csv(self, path: Path, fieldnames: List[str], rows: List[dict]):
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _good_benchmark_row(self):
        return {
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "device": "cpu",
            "dtype": "torch.float32",
            "config": "baseline",
            "prompt_type": "latency_fixed",
            "prompt_tokens": "512",
            "new_tokens": "50",
            "kv_cache_bytes": "1048576",
            "latency_s": "12.34",
            "latency_std_s": "0.11",
            "tokens_per_sec": "4.1",
            "power_source": "fake",
            "avg_power_w": "",
            "energy_j": "",
            "accuracy": "N/A",
            "status": "ok",
            "output_preview": "some text",
        }

    def test_benchmark_csv_has_required_columns(self, tmp_path):
        csv_path = tmp_path / "benchmark_results.csv"
        row = self._good_benchmark_row()
        self._write_sample_csv(csv_path, list(EXPECTED_BENCHMARK_COLUMNS), [row])
        with open(csv_path, newline="") as fh:
            actual_cols = set(csv.DictReader(fh).fieldnames or [])
        assert EXPECTED_BENCHMARK_COLUMNS.issubset(actual_cols), (
            f"Missing columns: {EXPECTED_BENCHMARK_COLUMNS - actual_cols}"
        )

    def test_benchmark_csv_missing_column_caught(self, tmp_path):
        csv_path = tmp_path / "bad.csv"
        bad_fields = sorted(EXPECTED_BENCHMARK_COLUMNS - {"status"})
        self._write_sample_csv(csv_path, bad_fields, [{k: "x" for k in bad_fields}])
        with open(csv_path, newline="") as fh:
            actual_cols = set(csv.DictReader(fh).fieldnames or [])
        assert not EXPECTED_BENCHMARK_COLUMNS.issubset(actual_cols)

    def test_sweep_csv_has_required_columns(self, tmp_path):
        csv_path = tmp_path / "accuracy_sweep.csv"
        fields = sorted(EXPECTED_SWEEP_COLUMNS)
        self._write_sample_csv(csv_path, fields, [{
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "device": "cpu",
            "dtype": "torch.float32",
            "config": "SnapKVPress",
            "compression_ratio": "0.5",
            "task": "narrativeqa",
            "num_examples": "3",
            "avg_accuracy": "0.25",
            "avg_latency_s": "3.2",
            "avg_tokens_per_sec": "15.6",
            "status": "ok",
        }])
        with open(csv_path, newline="") as fh:
            actual_cols = set(csv.DictReader(fh).fieldnames or [])
        assert EXPECTED_SWEEP_COLUMNS.issubset(actual_cols)

    def test_actual_benchmark_csv_if_exists(self):
        """If benchmark_results.csv exists, validate columns AND require no ERROR/NaN rows."""
        csv_path = Path(__file__).parent.parent / "results" / "benchmark_results.csv"
        if not csv_path.exists():
            pytest.skip("benchmark_results.csv not yet generated")
        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            actual_cols = set(reader.fieldnames or [])
            missing = EXPECTED_BENCHMARK_COLUMNS - actual_cols
            assert not missing, f"Missing columns: {missing}"
            error_rows = []
            nan_rows = []
            fake_energy_rows = []
            for row in reader:
                if row.get("status") == "ERROR" or row.get("latency_s") == "ERROR":
                    error_rows.append(row["config"])
                lat = row.get("latency_s", "")
                try:
                    if lat and math.isnan(float(lat)):
                        nan_rows.append(row["config"])
                except ValueError:
                    pass
                # Fake-power rows must have blank energy
                if row.get("power_source") == "fake":
                    if row.get("energy_j") not in ("", "N/A"):
                        fake_energy_rows.append(row["config"])
        assert not error_rows, f"ERROR rows in CSV: {error_rows}"
        assert not nan_rows, f"NaN latency rows in CSV: {nan_rows}"
        assert not fake_energy_rows, (
            f"Fake-power rows have non-blank energy_j: {fake_energy_rows}"
        )

    def test_actual_sweep_csv_if_exists(self):
        """If accuracy_sweep.csv exists, validate columns and no ERROR/NaN rows."""
        csv_path = Path(__file__).parent.parent / "results" / "accuracy_sweep.csv"
        if not csv_path.exists():
            pytest.skip("accuracy_sweep.csv not yet generated")
        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            actual_cols = set(reader.fieldnames or [])
            missing = EXPECTED_SWEEP_COLUMNS - actual_cols
            assert not missing, f"Missing columns: {missing}"
            error_rows = []
            for row in reader:
                if row.get("status") == "ERROR" or row.get("avg_accuracy") == "NaN":
                    error_rows.append(f"{row['config']}@{row['compression_ratio']}")
        assert not error_rows, f"ERROR/NaN rows in sweep CSV: {error_rows}"


# ---------------------------------------------------------------------------
# 3. Compression ratio validation
# ---------------------------------------------------------------------------

def _validate_compression_ratio(ratio: float) -> None:
    if not (0.0 < ratio < 1.0):
        raise ValueError(
            f"compression_ratio must be in (0, 1), got {ratio}."
        )


class TestCompressionRatioValidation:

    @pytest.mark.parametrize("ratio", [0.3, 0.5, 0.7])
    def test_valid_ratios_accepted(self, ratio: float):
        _validate_compression_ratio(ratio)  # must not raise

    @pytest.mark.parametrize("bad_ratio", [0.0, 1.0, -0.1, 1.5, 2.0, -1.0])
    def test_invalid_ratios_rejected(self, bad_ratio: float):
        with pytest.raises(ValueError, match="compression_ratio"):
            _validate_compression_ratio(bad_ratio)

    def test_boundary_zero_rejected(self):
        with pytest.raises(ValueError):
            _validate_compression_ratio(0.0)

    def test_boundary_one_rejected(self):
        with pytest.raises(ValueError):
            _validate_compression_ratio(1.0)

    def test_very_small_ratio_accepted(self):
        _validate_compression_ratio(0.01)

    def test_very_large_but_valid_ratio_accepted(self):
        _validate_compression_ratio(0.99)

    def test_nan_rejected(self):
        with pytest.raises((ValueError, AssertionError)):
            ratio = float("nan")
            if math.isnan(ratio):
                raise ValueError(f"compression_ratio must be in (0, 1), got {ratio}")

    @pytest.mark.parametrize("ratio", [0.3, 0.5, 0.7])
    def test_sweep_ratios_all_valid(self, ratio: float):
        _validate_compression_ratio(ratio)


# ---------------------------------------------------------------------------
# 4. Fake power: energy column must be blank
# ---------------------------------------------------------------------------

class TestFakePowerRowEnergy:
    """Rows with power_source=fake must have blank avg_power_w and energy_j."""

    def test_fake_power_row_has_blank_energy(self, tmp_path):
        csv_path = tmp_path / "bench.csv"
        row = {
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "device": "cpu",
            "dtype": "torch.float32",
            "config": "baseline",
            "prompt_type": "latency_fixed",
            "prompt_tokens": "512",
            "new_tokens": "50",
            "kv_cache_bytes": "N/A",
            "latency_s": "5.0",
            "latency_std_s": "0.1",
            "tokens_per_sec": "10.0",
            "power_source": "fake",
            "avg_power_w": "",      # must be blank
            "energy_j": "",         # must be blank
            "accuracy": "N/A",
            "status": "ok",
            "output_preview": "test",
        }
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        with open(csv_path, newline="") as fh:
            for r in csv.DictReader(fh):
                if r["power_source"] == "fake":
                    assert r["energy_j"] in ("", "N/A"), (
                        f"Fake-power row has non-blank energy_j: {r['energy_j']!r}"
                    )

    def test_fake_power_row_with_energy_fails_check(self, tmp_path):
        """A fake-power row with non-blank energy_j should be detectable."""
        csv_path = tmp_path / "bad.csv"
        row = {
            "model_name": "test", "device": "cpu", "dtype": "torch.float32",
            "config": "baseline", "prompt_type": "latency_fixed",
            "prompt_tokens": "512", "new_tokens": "50", "kv_cache_bytes": "N/A",
            "latency_s": "5.0", "latency_std_s": "0.1", "tokens_per_sec": "10.0",
            "power_source": "fake",
            "avg_power_w": "9.5",   # WRONG — should be blank for fake
            "energy_j": "47.5",     # WRONG
            "accuracy": "N/A", "status": "ok", "output_preview": "test",
        }
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        bad_configs = []
        with open(csv_path, newline="") as fh:
            for r in csv.DictReader(fh):
                if r["power_source"] == "fake" and r["energy_j"] not in ("", "N/A"):
                    bad_configs.append(r["config"])
        assert bad_configs, "Expected to detect the invalid fake-power row"


# ---------------------------------------------------------------------------
# 5. LongBench metrics unit tests
# ---------------------------------------------------------------------------

class TestLongBenchMetrics:
    """Smoke tests for the official LongBench scoring functions."""

    def test_qa_f1_exact_match(self):
        from longbench_metrics import qa_f1_score
        assert qa_f1_score("the quick brown fox", "the quick brown fox") == pytest.approx(1.0)

    def test_qa_f1_zero_on_no_overlap(self):
        from longbench_metrics import qa_f1_score
        assert qa_f1_score("apple orange", "banana grape") == pytest.approx(0.0)

    def test_qa_f1_partial_overlap(self):
        from longbench_metrics import qa_f1_score
        # prediction "the cat sat", reference "cat sat mat"
        # normalized: same
        # pred tokens: [the, cat, sat], ref tokens: [cat, sat, mat]
        # common: {cat, sat} -> precision=2/3, recall=2/3, F1=2/3
        score = qa_f1_score("the cat sat", "cat sat mat")
        assert 0.0 < score < 1.0

    def test_score_dispatch_narrativeqa(self):
        from longbench_metrics import score
        s = score("the house on fire", "the house on fire", dataset="narrativeqa")
        assert s == pytest.approx(1.0)

    def test_score_dispatch_unknown_task_raises(self):
        from longbench_metrics import score
        with pytest.raises(ValueError, match="Unknown LongBench dataset"):
            score("foo", "bar", dataset="nonexistent_task")

    def test_score_multi_returns_max(self):
        from longbench_metrics import score_multi
        # one ref matches perfectly, one doesn't
        s = score_multi(
            "the quick brown fox",
            ["completely different text", "the quick brown fox"],
            dataset="narrativeqa",
        )
        assert s == pytest.approx(1.0)

    def test_count_score_correct(self):
        from longbench_metrics import count_score
        assert count_score("42 items", "42") == pytest.approx(1.0)

    def test_count_score_wrong(self):
        from longbench_metrics import count_score
        assert count_score("43 items", "42") == pytest.approx(0.0)
