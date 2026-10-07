"""
check_env.py — Environment sanity checker for kv-cache-project.

Checks:
  1. PyTorch is installed; reports CUDA availability (non-fatal if False).
  2. transformers is importable; prints its version.
  3. kvpress is importable and KnormPress, SnapKVPress, ObservedAttentionPress
     can each be imported individually.

Exits with code 0 even when CUDA is absent — this is expected on a laptop.
"""

import sys


def _ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


# ---------------------------------------------------------------------------
# Check 1 - PyTorch
# ---------------------------------------------------------------------------
print("\n-- Check 1: PyTorch " + "-"*52)
try:
    import torch  # noqa: F401

    _ok(f"torch {torch.__version__} imported successfully")

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        _ok(f"CUDA is available - device: {device_name}")
    else:
        print("  [INFO] CUDA is NOT available (running on CPU - expected on laptop)")
except ImportError as exc:
    _fail(f"torch could not be imported: {exc}")

# ---------------------------------------------------------------------------
# Check 2 - transformers
# ---------------------------------------------------------------------------
print("\n-- Check 2: transformers " + "-"*47)
try:
    import transformers  # noqa: F401

    _ok(f"transformers {transformers.__version__} imported successfully")
except ImportError as exc:
    _fail(f"transformers could not be imported: {exc}")

# ---------------------------------------------------------------------------
# Check 3 - kvpress + specific press classes
# ---------------------------------------------------------------------------
print("\n-- Check 3: kvpress " + "-"*51)
try:
    import kvpress  # noqa: F401

    _ok("kvpress top-level package imported successfully")
except ImportError as exc:
    _fail(f"kvpress could not be imported: {exc}")
    print()
    sys.exit(1)  # No point checking sub-imports if the package itself is missing

_press_classes = ["KnormPress", "SnapKVPress", "ObservedAttentionPress"]
for cls_name in _press_classes:
    try:
        obj = getattr(kvpress, cls_name)
        _ok(f"kvpress.{cls_name} is accessible ({obj})")
    except AttributeError:
        # Fall back to trying a direct import from the module
        try:
            import importlib

            mod = importlib.import_module(f"kvpress.{cls_name.lower()}")
            obj = getattr(mod, cls_name)
            _ok(f"kvpress.{cls_name} importable via sub-module")
        except Exception as sub_exc:
            _fail(f"kvpress.{cls_name} could not be imported: {sub_exc}")

print()
