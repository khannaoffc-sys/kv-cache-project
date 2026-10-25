"""make_line_mem.py - prepare the key ROM files for the on-board demo.

Why: kv_demo_top.v stores one whole token (64 bytes = 512 bits) in each ROM word.
$readmemh treats every space-separated number as a SEPARATE word, so the original
layerN_keys_int8.mem (64 bytes per line, separated by spaces) would load as 1 byte
per word. This script joins each line into ONE 128-hex-digit number (no spaces) and
writes layerN_keys_line.mem. It then checks every token's sum of squares against
layerN_scores_int32.mem, so you know the new files are correct.

Run from the project root:   python src/make_line_mem.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "results" / "fpga"
HEAD_DIM = 64
SEQ_LEN = 512


def data_lines(path):
    """Non-empty lines with // comments removed."""
    out = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split("//")[0].strip()
        if line:
            out.append(line)
    return out


all_ok = True
for layer in (0, 10):
    keys_in = D / f"layer{layer}_keys_int8.mem"
    scores_in = D / f"layer{layer}_scores_int32.mem"
    keys_out = D / f"layer{layer}_keys_line.mem"
    for p in (keys_in, scores_in):
        if not p.exists():
            sys.exit(f"MISSING FILE: {p}")

    rows = data_lines(keys_in)
    expected = [int(x, 16) for x in data_lines(scores_in)]
    if len(rows) != SEQ_LEN or len(expected) != SEQ_LEN:
        sys.exit(f"layer {layer}: expected {SEQ_LEN} key rows and scores, "
                 f"got {len(rows)} and {len(expected)}")

    out_lines = []
    bad = 0
    for t, row in enumerate(rows):
        toks = row.split()
        if len(toks) != HEAD_DIM or any(len(x) != 2 for x in toks):
            sys.exit(f"layer {layer} token {t}: expected {HEAD_DIM} two-digit hex bytes")
        vals = [int(x, 16) - 256 if int(x, 16) >= 128 else int(x, 16) for x in toks]
        if sum(v * v for v in vals) != expected[t]:
            bad += 1
        out_lines.append("".join(toks).lower())

    keys_out.write_text("\n".join(out_lines) + "\n", encoding="ascii")
    status = "PASS" if bad == 0 else f"FAIL ({bad} tokens differ)"
    all_ok &= (bad == 0)
    print(f"layer {layer:>2}: wrote {keys_out.name} "
          f"({len(out_lines)} lines x {HEAD_DIM * 2} hex digits); "
          f"scores recomputed from the new file vs expected: {status}")

print("ALL PASS" if all_ok else "PROBLEM: do not use these files")
sys.exit(0 if all_ok else 1)
