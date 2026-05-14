"""Compare Rust forward-pass dumps against Python reference dumps.

Reads two directories of ``.safetensors`` files (each containing a single
``"data"`` tensor, as produced by ``dump_reference.py``) and reports the
maximum absolute and relative error per tensor, plus the first divergence.

Usage examples (run from ``reference/``)::

    uv run compare.py Hello__abc12345 ../rust_dumps/Hello__abc12345
    uv run compare.py Hello__abc12345 ../rust_dumps/Hello__abc12345 --threshold 1e-3
    uv run compare.py --list                       # list available reference dumps
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

DEFAULT_REF_ROOT = "outputs"


def relative_error(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Returns (max_abs_diff, max_relative_diff)."""
    diff = np.abs(a - b)
    max_abs = float(diff.max())
    scale = max(float(np.abs(a).max()), float(np.abs(b).max()), 1e-12)
    return max_abs, max_abs / scale


def load_data(path: Path) -> np.ndarray:
    return load_file(str(path))["data"].to(torch.float32).numpy()


def list_dumps(ref_root: Path) -> None:
    if not ref_root.exists():
        print(f"No reference root: {ref_root}")
        return
    subs = sorted(p.name for p in ref_root.iterdir() if p.is_dir())
    if not subs:
        print(f"No prompt dumps under {ref_root}")
        return
    for s in subs:
        meta = ref_root / s / "metadata.json"
        if meta.exists():
            import json

            m = json.loads(meta.read_text(encoding="utf-8"))
            print(f"  {s:40s} prompt={m.get('prompt')!r}")
        else:
            print(f"  {s}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Rust kernel dumps against Python reference dumps."
    )
    parser.add_argument(
        "prompt_slug",
        nargs="?",
        help="Prompt subdirectory under --ref-root (e.g. 'Hello__abc12345').",
    )
    parser.add_argument(
        "rust_dir",
        nargs="?",
        help="Directory containing the Rust-side .safetensors dumps for the same prompt.",
    )
    parser.add_argument(
        "--ref-root",
        default=DEFAULT_REF_ROOT,
        help="Root of reference dumps (default: %(default)s).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-3,
        help="Pass when max relative error < threshold (default: %(default)s).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available reference prompt dumps and exit.",
    )
    args = parser.parse_args()

    ref_root = Path(args.ref_root)
    if args.list:
        list_dumps(ref_root)
        return 0

    if not args.prompt_slug or not args.rust_dir:
        parser.error("prompt_slug and rust_dir are required (unless --list)")

    ref_dir = ref_root / args.prompt_slug
    rust_dir = Path(args.rust_dir)
    if not ref_dir.exists():
        raise SystemExit(f"Reference dir not found: {ref_dir}")
    if not rust_dir.exists():
        raise SystemExit(f"Rust dir not found: {rust_dir}")

    ref_files = sorted(p.name for p in ref_dir.glob("*.safetensors"))
    rust_files = {p.name for p in rust_dir.glob("*.safetensors")}
    common = [f for f in ref_files if f in rust_files]
    only_ref = [f for f in ref_files if f not in rust_files]
    only_rust = sorted(rust_files - set(ref_files))

    print(
        f"Comparing {len(common)} tensors in {args.prompt_slug} "
        f"(threshold={args.threshold:.1e})\n"
    )
    print(
        f"{'tensor':<40s} {'shape':<25s} {'max_abs':>12s} {'max_rel':>12s}  status"
    )
    print("-" * 100)

    fail_count = 0
    first_fail: str | None = None
    for fname in common:
        a = load_data(ref_dir / fname)
        b = load_data(rust_dir / fname)
        if a.shape != b.shape:
            # Same element count but different shape (e.g. [seq, heads, head_dim]
            # vs flat [seq, heads*head_dim]) — flatten both and compare values.
            # The comparison is over the same underlying memory layout
            # (row-major), so this is numerically meaningful.
            if a.size == b.size:
                a_flat = a.reshape(-1)
                b_flat = b.reshape(-1)
                max_abs, max_rel = relative_error(a_flat, b_flat)
                status = "OK*" if max_rel < args.threshold else "FAIL*"
                shape_str = f"{a.shape}~{b.shape}"
                print(
                    f"{fname:<40s} {shape_str:<25s} {max_abs:>12.3e} {max_rel:>12.3e}  {status}"
                )
                if status.startswith("FAIL"):
                    fail_count += 1
                    if first_fail is None:
                        first_fail = fname
                continue
            print(
                f"{fname:<40s} SHAPE MISMATCH: ref {a.shape} vs rust {b.shape}"
            )
            fail_count += 1
            if first_fail is None:
                first_fail = fname
            continue
        max_abs, max_rel = relative_error(a, b)
        status = "OK" if max_rel < args.threshold else "FAIL"
        print(
            f"{fname:<40s} {str(a.shape):<25s} {max_abs:>12.3e} {max_rel:>12.3e}  {status}"
        )
        if status == "FAIL":
            fail_count += 1
            if first_fail is None:
                first_fail = fname

    if only_ref:
        head = ", ".join(only_ref[:5])
        more = "" if len(only_ref) <= 5 else f" (+{len(only_ref) - 5} more)"
        print(f"\nReference-only files (Rust did not dump): {head}{more}")
    if only_rust:
        head = ", ".join(only_rust[:5])
        more = "" if len(only_rust) <= 5 else f" (+{len(only_rust) - 5} more)"
        print(f"Rust-only files (no reference): {head}{more}")

    print(
        f"\nSummary: {len(common) - fail_count}/{len(common)} OK"
        + (f", first FAIL = {first_fail}" if first_fail else "")
    )
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
