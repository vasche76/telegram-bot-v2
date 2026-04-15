#!/usr/bin/env python3
"""
install_ml_deps.py — One-shot helper to install ML dependencies for the fish CV system.

Detects hardware (Apple Silicon MPS / NVIDIA CUDA / CPU-only) and installs the
correct torch build, then installs ultralytics and Pillow.

Usage:
    python3 scripts/install_ml_deps.py [--dry-run]

Flags:
    --dry-run   Print the pip commands that would be run without executing them.

Exit codes:
    0 — All dependencies installed and verified.
    1 — Installation or verification failed.
"""

import argparse
import importlib
import platform
import subprocess
import sys
from typing import Optional


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _ok(msg: str) -> None:
    print(f"[ OK ] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)


def _banner(msg: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}")


def _run_pip(args: list[str], dry_run: bool) -> bool:
    """Run pip with the given args. Returns True on success."""
    cmd = [sys.executable, "-m", "pip"] + args
    print(f"\n  $ {' '.join(cmd)}")
    if dry_run:
        print("  (dry-run — skipping execution)")
        return True
    result = subprocess.run(cmd)
    return result.returncode == 0


# ─── Checks ──────────────────────────────────────────────────────────────────

def check_python_version() -> bool:
    """
    Require Python 3.9–3.12.

    PyTorch does not publish wheels for Python 3.13+ on macOS x86_64 (Intel).
    If you only have Python 3.13, install 3.12 via pyenv:
        brew install pyenv
        pyenv install 3.12.10
        ~/.pyenv/versions/3.12.10/bin/python3 scripts/install_ml_deps.py
    """
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 9):
        _fail(f"Python 3.9+ required. You have {major}.{minor}.")
        return False

    if (major, minor) >= (3, 13):
        _warn(
            f"Python {major}.{minor} detected. "
            "PyTorch does not publish wheels for Python 3.13+ on macOS x86_64 (Intel Mac)."
        )
        _warn(
            "Install Python 3.12 via pyenv and use it for training:\n"
            "  brew install pyenv\n"
            "  pyenv install 3.12.10\n"
            "  ~/.pyenv/versions/3.12.10/bin/python3 scripts/install_ml_deps.py"
        )
        # Don't abort — the user may be on Apple Silicon or Linux where 3.13 works
        _warn("Proceeding anyway — installation may fail.")

    _ok(f"Python {major}.{minor}")
    return True


def detect_hardware() -> str:
    """
    Detect available accelerator hardware.
    Returns one of: "mps", "cuda", "cpu".
    """
    machine = platform.machine().lower()
    system = platform.system().lower()

    # Apple Silicon
    if system == "darwin" and machine in ("arm64", "aarch64"):
        _info("Detected Apple Silicon (arm64) — will use MPS backend if available")
        return "mps"

    # NVIDIA CUDA — check for nvidia-smi or nvcc
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip().splitlines()[0]
            _info(f"Detected NVIDIA GPU: {gpu_name} — will use CUDA backend")
            return "cuda"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    _info("No GPU detected — will use CPU-only PyTorch")
    return "cpu"


def get_torch_install_args(hardware: str) -> list[str]:
    """
    Return pip install args for the correct torch build.
    Torch 2.x stable wheels at pytorch.org.
    """
    base = ["install", "torch>=2.0.0", "torchvision>=0.15.0"]
    if hardware == "cuda":
        # CUDA 12.1 wheel (covers most modern GPUs)
        return base + ["--index-url", "https://download.pytorch.org/whl/cu121"]
    elif hardware == "mps":
        # Standard macOS wheel includes MPS support; no special index needed
        return base
    else:
        # CPU-only — smaller download
        return base + ["--index-url", "https://download.pytorch.org/whl/cpu"]


# ─── Verification ─────────────────────────────────────────────────────────────

def _try_import(module_name: str) -> Optional[object]:
    """Try to import a module; return the module or None."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def verify_imports() -> bool:
    """Verify all ML packages import correctly. Returns True if all OK."""
    all_ok = True

    # torch
    torch = _try_import("torch")
    if torch is None:
        _fail("torch is not importable after installation")
        all_ok = False
    else:
        _ok(f"torch {torch.__version__}")  # type: ignore[union-attr]
        # Check MPS / CUDA availability
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[union-attr]
            _ok("torch.backends.mps.is_available() = True  (Apple Silicon GPU ready)")
        elif torch.cuda.is_available():  # type: ignore[union-attr]
            _ok(f"torch.cuda.is_available() = True  ({torch.cuda.get_device_name(0)})")  # type: ignore[union-attr]
        else:
            _info("No GPU backend available — training will use CPU")

    # torchvision
    tv = _try_import("torchvision")
    if tv is None:
        _fail("torchvision is not importable after installation")
        all_ok = False
    else:
        _ok(f"torchvision {tv.__version__}")  # type: ignore[union-attr]

    # ultralytics
    ul = _try_import("ultralytics")
    if ul is None:
        _fail("ultralytics is not importable after installation")
        all_ok = False
    else:
        _ok(f"ultralytics {ul.__version__}")  # type: ignore[union-attr]

    # Pillow
    pil = _try_import("PIL")
    if pil is None:
        _fail("Pillow (PIL) is not importable after installation")
        all_ok = False
    else:
        import PIL  # noqa: PLC0415
        _ok(f"Pillow {PIL.__version__}")

    return all_ok


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install ML dependencies for the fish CV pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the pip commands that would run without executing them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dry_run: bool = args.dry_run

    _banner("Fish CV ML Dependency Installer")

    if dry_run:
        _info("DRY-RUN mode — no packages will actually be installed")

    # ── Step 1: Python version check ─────────────────────────────────────────
    _banner("Step 1: Python version check")
    if not check_python_version():
        return 1

    # ── Step 2: Hardware detection ───────────────────────────────────────────
    _banner("Step 2: Hardware detection")
    hardware = detect_hardware()

    # ── Step 3: Install PyTorch ──────────────────────────────────────────────
    _banner("Step 3: Installing PyTorch + torchvision")
    torch_args = get_torch_install_args(hardware)
    if not _run_pip(torch_args, dry_run):
        _fail("PyTorch installation failed. Check the error above.")
        return 1

    # ── Step 4: Install ultralytics ──────────────────────────────────────────
    _banner("Step 4: Installing ultralytics (YOLOv8)")
    if not _run_pip(["install", "ultralytics>=8.0.0"], dry_run):
        _fail("ultralytics installation failed.")
        return 1

    # ── Step 5: Install Pillow ───────────────────────────────────────────────
    _banner("Step 5: Installing Pillow")
    if not _run_pip(["install", "Pillow>=10.0.0"], dry_run):
        _fail("Pillow installation failed.")
        return 1

    # ── Step 6: Verify imports ───────────────────────────────────────────────
    if not dry_run:
        _banner("Step 6: Verifying imports")
        if not verify_imports():
            _fail("One or more packages failed to import. See above for details.")
            return 1

    # ── Summary ───────────────────────────────────────────────────────────────
    _banner("Summary")
    if dry_run:
        _ok("Dry-run complete — no changes made.")
    else:
        _ok("All ML dependencies installed and verified.")
        print()
        _info("Next steps:")
        _info("  1. Add fish images to data/fish_dataset/")
        _info("  2. python3 scripts/validate_dataset.py   # check dataset is ready")
        _info("  3. python3 scripts/train_stage_a.py      # train YOLO detector")
        _info("  4. python3 scripts/train_stage_b.py      # train species classifier")
        _info("  5. Set FISH_DETECTOR_BACKEND=local and FISH_CLASSIFIER_BACKEND=local in .env")

    return 0


if __name__ == "__main__":
    sys.exit(main())
