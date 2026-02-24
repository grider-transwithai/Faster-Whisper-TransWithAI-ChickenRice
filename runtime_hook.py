#!/usr/bin/env python3
"""
Runtime hook for PyInstaller to set environment variables before the application starts.
This resolves OpenMP conflicts when multiple libraries bring their own OpenMP implementations.
"""

import os
import sys
import multiprocessing

# Set KMP_DUPLICATE_LIB_OK to allow multiple OpenMP libraries
# This is needed because different packages (numpy, scipy, ctranslate2, onnxruntime)
# may bring different OpenMP implementations (libiomp5md.dll vs mk2iomp5md.dll)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Suppress transformers advisory warnings
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Configure ONNX Runtime to use half of available CPU cores for better performance
# This prevents oversubscription and resource contention
cpu_count = multiprocessing.cpu_count()
optimal_threads = max(1, cpu_count // 2)

# Set ONNX Runtime environment variables for CPU execution
os.environ["OMP_NUM_THREADS"] = str(optimal_threads)
os.environ["MKL_NUM_THREADS"] = str(optimal_threads)


def _add_windows_dll_directory(path: str) -> None:
    """Best-effort DLL search path setup (Windows only)."""
    if not path or not os.path.isdir(path):
        return
    add_dir = getattr(os, "add_dll_directory", None)
    if add_dir is None:
        return
    try:
        add_dir(path)
        print(f"Runtime hook: Added DLL directory: {path}")
    except Exception as e:
        print(f"Runtime hook: Failed to add DLL directory '{path}': {e}")


# If this build bundles AMD ROCm runtime wheels, ensure their DLL directories are
# on the search path before importing GPU libraries (e.g., ctranslate2).
if sys.platform == "win32" and getattr(sys, "frozen", False):
    bundle_root = os.path.dirname(sys.executable)
    _add_windows_dll_directory(os.path.join(bundle_root, "_rocm_sdk_core", "bin"))
    _add_windows_dll_directory(
        os.path.join(bundle_root, "_rocm_sdk_libraries_custom", "bin")
    )

print(f"Runtime hook: Set KMP_DUPLICATE_LIB_OK=TRUE to resolve OpenMP conflicts")
print(
    f"Runtime hook: Set TRANSFORMERS_NO_ADVISORY_WARNINGS=1 to suppress advisory warnings"
)
print(
    f"Runtime hook: Configured ONNX Runtime to use {optimal_threads} threads (half of {cpu_count} available CPUs)"
)
