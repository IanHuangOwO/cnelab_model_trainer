from __future__ import annotations

import csv
import os
from typing import Optional

import numpy as np


def read_eeg(path: str) -> np.ndarray:
    """Dispatch EEG file reading based on extension and ensure (C, T) layout.

    Supports:
      - .csv: comma-separated values, rows are channels or timepoints
      - .npy: NumPy binary arrays

    Returns a float32 numpy array of shape (C, T) with C < T.
    Raises ValueError if shape cannot be interpreted as (C, T).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        arr = _read_csv(path)
    elif ext == ".npy":
        arr = _read_npy(path)
    else:
        raise ValueError(f"Unsupported EEG file extension: {ext} for path: {path}")
    return _ensure_channels_time(arr, path)


def _read_csv(path: str) -> np.ndarray:
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        data = [row for row in reader]
    return np.asarray(data, dtype=np.float32)


def _read_npy(path: str) -> np.ndarray:
    arr = np.load(path)
    # Squeeze singular dims; accept 1D/2D; reject >2D after squeeze
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    else:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array for EEG data, got shape {arr.shape} from {path}")
    return arr.astype(np.float32, copy=False)


def _ensure_channels_time(arr: np.ndarray, src: str | None = None) -> np.ndarray:
    """Ensure array is (C, T) with channels first and C < T.

    If the first dimension is greater than the second, transpose.
    After possible transpose, validate that C < T; raise on ambiguity or invalid.
    """
    if arr.ndim != 2:
        raise ValueError(f"EEG array must be 2D, got shape {arr.shape} from {src or 'array'}")
    c, t = arr.shape
    if c > t:
        arr = arr.T
        c, t = arr.shape
    if c >= t:
        raise ValueError(
            f"Cannot determine (C, T) with C < T for {src or 'array'}; got shape {arr.shape}"
        )
    return arr

def gen_eeg(
    C: int = 32,
    T: int = 1024,
    *,
    sample_rate: float = 256.0,
    mode: str = "mixed",  # one of: "sine", "noise", "mixed"
    noise_std: float = 0.1,
    num_components: int = 3,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a synthetic EEG-like signal array of shape (C, T).

    - "sine": sum of a few random sinusoidal components per channel
    - "noise": Gaussian noise only
    - "mixed": sinusoid components + Gaussian noise

    Returns float32 with C < T guaranteed if inputs satisfy that.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=np.float32) / float(sample_rate)

    x = np.zeros((C, T), dtype=np.float32)

    if mode not in {"sine", "noise", "mixed"}:
        raise ValueError(f"Unsupported mode: {mode}")

    if mode in {"sine", "mixed"}:
        for c in range(C):
            # Randomly pick component freqs (1–40 Hz), amplitudes, and phases
            freqs = rng.uniform(1.0, 40.0, size=(num_components,)).astype(np.float32)
            amps = rng.uniform(0.1, 1.0, size=(num_components,)).astype(np.float32)
            phases = rng.uniform(0.0, 2.0 * np.pi, size=(num_components,)).astype(np.float32)
            # Sum sinusoids
            s = np.zeros_like(t)
            for f, a, p in zip(freqs, amps, phases):
                s += a * np.sin(2.0 * np.pi * f * t + p)
            x[c] += s.astype(np.float32)

    if mode in {"noise", "mixed"}:
        x += rng.normal(loc=0.0, scale=noise_std, size=(C, T)).astype(np.float32)

    return x

__all__ = [
    "read_eeg",
    "gen_eeg",
]
