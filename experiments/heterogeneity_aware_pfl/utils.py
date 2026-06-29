"""Utility functions: seeding, timing, GPU memory, communication cost, logging."""

import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


class _TeeStream:
    """Write to both a file (with timestamps) and the original stream."""

    def __init__(self, stream, filepath: Path):
        self._stream = stream
        self._file = open(filepath, "a", buffering=1)
        self._buf = ""  # buffer incomplete lines for timestamping

    def write(self, data: str) -> int:
        self._stream.write(data)
        # File: buffer until newline, then write with timestamp.
        # \r is treated as a line end (tqdm overwrite style) but discarded —
        # this suppresses per-step tqdm spam in the log file.
        self._buf += data
        while True:
            nl = self._buf.find("\n")
            cr = self._buf.find("\r")
            if nl == -1 and cr == -1:
                break
            if nl != -1 and (cr == -1 or nl < cr):
                # newline comes first — write the line with timestamp
                line = self._buf[:nl]
                self._buf = self._buf[nl + 1:]
                ts = datetime.now().strftime("%H:%M:%S")
                self._file.write(f"[{ts}] {line}\n")
            else:
                # carriage return comes first — discard (tqdm overwrite, not useful in file)
                self._buf = self._buf[cr + 1:]
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        if self._buf:
            ts = datetime.now().strftime("%H:%M:%S")
            self._file.write(f"[{ts}] {self._buf}")
            self._buf = ""
        self._file.flush()

    def fileno(self):
        return self._stream.fileno()

    def __getattr__(self, attr):
        return getattr(self._stream, attr)


def setup_file_logging(log_path: Path) -> None:
    """Redirect stdout and stderr to both console and a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = _TeeStream(sys.stdout, log_path)
    sys.stderr = _TeeStream(sys.stderr, log_path)
    print(f"Logging to: {log_path}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_peak_gpu_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def reset_gpu_memory_tracking() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def measure_serialization_bytes(params: List[np.ndarray]) -> int:
    return sum(p.nbytes for p in params)


class RoundTimer:
    """Context manager for timing phases."""

    def __init__(self, label: str = ""):
        self.label = label
        self._start = 0.0
        self._elapsed = 0.0

    def __enter__(self) -> "RoundTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self._elapsed = time.perf_counter() - self._start

    @property
    def elapsed_seconds(self) -> float:
        return self._elapsed


class ExperimentLogger:
    """Structured logging for experiment metrics to JSONL files."""

    def __init__(self, save_dir: str, experiment_name: str):
        self.base_dir = Path(save_dir) / experiment_name
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.base_dir / "metrics.jsonl"
        self.validation_path = self.base_dir / "validation.jsonl"
        self.test_path = self.base_dir / "test_results.json"

    def log_round(self, round_num: int, metrics: Dict) -> None:
        entry = {"round": round_num, **metrics}
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def reset_round_logs(self) -> None:
        """Truncate round/validation logs (used on a fresh run with no checkpoint)."""
        for p in (self.metrics_path, self.validation_path):
            if p.exists():
                p.unlink()

    def log_validation(self, round_num: int, client_id: int, scores: Dict[str, float]) -> None:
        entry = {"round": round_num, "client_id": client_id, **scores}
        with open(self.validation_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_test(self, client_id: int, scores: Dict[str, float]) -> None:
        entry = {"client_id": client_id, **scores}
        # Append to a JSONL for individual client results
        test_individual = self.base_dir / "test_individual.jsonl"
        with open(test_individual, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def save_summary(self, summary: Dict) -> None:
        with open(self.test_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {self.test_path}")
