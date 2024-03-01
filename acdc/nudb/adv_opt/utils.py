import json
import os
from hashlib import md5
from pathlib import Path
from typing import Any, Callable, TypeVar

import torch
from joblib import Memory

num_examples = 5
metric = "kl_div"
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

CIRCUITBENCHMARKS_DATA_DIR = Path(os.environ.get("CIRCUITBENCHMARKS_DATA_DIR", "/tmp/circuitbenchmarks_data"))
CIRCUITBENCHMARKS_DATA_DIR.mkdir(exist_ok=True)


def tensor_fingerprint(tensor: torch.Tensor) -> str:
    """For use in debugging"""
    # numpy tobytes, then md5, then hexdigest
    return md5(json.dumps((tensor * 1000).int().tolist()).encode("utf8")).hexdigest()[0:8]


joblib_memory = Memory(CIRCUITBENCHMARKS_DATA_DIR / "joblib", verbose=0)
TIn = TypeVar("TIn")


def deep_map(func: Callable[[TIn], Any], x: list | TIn) -> Any:
    if isinstance(x, list):
        return [deep_map(func, xi) for xi in x]
    else:
        return func(x)
