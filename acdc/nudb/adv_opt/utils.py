import json
import os
from hashlib import md5
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

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


def _deep_map_with_depth(func: Callable[[TIn], Any], x: list | TIn, map_depth: int, current_depth: int) -> Any:
    if current_depth == map_depth:
        return func(x)  # pyright: ignore
    if isinstance(x, Iterable):
        return [_deep_map_with_depth(func, xi, map_depth, current_depth + 1) for xi in x]

    raise ValueError("The input is not iterable and the current depth is not the map depth.")


def deep_map_with_depth(func: Callable[[TIn], Any], x: list | TIn, map_depth: int) -> Any:
    """Quick explanation:

    given a list[list[list[.... [X] ]]]
            |___max_depth___| (max_depth number of lists)
    return list[list[list[.... [func(X)] ]]]

    Use instead of deep_map if you don't want to map over all lists in the input,
    but only up to a certain depth.
    """
    return _deep_map_with_depth(func, x, map_depth, 0)


def deep_map(func: Callable[[TIn], Any], x: list | TIn) -> Any:
    if isinstance(x, list):
        return [deep_map(func, xi) for xi in x]
    else:
        return func(x)
