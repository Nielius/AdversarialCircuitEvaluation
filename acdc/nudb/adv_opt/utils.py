import os
from pathlib import Path

import torch

num_examples = 5
metric = "kl_div"
device = "cuda" if torch.cuda.is_available() else "cpu"

CIRCUITBENCHMARKS_DATA_DIR = Path(
    os.environ.get("CIRCUITBENCHMARKS_DATA_DIR", "/tmp/circuitbenchmarks_data")
)
CIRCUITBENCHMARKS_DATA_DIR.mkdir(exist_ok=True)
