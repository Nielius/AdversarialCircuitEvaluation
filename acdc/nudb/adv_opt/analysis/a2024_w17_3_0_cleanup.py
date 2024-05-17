import shutil
from pathlib import Path

base_path = Path("/home/niels/data/advopt/tidy/2024-04-13-01-halving/data")

success_path = base_path / "success"
failure_path = base_path / "failure"

success_path.mkdir(exist_ok=True)
failure_path.mkdir(exist_ok=True)


for p in base_path.glob("2024-*"):
    if (p / "artifacts").exists():
        shutil.move(p, success_path)
    else:
        shutil.move(p, failure_path)
