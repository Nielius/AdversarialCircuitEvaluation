import json
from pathlib import Path

from acdc.nudb.adv_opt.analysis.analyzer import AdvOptAnalyzer
from acdc.nudb.adv_opt.analysis.output_parser import AdvOptHydraOutputDir

base_path = Path("/home/niels/data/advopt/tidy/2024-04-13-01-halving/data/success")
# base_path = Path(sys.argv[1])

print("Analysing directory at", base_path)


# topk_inputs, topk_inputs_patch = analyser.topk_inputs()
# tokenizer = analyser.get_tokenizer()
# topk_inputs_decoded, topk_inputs_patch_decoded = analyser.topk_inputs_decoded(tokenizer)

max_losses: dict[str, float] = {}

max_losses_path = base_path / "max_losses.json"
for i, experiment_path in enumerate(base_path.glob("2024-*")):
    if i % 10 == 0:
        max_losses_path.write_text(json.dumps(max_losses, indent=4))

    analyser = AdvOptAnalyzer.from_dir(AdvOptHydraOutputDir(experiment_path))
    experiment_data = analyser.get_experiment_data()
    losses = analyser.loss_for_topk_inputs(experiment_data)

    max_losses[experiment_path.name] = losses.max().item()


max_losses_path.write_text(json.dumps(max_losses, indent=4))


max_losses = json.loads(max_losses_path.read_text())


def extract_task(key: str) -> str:
    return key.split("-")[4]


max_losses_split = {
    "ioi": {k: v for k, v in max_losses.items() if extract_task(k) == "ioi"},
    "greaterthan": {k: v for k, v in max_losses.items() if extract_task(k) == "greaterthan"},
    "docstring": {k: v for k, v in max_losses.items() if extract_task(k) == "docstring"},
}

max_losses_allmax = {k: max(v.values()) for k, v in max_losses_split.items()}


# experiment_path = next(base_path.glob("2024-*"))

# losses.max()

# print(losses)


# test outputs
# test loss for topk inputs
# run on all/most experiments, keeping track of the loss
