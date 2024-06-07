# pyright: noqa
"""Plot heatmaps of the coefficients obtained from the convex optimization experiments."""
from pathlib import Path

import jaxtyping
import matplotlib.pyplot as plt
import sklearn.decomposition
import torch

from acdc.nudb.adv_opt.analysis.analyzer import AdvOptAnalyzer
from acdc.nudb.adv_opt.analysis.output_parser import AdvOptHydraOutputDir

base_path = Path("/home/niels/proj/mats/data/outputs/2024-04-12-coefficient-renormalization") / "success"


fn = next(base_path.glob("2024*"))

hydra_output = AdvOptHydraOutputDir(fn)
artifacts = hydra_output.artifacts

assert artifacts.coefficients_final is not None


analyzer = AdvOptAnalyzer.from_dir(AdvOptHydraOutputDir(fn))
tokenizer = analyzer.get_tokenizer()

topk, tokp_patch = analyzer.topk_coefficients()
topk_decoded, tokp_patch_decoded = analyzer.topk_inputs_decoded(tokenizer)

print(topk_decoded)
print(tokp_patch_decoded)


# I want to see how different the final coefficients are. So maybe just create a heatmap of the final coefficients?
# Maybe I can just stack the coefficients and then plot them as a heatmap?
# I could also plot the difference between the coefficients, but I think that would be harder to interpret.

# What would I want to do?
# - compare these results to the brute force results
# - combine patch and input to see which input is worst
# - include the summary statistics (entropy, highest loss, ...)

# Can we draw a path from the coefficients?

# NOTE: for this to make sense, you need to center the coefficients! otherwise you can't compare between experiments


all_final_coefficients = [AdvOptHydraOutputDir(fn).artifacts.coefficients_final for fn in base_path.glob("2024*")]


def plot_coeffs(coefficients: list[torch.Tensor]):
    all_coefficients: jaxtyping.Float[torch.Tensor, "experiment input"] = torch.stack(coefficients)
    all_coefficients_centered = all_coefficients - all_coefficients.mean(dim=1, keepdim=True)
    plt.imshow(all_coefficients_centered.numpy(), cmap="hot", interpolation="none", aspect="auto")
    plt.show()


plot_coeffs([AdvOptHydraOutputDir(fn).artifacts.coefficients_final for fn in base_path.glob("2024*")])
plot_coeffs([AdvOptHydraOutputDir(fn).artifacts.coefficients_final_patch for fn in base_path.glob("2024*")])


all_coefficients: jaxtyping.Float[torch.Tensor, "experiment input"] = torch.stack(all_final_coefficients)
all_coefficients_centered = all_coefficients - all_coefficients.mean(dim=1, keepdim=True)
plt.imshow(all_coefficients_centered.numpy(), cmap="hot", interpolation="none", aspect="auto")
plt.show()

all_initial_coefficients = [AdvOptHydraOutputDir(fn).artifacts.coefficients_init for fn in base_path.glob("2024*")]
all_coefficients: jaxtyping.Float[torch.Tensor, "experiment input"] = torch.stack(all_initial_coefficients)
all_coefficients_centered = all_coefficients - all_coefficients.mean(dim=1, keepdim=True)
plt.imshow(all_coefficients_centered.numpy(), cmap="hot", interpolation="none", aspect="auto")
plt.show()


decomp = sklearn.decomposition.PCA(n_components=3).fit_transform(all_coefficients.numpy())

plt.scatter(decomp[:, 0], decomp[:, 1], c=range(len(decomp)))


# base_path = Path("/home/niels/proj/acdc/outputs/2024-04-16-141417-docstring_500")  # every 10 epochs
base_path = Path("/home/niels/proj/acdc/outputs/2024-04-16-144219-docstring_500")  # every single epoch


def epoch_from_path(fn: Path) -> int:
    return int(fn.stem.split("_")[-1])


coefficients_tuples = sorted(
    ((epoch_from_path(fn), torch.load(fn)) for fn in base_path.glob("coefficients_patch_[0123456789]*.pt")),
    key=lambda t: t[0],
)

epochs, coefficients_list = zip(*coefficients_tuples)

coefficients = torch.stack(list(coefficients_list)).detach().numpy()


decomp = sklearn.decomposition.PCA(n_components=3).fit_transform(coefficients)


plt.scatter(decomp[:, 0], decomp[:, 1], c=epochs)
plt.show()

plt.figure(figsize=(30, 18))
plt.plot(decomp[:, 0], decomp[:, 1])
[plt.text(i, j, str(epoch)) for ((i, j, k), epoch) in zip(decomp, epochs)]
plt.show()


plt.plot(decomp[:, 0], decomp[:, 2])
[plt.text(i, k, str(epoch)) for ((i, j, k), epoch) in zip(decomp, epochs)]
plt.show()

plt.plot(decomp[:, 1], decomp[:, 2])
[plt.text(j, k, str(epoch)) for ((i, j, k), epoch) in zip(decomp, epochs)]
plt.show()
