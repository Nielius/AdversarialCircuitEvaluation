"""What we want to see here: how does the change in Wasserstein distance relate
to the change in quantile metrics? Are they stable?

Basically the same as acdc/nudb/adv_opt/analysis/a2024_w24_01_analyze_quantile_metric_robustness.py,
but it combines data from several directories.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from acdc.nudb.adv_opt.analysis.a2024_w18_analyze_different_prompts import IOIBruteForceResultsCollection

# Let's calculate all the quantile metrics

results = IOIBruteForceResultsCollection.merge(
    *(
        IOIBruteForceResultsCollection.from_dir(path)
        for path in [
            Path("/home/niels/data/advopt/raw/tidy/2024-05-02-bruteforce-ioi-1000samples"),
            Path("/home/niels/data/advopt/raw/tidy/2024-06-12-bruteforce-ioi-100k-sample-pairs"),
            Path("/home/niels/data/advopt/raw/tidy/2024-06-12-bruteforce-ioi-10k-ood-sample-pairs"),
        ]
    )
)

# The loss quantiles for all the brute force results that we've got
all_loss_quantiles = pd.DataFrame(
    [
        pd.Series(result.circuit_loss).describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 0.9999])
        for result in results.results
    ]
)


pairwise_distances = pd.read_csv(
    "/home/niels/data/advopt/processed/2024-06-12-wasserstein-distances-pairwise-for-100k-sample-pairs-and-10k-ood/pairwise_distances.csv"
).set_index("Unnamed: 0")


# %%

(
    all_loss_quantiles.assign(distance=pairwise_distances.iloc[0].values)
    .drop(columns=["count", "std"])
    .sort_values("distance")
    .plot(x="distance", style=".-")
)
plt.show()

# %%

sns.heatmap(pairwise_distances)
plt.show()

# %%
circuitlosses = {i: result.circuit_loss for i, result in enumerate(results.results)}

df = pd.DataFrame.from_dict(
    {i: cl for i, cl in circuitlosses.items()}
)  # <-- ValueError: All arrays must be of the same length
df = pd.DataFrame.from_dict({i: pd.Series(cl) for i, cl in circuitlosses.items()})  # <-- ✅ ; auto-pad with NaN

# %%

df = pd.DataFrame.from_dict({i: pd.Series(cl) for i, cl in circuitlosses.items()})  # <-- ✅ ; auto-pad with NaN

# %%


df.tail()


# %%

df.hist(bins=100)
plt.show()
