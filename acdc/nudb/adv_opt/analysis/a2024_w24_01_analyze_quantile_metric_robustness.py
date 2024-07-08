"""What we want to see here: how does the change in Wasserstein distance relate
to the change in quantile metrics? Are they stable?"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from acdc.nudb.adv_opt.analysis.a2024_w18_analyze_different_prompts import IOIBruteForceResultsCollection

# Let's calculate all the quantile metrics

base_dir = Path("/home/niels/data/advopt/raw/tidy/2024-05-02-bruteforce-ioi-1000samples")
results = IOIBruteForceResultsCollection.from_dir(base_dir)

# The loss quantiles for all the brute force results that we've got
all_loss_quantiles = pd.DataFrame(
    [
        pd.Series(result.circuit_loss).describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 0.9999])
        for result in results.results
    ]
)

pairwise_distances = pd.read_csv(
    "/home/niels/data/advopt/processed/2024-05-14-wasserstein-distances-pairwise/pairwise_distances.csv"
).set_index("Unnamed: 0")

# %%

(
    all_loss_quantiles.assign(distance=pairwise_distances.iloc[0].values)
    .drop(columns=["count", "std"])
    .sort_values("distance")
    .plot(
        x="distance",
    )
)
plt.show()
