"""
Functions to calculate how many samples you need to estimate upper bounds of quantiles with high probability.

The setup is as follows:

Given n iid samples of a random variable X, the math.ceil((p + epsilon) * n)-th smallest sample value is
an upper bound for the p-th percentile of X with probability at least (or precisely, for the formulas labelled 'exact')
delta.
"""

import math

import pandas as pd
from scipy.stats import binom


def get_num_samples_hoeffding(delta, epsilon):
    """Use Hoeffding's inequality to calculate a lower bound for the number of samples needed for delta, epsilon."""
    return math.ceil(-1 * math.log(1 - delta) / (2 * epsilon**2))


def relative_entropy_of_two_binomial_distributions(p: float, q: float) -> float:
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def get_num_sample_with_chernoff(delta, epsilon, p) -> int:
    """Use Chernoff's bound."""
    return math.ceil(-1 * math.log(1 - delta) / (relative_entropy_of_two_binomial_distributions(p + epsilon, p)))


def get_num_samples_exact(delta, epsilon, p) -> int:
    # ppf (percent-point function) is the inverse of the cdf, i.e. it sends a probability
    # to the value x such that Pr(X <= x) is equal to the probability.
    # It is also called the quantile function.
    return math.ceil((binom.ppf(delta) + 1) / (p + epsilon))


def calculate_delta_exactly(n, p, epsilon) -> float:
    return binom.cdf(math.ceil((p + epsilon) * n) - 1, n, p)


def calculate_epsilon_exactly(n, p, delta) -> float:
    return (binom.ppf(delta, n, p) + 1) / n - p


if __name__ == "__main__":
    for p, delta, epsilon in [
        (0.95, 0.95, 0.01),
        (0.95, 0.99, 0.01),
        (0.95, 0.95, 0.04),
        (0.99, 0.95, 0.005),
        (0.99, 0.99, 0.005),
    ]:
        print(
            f"{p=}, {delta=}, {epsilon=}, simple={get_num_samples_hoeffding(delta, epsilon)}, smart={get_num_sample_with_chernoff(delta, epsilon, p)}"
        )

    # calculate the appropriate value of delta for various n, p, epsilon combinations
    delta_calculations = pd.DataFrame({"n": [1e6] * 4, "p": [0.95, 0.99, 0.999, 0.9999], "epsilon": [0.0005] * 4})
    delta_calculations.assign(
        delta=lambda x: x.apply(lambda row: calculate_delta_exactly(row["n"], row["p"], row["epsilon"]), axis=1)
    )
