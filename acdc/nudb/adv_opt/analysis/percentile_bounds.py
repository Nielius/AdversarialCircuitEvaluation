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
    """Calculates the smallest n such that calculate_delta_exactly(n, p, epsilon) >= delta.
    So it assumes you fix epsilon, p, and a lower bound for delta, and then find the smallest n that satisfies the
    conditions."""
    # algorithm: first find an upper bound (by doubling), then do binary search

    n = 1
    while calculate_delta_exactly(n, p, epsilon) < delta:
        n *= 2

    n_lower = n // 2
    n_upper = n

    # Invariant: the solution is somewhere in (n_lower, n_upper]
    while n_upper - n_lower > 1:
        n = (n_lower + n_upper) // 2
        if calculate_delta_exactly(n, p, epsilon) < delta:
            n_lower = n
        else:
            n_upper = n

    return n_upper


def get_num_samples_exact_alternative(delta, epsilon, p) -> int:
    """Uses binary search and the inverse of the binomial cdf to find
    the exact number."""

    def is_large_enough(n: int) -> bool:
        return math.ceil(n * (p + epsilon)) >= binom.ppf(delta, n, p) + 1

    # we need to find n with n >= helper(n)
    # to do that, we use binary search + the smart bound from above

    upper_bound = 1
    while not is_large_enough(upper_bound):
        upper_bound *= 2
    assert is_large_enough(upper_bound)

    lower_bound = 0

    while lower_bound < upper_bound:
        middle = (lower_bound + upper_bound) // 2  # rounded down

        if upper_bound == lower_bound + 1:
            if is_large_enough(lower_bound):
                return lower_bound
            return upper_bound

        if is_large_enough(middle):
            upper_bound = middle
        else:
            lower_bound = middle

    return lower_bound


def calculate_delta_exactly(n, p, epsilon) -> float:
    return binom.cdf(math.ceil((p + epsilon) * n) - 1, n=n, p=p)


def calculate_epsilon_exactly(n, p, delta) -> float:
    """Note: this is basically the inverse of calculate_delta_exactly, but you need to take into account that math.ceil
    does strictly speaking not have an inverse, so we need to make a choice. In this case, we choose to return the largest
    value of epsilon such that math.ceil((p + epsilon) * n) = ppf(delta) + 1. But in fact, any value of epsilon that is
    larger than epsilon - 1/n would work.
    """
    return (binom.ppf(delta, n, p) + 1) / n - p


if __name__ == "__main__":
    for p, delta, epsilon in [
        (0.95, 0.95, 0.01),
        (0.95, 0.99, 0.01),
        (0.95, 0.95, 0.04),
        (0.99, 0.95, 0.005),
        (0.99, 0.99, 0.005),
        (0.999, 0.999, 0.0005),
    ]:
        print(
            f"{p=}, {delta=}, {epsilon=}, simple={get_num_samples_hoeffding(delta, epsilon)}, smart={get_num_sample_with_chernoff(delta, epsilon, p)}, exact={get_num_samples_exact(delta, epsilon, p)}"
        )

    for n in [1e3, 1e6]:
        for p in [0.95, 0.99, 0.999, 0.9999]:
            for epsilon in [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]:
                if epsilon + p >= 1:
                    continue
                print(f"calculate_delta_exactly({n=}, {p=}, {epsilon=})={calculate_delta_exactly(n, p, epsilon)}")

    # calculate the appropriate value of delta for various n, p, epsilon combinations
    delta_calculations = pd.DataFrame({"n": [1e6] * 4, "p": [0.95, 0.99, 0.999, 0.9999], "epsilon": [0.0005] * 4})
    delta_calculations.assign(
        delta=lambda x: x.apply(lambda row: calculate_delta_exactly(row["n"], row["p"], row["epsilon"]), axis=1)
    )
