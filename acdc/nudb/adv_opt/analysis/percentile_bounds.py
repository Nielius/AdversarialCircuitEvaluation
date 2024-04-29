import math

from scipy.stats import binom

delta = 0.95
epsilon = 0.01
p = 0.95  # the percentile we're looking for


# Why does this not depend on the percentile? Quite strange...
# Yeah it's just because that is not part of the bound: it only depends on the divergence from what you expect
def get_num_samples(delta, epsilon):
    return math.ceil(-1 * math.log(1 - delta) / (2 * epsilon**2))


get_num_samples(delta, epsilon)


def relative_entropy_of_two_binomial_distributions(p: float, q: float) -> float:
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def get_num_sample_with_rel_entropy(delta, epsilon, p) -> int:
    return math.ceil(-1 * math.log(1 - delta) / (relative_entropy_of_two_binomial_distributions(p + epsilon, p)))


def get_num_samples_exact(delta, epsilon, p) -> int:
    return math.ceil((binom.ppf(delta) + 1) / (p + epsilon))
    return math.ceil(-1 * math.log(1 - delta) / (relative_entropy_of_two_binomial_distributions(p + epsilon, p)))

    binom.cdf(9, 10, 0.5)
    binom.ppf(0.99, 10, 0.5)


get_num_sample_with_rel_entropy(delta, epsilon, p)

for p, delta, epsilon in [
    (0.95, 0.95, 0.01),
    (0.95, 0.99, 0.01),
    (0.95, 0.95, 0.04),
    (0.99, 0.95, 0.005),
    (0.99, 0.99, 0.005),
]:
    print(
        f"{p=}, {delta=}, {epsilon=}, simple={get_num_samples(delta, epsilon)}, smart={get_num_sample_with_rel_entropy(delta, epsilon, p)}"
    )
