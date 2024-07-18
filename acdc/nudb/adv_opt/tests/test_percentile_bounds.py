from acdc.nudb.adv_opt.analysis.percentile_bounds import (
    calculate_delta_exactly,
    calculate_epsilon_exactly,
    get_num_samples_exact,
)


def test_calculate_num_examples_exact():
    for p, delta, epsilon in [
        (0.95, 0.95, 0.01),
        (0.95, 0.99, 0.01),
        (0.95, 0.95, 0.04),
        (0.99, 0.95, 0.005),
        (0.99, 0.99, 0.005),
    ]:
        n_exact = get_num_samples_exact(delta, epsilon, p)
        assert calculate_delta_exactly(n_exact, p, epsilon) >= delta
        assert calculate_delta_exactly(n_exact - 1, p, epsilon) < delta

        assert (
            calculate_epsilon_exactly(n_exact, p, delta) < epsilon + 1 / n_exact
        )  # the addition of 1 / n_exact is because calculate_epsilon_exactly returns a larger value of epsilon that strictly necessary
