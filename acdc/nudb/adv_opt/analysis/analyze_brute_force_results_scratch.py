# ruff: noqa

# TODO Niels: move all the below to a new file; separate getting the results from the analysis, to get
# better feedback loops with the plotting and analysis


def plot():
    # plot histogram of output
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)  # Create a figure containing a single axes.
    ((ax_all, ax_1), (ax_2, ax_3)) = axes
    range = (
        0,
        max(
            # metrics_with_full_model.max().item(),  # can safely exclude this, as it's always supposed to be 0
            metrics_with_canonical_circuit.max().item(),
            metrics_with_random_circuit.max().item(),
            metrics_with_corrupted_canonical_circuit.max().item(),
        ),
    )
    if settings.include_full_circuit:
        ax_all.stairs(*torch.histogram(metrics_with_full_model, bins=100, range=range), label="full model")
    ax_all.stairs(*torch.histogram(metrics_with_canonical_circuit, bins=100, range=range), label="canonical circuit")
    ax_all.stairs(*torch.histogram(metrics_with_random_circuit, bins=100, range=range), label="random circuit")
    ax_all.stairs(
        *torch.histogram(metrics_with_corrupted_canonical_circuit, bins=100, range=range),
        label="corrupted canonical circuit",
    )
    fig.suptitle(
        f"KL divergence between output of the full model and output of a circuit, for {experiment_name}, histogram"
    )
    ax_1.stairs(*torch.histogram(metrics_with_canonical_circuit, bins=100, range=range), label="canonical circuit")
    ax_2.stairs(*torch.histogram(metrics_with_random_circuit, bins=100, range=range), label="random circuit")
    ax_3.stairs(
        *torch.histogram(metrics_with_corrupted_canonical_circuit, bins=100, range=range),
        label="corrupted canonical circuit",
    )
    for ax in itertools.chain(*axes):
        ax.set_xlabel("KL divergence")
        ax.set_ylabel("Frequency")
        ax.legend()

    plot_dir = artifact_dir / "plots"
    plot_dir.mkdir(exist_ok=True, parents=True)
    figure_path = plot_dir / f"{experiment_name}_histogram_{datetime.datetime.now().isoformat()}.png"
    fig.savefig(figure_path)
    logger.info("Saved histogram to %s", figure_path)


plot()


def decode(input: Integer[torch.Tensor, "batch pos"]) -> str | list:
    tokenizer = experiment.experiment_data.masked_runner.masked_transformer.model.tokenizer
    if tokenizer is not None:
        return [tokenizer.decode(token) for token in input]
    return input.tolist()


top_k_out = 3  # how many of the most likely output to show


def analyse_metrics(metrics: Float[torch.Tensor, " batch"], circuit: list[Edge]) -> tuple[float, torch.Tensor]:
    topk_most_adversarial = torch.topk(metrics, k=5, sorted=True)
    topk_most_adversarial_input = test_data[topk_most_adversarial.indices, :]
    topk_most_adversarial_patch_input = test_patch_data[topk_most_adversarial.indices, :]

    loss = topk_most_adversarial.values

    # output of circuit and output of full model, decoded
    output_circuit = experiment.experiment_data.masked_runner.run(
        input=topk_most_adversarial_input,
        patch_input=topk_most_adversarial_patch_input,
        edges_to_ablate=list(experiment.experiment_data.masked_runner.all_ablatable_edges - set(circuit)),
    )

    top_k_most_likely_output = torch.topk(output_circuit[:, -1, :], k=top_k_out)
    # [batch, order_in_most_likely] -> vocab_index
    top_k_most_likely_output_decoded = decode(top_k_most_likely_output)

    output_model = experiment.experiment_data.masked_runner.run(
        input=topk_most_adversarial_input,
        patch_input=topk_most_adversarial_patch_input,
        edges_to_ablate=list(experiment.experiment_data.masked_runner.all_ablatable_edges - set(circuit)),
    )

    top_k_most_likely_output_model = torch.topk(output_model[:, -1, :], k=top_k_out)
    # [batch, order_in_most_likely] -> vocab_index
    top_k_most_likely_output_model_decoded = decode(top_k_most_likely_output_model)

    return topk_most_adversarial.values.tolist(), topk_most_adversarial_input


analyse_metrics(metrics_with_random_circuit, random_circuit)

topk_most_adversarial = torch.topk(metrics_with_random_circuit, k=5, sorted=True)
topk_most_adversarial_input = test_data[topk_most_adversarial.indices, :]
topk_most_adversarial_patch_input = test_patch_data[topk_most_adversarial.indices, :]
# For each of the three circuits, calculate
# - topk most adversarial input (decoded),
# - maybe also calculate the output?
# - the los
