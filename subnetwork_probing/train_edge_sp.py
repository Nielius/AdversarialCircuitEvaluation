# %%

import argparse
import gc
import os
import pickle
import random
from typing import Callable

import torch
from tqdm import tqdm

import wandb
from acdc.acdc_utils import reset_network
from acdc.docstring.utils import (
    AllDataThings,
    get_all_docstring_things,
    get_docstring_subgraph_true_edges,
)
from acdc.greaterthan.utils import (
    get_all_greaterthan_things,
    get_greaterthan_true_edges,
)
from acdc.induction.utils import get_all_induction_things  # , get_induction_true_edges
from acdc.ioi.utils import get_all_ioi_things, get_ioi_true_edges
from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.TLACDCEdge import EdgeType
from acdc.tracr_task.utils import (
    get_all_tracr_things,
    get_tracr_proportion_edges,
    get_tracr_reverse_edges,
)
from subnetwork_probing.sp_utils import (
    MaskedTransformer,
    edge_level_corr,
    print_stats,
    set_ground_truth_edges,
)


def save_edges(corr: TLACDCCorrespondence, fname: str):
    edges_list = []
    for t, e in corr.all_edges().items():
        if e.present and e.edge_type != EdgeType.PLACEHOLDER:
            edges_list.append((t, e.effect_size))

    with open(fname, "wb") as f:
        pickle.dump(edges_list, f)


def train_edge_sp(
    args,
    masked_model: MaskedTransformer,
    all_task_things: AllDataThings,
    print_every: int = 100,
    get_true_edges: Callable = None,
):
    print(f"Using memory {torch.cuda.memory_allocated():_} bytes at training start")
    epochs = args.epochs
    lambda_reg = args.lambda_reg

    torch.manual_seed(args.seed)

    wandb.init(
        name=args.wandb_name,
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        config=args,
        dir=args.wandb_dir,
        mode=args.wandb_mode,
    )
    test_metric_fns = all_task_things.test_metrics
    print(args)
    print("Reset subject:", args.reset_subject)
    if args.reset_subject:
        reset_network(args.task, args.device, masked_model.model)
        gc.collect()
        torch.cuda.empty_cache()
        masked_model.freeze_weights()

        with torch.no_grad():
            reset_logits = masked_model.model(all_task_things.validation_data)
            print(
                "Reset validation metric: ",
                all_task_things.validation_metric(reset_logits),
            )
            reset_logits = masked_model.model(all_task_things.test_data)
            print(
                "Reset test metric: ",
                {k: v(reset_logits).item() for k, v in all_task_things.test_metrics.items()},
            )

    # one parameter per thing that is masked
    mask_params = list(p for p in masked_model.mask_logits if p.requires_grad)
    # parameters for the probe (we don't use a probe)
    model_params = list(p for p in masked_model.model.parameters() if p.requires_grad)
    assert len(model_params) == 0, ("MODEL should be empty", model_params)
    trainer = torch.optim.Adam(mask_params, lr=args.lr)

    print(f"Using memory {torch.cuda.memory_allocated():_} bytes after optimizer init")
    if args.zero_ablation:
        valid_context_args = test_context_args = dict(ablation="zero")
    else:
        valid_context_args = dict(ablation="resample", ablation_data=all_task_things.validation_patch_data)
        test_context_args = dict(ablation="resample", ablation_data=all_task_things.test_patch_data)

    # Get canonical subgraph so we can print TPR, FPR
    canonical_circuit_subgraph = TLACDCCorrespondence.setup_from_model(
        masked_model.model, use_pos_embed=masked_model.use_pos_embed
    )
    try:
        d_trues = set(get_true_edges())
        set_ground_truth_edges(canonical_circuit_subgraph, d_trues)
    except:
        pass

    for epoch in tqdm(range(epochs)):  # tqdm.notebook.tqdm(range(epochs)):
        masked_model.train()
        trainer.zero_grad()
        with masked_model.with_fwd_hooks_and_new_cache(**valid_context_args) as hooked_model:
            # print(f"Using memory {torch.cuda.memory_allocated():_} bytes before forward")
            metric_loss = all_task_things.validation_metric(hooked_model(all_task_things.validation_data))
            # print(f"Using memory {torch.cuda.memory_allocated():_} bytes after forward")
        regularizer_term = masked_model.regularization_loss()
        loss = metric_loss + regularizer_term * lambda_reg
        loss.backward()

        trainer.step()

        if epoch % print_every == 0 and args.print_stats:
            statss = []
            for i in range(3):  # sample multiple times to get average edge_tpr etc.
                corr = edge_level_corr(masked_model)
                try:
                    stats = print_stats(corr, canonical_circuit_subgraph, do_print=False)
                    statss.append(stats)
                except:
                    pass
            stats = {k: sum(s[k] for s in statss) / len(statss) for k in statss[0]}
            with torch.no_grad():
                with masked_model.with_fwd_hooks_and_new_cache(**test_context_args) as hooked_model:
                    test_metric_loss = all_task_things.validation_metric(hooked_model(all_task_things.test_data))
            test_loss = test_metric_loss + regularizer_term * lambda_reg

            wandb.log(
                {
                    "epoch": epoch,
                    "num_edges": masked_model.num_edges(),
                    "regularization_loss": regularizer_term.item(),
                    "validation_metric_loss": metric_loss.item(),
                    "test_metric_loss": test_metric_loss.item(),
                    "total_loss": loss.item(),
                    "test_total_loss": test_loss,
                }
                | stats
            )
            # TODO edit this to create a corr from masked edges
            # number_of_nodes, nodes_to_mask = visualize_mask(masked_model)
            # corr, _ = iterative_correspondence_from_mask(masked_model.model, nodes_to_mask)
            # print_stats(corr, d_trues, canonical_circuit_subgraph)

    # Save edges to create data for plots later
    corr = edge_level_corr(masked_model)
    edges_fname = "edges.pth"  # note this is a pickle file
    wandb_dir = os.environ.get("WANDB_DIR")
    if wandb_dir is None:
        save_edges(corr, edges_fname)
    else:
        save_edges(corr, os.path.join(wandb_dir, edges_fname))
    artifact = wandb.Artifact(edges_fname, type="dataset")
    artifact.add_file(edges_fname)
    wandb.log_artifact(artifact)
    os.remove(edges_fname)

    # Now calculate final metrics
    with torch.no_grad():
        # The loss has a lot of variance so let's just average over a few runs with the same seed
        rng_state = torch.random.get_rng_state()

        # Final training loss
        metric_loss = 0.0
        if args.zero_ablation:
            masked_model.do_zero_caching()
        else:
            masked_model.do_random_resample_caching(all_task_things.validation_patch_data)

        for _ in range(args.n_loss_average_runs):
            with masked_model.with_fwd_hooks_and_new_cache(**valid_context_args) as hooked_model:
                metric_loss += all_task_things.validation_metric(hooked_model(all_task_things.validation_data)).item()
        print(f"Final train/validation metric: {metric_loss:.4f}")

        if args.zero_ablation:
            masked_model.do_zero_caching()
        else:
            masked_model.do_random_resample_caching(all_task_things.test_patch_data)

        test_specific_metrics = {}
        for k, fn in test_metric_fns.items():
            torch.random.set_rng_state(rng_state)
            test_specific_metric_term = 0.0
            # Test loss
            for _ in range(args.n_loss_average_runs):
                with masked_model.with_fwd_hooks_and_new_cache(**valid_context_args) as hooked_model:
                    test_specific_metric_term += fn(hooked_model(all_task_things.test_data)).item()
            test_specific_metrics[f"test_{k}"] = test_specific_metric_term

        print(f"Final test metric: {test_specific_metrics}")

        log_dict = dict(
            # number_of_nodes=number_of_nodes,
            specific_metric=metric_loss,
            # nodes_to_mask=nodes_to_mask,
            **test_specific_metrics,
        )
    return masked_model, log_dict


def proportion_of_binary_scores(model: MaskedTransformer) -> float:
    """How many of the scores are binary, i.e. 0 or 1
    (after going through the sigmoid with fp32 precision loss)
    """
    binary_count = 0
    total_count = 0

    for mask_name in model.mask_logits_names:
        mask = model.sample_mask(mask_name)
        for v in mask.view(-1):
            total_count += 1
            if v == 0 or v == 1:
                binary_count += 1
    return binary_count / total_count


parser = argparse.ArgumentParser("python train_edge_sp.py")
parser.add_argument("--wandb-name", type=str, required=False)
parser.add_argument("--wandb-project", type=str, default="subnetwork-probing")
parser.add_argument("--wandb-entity", type=str, required=True)
parser.add_argument("--wandb-group", type=str, required=True)
parser.add_argument("--wandb-dir", type=str, default="/tmp/wandb")
parser.add_argument("--wandb-mode", type=str, default="online")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--loss-type", type=str, required=True)
parser.add_argument("--epochs", type=int, default=3000)
parser.add_argument("--verbose", type=int, default=1)
parser.add_argument("--lambda-reg", type=float, default=100)
parser.add_argument("--zero-ablation", type=int, required=True)
parser.add_argument("--reset-subject", type=int, default=0)
parser.add_argument(
    "--seed",
    type=int,
    default=random.randint(0, 2**31 - 1),
    help="Random seed (default: random)",
)
parser.add_argument("--num-examples", type=int, default=50)
parser.add_argument("--seq-len", type=int, default=300)
parser.add_argument("--n-loss-average-runs", type=int, default=4)
parser.add_argument("--task", type=str, required=True)
parser.add_argument(
    "--torch-num-threads",
    type=int,
    default=0,
    help="How many threads to use for torch (0=all)",
)
parser.add_argument("--print-stats", type=int, default=1, required=False)

# %%
if __name__ == "__main__":
    args = parser.parse_args()

    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    torch.manual_seed(args.seed)

    if args.task == "ioi":
        all_task_things = get_all_ioi_things(
            num_examples=args.num_examples,
            device=torch.device(args.device),
            metric_name=args.loss_type,
        )
        get_true_edges = lambda: get_ioi_true_edges(all_task_things.tl_model)  # noqa: E731
    elif args.task == "induction":
        all_task_things = get_all_induction_things(
            args.num_examples,
            args.seq_len,
            device=torch.device(args.device),
            metric=args.loss_type,
        )
        # get_true_edges = get_induction_true_edges # missing because there is no canonical induction circuit -tkwa
    elif args.task == "tracr-reverse":
        all_task_things = get_all_tracr_things(
            task="reverse",
            metric_name=args.loss_type,
            num_examples=args.num_examples,
            device=torch.device(args.device),
        )
        get_true_edges = get_tracr_reverse_edges
    elif args.task == "tracr-proportion":
        all_task_things = get_all_tracr_things(
            task="proportion",
            metric_name=args.loss_type,
            num_examples=args.num_examples,
            device=torch.device(args.device),
        )
        get_true_edges = get_tracr_proportion_edges
    elif args.task == "docstring":
        all_task_things = get_all_docstring_things(
            num_examples=args.num_examples,
            seq_len=args.seq_len,
            device=torch.device(args.device),
            metric_name=args.loss_type,
            correct_incorrect_wandb=True,
        )
        get_true_edges = get_docstring_subgraph_true_edges
    elif args.task == "greaterthan":
        all_task_things = get_all_greaterthan_things(
            num_examples=args.num_examples,
            metric_name=args.loss_type,
            device=args.device,
        )
        get_true_edges = lambda: get_greaterthan_true_edges(all_task_things.tl_model)  # noqa: E731
    else:
        raise ValueError(f"Unknown task {args.task}")

    masked_model = MaskedTransformer(all_task_things.tl_model)
    masked_model = masked_model.to(args.device)

    masked_model.freeze_weights()
    print("Finding subnetwork...")
    masked_model, log_dict = train_edge_sp(
        args=args,
        masked_model=masked_model,
        all_task_things=all_task_things,
        get_true_edges=get_true_edges,
    )

    percentage_binary = proportion_of_binary_scores(masked_model)

    # Update dict with some different things
    # log_dict["nodes_to_mask"] = list(map(str, log_dict["nodes_to_mask"]))
    # to_log_dict["number_of_edges"] = corr.count_no_edges() TODO
    log_dict["percentage_binary"] = percentage_binary

    wandb.log(log_dict)

    wandb.finish()
