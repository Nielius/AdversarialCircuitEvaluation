import argparse
import collections
import gc
import math
import random
from typing import Callable, ContextManager, Dict, List, Optional, Tuple

import torch
import wandb
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.hook_points import HookPoint

from acdc.acdc_utils import filter_nodes, get_edge_stats, get_node_stats, get_present_nodes, reset_network
from acdc.docstring.utils import AllDataThings, get_all_docstring_things, get_docstring_subgraph_true_edges
from acdc.greaterthan.utils import get_all_greaterthan_things
from acdc.induction.utils import get_all_induction_things
from acdc.ioi.utils import get_all_ioi_things
from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.TLACDCEdge import Edge, EdgeType, TorchIndex
from acdc.TLACDCInterpNode import TLACDCInterpNode
from acdc.tracr_task.utils import get_all_tracr_things


def iterative_correspondence_from_mask(
    model: HookedTransformer,
    nodes_to_mask: List[TLACDCInterpNode], # Can be empty
    use_pos_embed: bool = False,
    corr: Optional[TLACDCCorrespondence] = None,
    head_parents: Optional[List] = None,
) -> Tuple[TLACDCCorrespondence, List]:
    """Given corr has some nodes masked, also mask the nodes_to_mask"""

    assert (corr is None) == (head_parents is None), "Ensure we're either masking from scratch or we provide details on `head_parents`"

    if corr is None:
        corr = TLACDCCorrespondence.setup_from_model(model, use_pos_embed=use_pos_embed)
    if head_parents is None:
        head_parents = collections.defaultdict(lambda: 0)

    additional_nodes_to_mask = []

    for node in nodes_to_mask:
        additional_nodes_to_mask.append(
            TLACDCInterpNode(node.name.replace(".attn.", ".") + "_input", node.index, EdgeType.ADDITION)
        )

        if node.name.endswith("_q") or node.name.endswith("_k") or node.name.endswith("_v"):
            child_name = node.name.replace("_q", "_result").replace("_k", "_result").replace("_v", "_result")
            head_parents[(child_name, node.index)] += 1

            if head_parents[(child_name, node.index)] == 3:
                additional_nodes_to_mask.append(TLACDCInterpNode(child_name, node.index, EdgeType.PLACEHOLDER))

            # Forgot to add these in earlier versions of Subnetwork Probing, and so the edge counts were inflated
            additional_nodes_to_mask.append(TLACDCInterpNode(child_name + "_input", node.index, EdgeType.ADDITION))

        if node.name.endswith(("mlp_in", "resid_mid")):
            additional_nodes_to_mask.append(
                TLACDCInterpNode(
                    node.name.replace("resid_mid", "mlp_out").replace("mlp_in", "mlp_out"),
                    node.index,
                    EdgeType.DIRECT_COMPUTATION,
                )
            )

    assert all([v <= 3 for v in head_parents.values()]), "We should have at most three parents (Q, K and V, connected via placeholders)"

    for node in nodes_to_mask + additional_nodes_to_mask:
        # Mark edges where this is child as not present
        rest2 = corr.edges[node.name][node.index]
        for rest3 in rest2.values():
            for edge in rest3.values():
                edge.present = False

        # Mark edges where this is parent as not present
        for rest1 in corr.edges.values():
            for rest2 in rest1.values():
                if node.name in rest2 and node.index in rest2[node.name]:
                    rest2[node.name][node.index].present = False

    return corr, head_parents


def log_plotly_bar_chart(x: List[str], y: List[float]) -> None:
    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Bar(x=x, y=y)])
    wandb.log({"mask_scores": fig})


class MaskedTransformer(torch.nn.Module):
    model: HookedTransformer
    cache: ActivationCache
    mask_logits: torch.nn.ParameterList
    mask_logits_names: List[str]
    _mask_logits_dict: Dict[str, torch.nn.Parameter]

    def __init__(self, model, beta=2 / 3, gamma=-0.1, zeta=1.1, mask_init_p=0.9):
        super().__init__()

        self.model = model
        self.mask_logits = torch.nn.ParameterList()
        self.mask_logits_names = []
        self._mask_logits_dict = {}
        self.cache = ActivationCache({}, self.model)
        # Hyperparameters
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta
        self.mask_init_p = mask_init_p

        # Copied from subnetwork probing code. Similar to log odds (but not the same)
        p = (self.mask_init_p - self.gamma) / (self.zeta - self.gamma)
        mask_init_constant = math.log(p / (1 - p))

        for layer_index, layer in enumerate(model.blocks):
            # MLP: turn on/off
            mask_name = f"blocks.{layer_index}.hook_mlp_out"
            self.mask_logits.append(torch.nn.Parameter(torch.zeros((1,)) + mask_init_constant))
            self.mask_logits_names.append(mask_name)
            self._mask_logits_dict[mask_name] = self.mask_logits[-1]

            # QKV: turn each head on/off
            for q_k_v in ["q", "k", "v"]:
                mask_name = f"blocks.{layer_index}.attn.hook_{q_k_v}"
                self.mask_logits.append(torch.nn.Parameter(
                    torch.zeros((model.cfg.n_heads, 1)) + mask_init_constant
                ))
                self.mask_logits_names.append(mask_name)
                self._mask_logits_dict[mask_name] = self.mask_logits[-1]

    def sample_mask(self, mask_name):
        """Samples a binary-ish mask from the mask_scores for the particular `mask_name` activation"""
        mask_scores = self._mask_logits_dict[mask_name]
        uniform_sample = torch.zeros_like(mask_scores).uniform_().clamp_(0.0001, 0.9999)
        s = torch.sigmoid((uniform_sample.log() - (1 - uniform_sample).log() + mask_scores) / self.beta)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        mask = s_bar.clamp(min=0.0, max=1.0)
        return mask

    def regularization_loss(self) -> torch.Tensor:
        center = self.beta * math.log(-self.gamma / self.zeta)
        per_parameter_loss = [
            torch.sigmoid(scores - center).mean()
            for scores in self.mask_logits
        ]
        return torch.mean(torch.stack(per_parameter_loss))

    def mask_logits_names_filter(self, name):
        return name in self.mask_logits_names

    def do_random_resample_caching(self, patch_data) -> torch.Tensor:
        # Only cache the tensors needed to fill the masked out positions
        with torch.no_grad():
            model_out, self.cache = self.model.run_with_cache(
                patch_data, names_filter=self.mask_logits_names_filter, return_cache_object=True
            )
        return model_out

    def do_zero_caching(self):
        """Caches zero for every possible mask point.

        Note: the shape of this is the mask shape, instead of the activation shape like for
        `do_random_resample_caching`; this is ultimately fine due to broadcasting.
        """
        self.cache = ActivationCache(
            {name: torch.zeros_like(scores) for name, scores in self._mask_logits_dict.items()}, self.model
        )

    def activation_mask_hook(self, hook_point_out: torch.Tensor, hook: HookPoint):
        mask = self.sample_mask(hook.name)
        out = mask * hook_point_out + (1 - mask) * self.cache[hook.name]
        return out

    def fwd_hooks(self) -> List[Tuple[str, Callable]]:
        return [(n, self.activation_mask_hook) for n in self.mask_logits_names]

    def with_fwd_hooks(self) -> ContextManager[HookedTransformer]:
        return self.model.hooks(self.fwd_hooks())

    def freeze_weights(self):
        for p in self.model.parameters():
            p.requires_grad = False


def visualize_mask(masked_model: MaskedTransformer) -> tuple[int, list[TLACDCInterpNode]]:
    number_of_heads = masked_model.model.cfg.n_heads
    number_of_layers = masked_model.model.cfg.n_layers
    node_name_list = []
    mask_scores_for_names = []
    total_nodes = 0
    nodes_to_mask: list[TLACDCInterpNode] = []
    for layer_index in range(number_of_layers):
        for head_index in range(number_of_heads):
            for q_k_v in ["q", "k", "v"]:
                total_nodes += 1
                node_name = f"blocks.{layer_index}.attn.hook_{q_k_v}"
                mask_sample = masked_model.sample_mask(node_name)[head_index].cpu().item()

                node_name_with_index = f"{node_name}[{head_index}]"
                node_name_list.append(node_name_with_index)
                node = TLACDCInterpNode(
                    node_name, TorchIndex((None, None, head_index)), incoming_edge_type=EdgeType.ADDITION
                )

                mask_scores_for_names.append(mask_sample)
                if mask_sample < 0.5:
                    nodes_to_mask.append(node)

        # MLPs
        # This is actually fairly wrong for getting the exact nodes and edges we keep in the circuit but in the `filter_nodes` function
        # used in post-processing (in roc_plot_generator.py we process hook_resid_mid/mlp_in and mlp_out hooks together properly) we iron
        # these errors so that plots are correct
        node_name = f"blocks.{layer_index}.hook_mlp_out"
        mask_sample = masked_model.sample_mask(node_name).cpu().item()
        mask_scores_for_names.append(mask_sample)
        node_name_list.append(node_name)

        for node_name, edge_type in [
            (f"blocks.{layer_index}.hook_mlp_out", EdgeType.PLACEHOLDER),
            (f"blocks.{layer_index}.hook_resid_mid", EdgeType.ADDITION),
        ]:
            node = TLACDCInterpNode(node_name, TorchIndex([None]), incoming_edge_type=edge_type)
            total_nodes += 1

            if mask_sample < 0.5:
                nodes_to_mask.append(node)

    # assert len(mask_scores_for_names) == 3 * number_of_heads * number_of_layers
    log_plotly_bar_chart(x=node_name_list, y=mask_scores_for_names)
    node_count = total_nodes - len(nodes_to_mask)
    return node_count, nodes_to_mask


def train_sp(
    args,
    masked_model: MaskedTransformer,
    all_task_things: AllDataThings,
):
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

    print("Reset subject:", args.reset_subject)
    if args.reset_subject:
        reset_network(args.task, args.device, masked_model.model)
        gc.collect()
        torch.cuda.empty_cache()
        masked_model.freeze_weights()

        with torch.no_grad():
            reset_logits = masked_model.model(all_task_things.validation_data)
            print("Reset validation metric: ", all_task_things.validation_metric(reset_logits))
            reset_logits = masked_model.model(all_task_things.test_data)
            print("Reset test metric: ", {k: v(reset_logits).item() for k, v in all_task_things.test_metrics.items()})

    # one parameter per thing that is masked
    mask_params = list(p for p in masked_model.mask_logits if p.requires_grad)
    # parameters for the probe (we don't use a probe)
    model_params = list(p for p in masked_model.model.parameters() if p.requires_grad)
    assert len(model_params) == 0, ("MODEL should be empty", model_params)
    trainer = torch.optim.Adam(mask_params, lr=args.lr)


    if args.zero_ablation:
        masked_model.do_zero_caching()
    else:
        masked_model.do_random_resample_caching(all_task_things.validation_patch_data)

    for epoch in tqdm(range(epochs)):  # tqdm.notebook.tqdm(range(epochs)):
        masked_model.train()
        trainer.zero_grad()

        with masked_model.with_fwd_hooks() as hooked_model:
            specific_metric_term = all_task_things.validation_metric(hooked_model(all_task_things.validation_data))
        regularizer_term = masked_model.regularization_loss()
        loss = specific_metric_term + regularizer_term * lambda_reg
        loss.backward()

        trainer.step()

    number_of_nodes, nodes_to_mask = visualize_mask(masked_model)
    wandb.log(
        {
            "regularisation_loss": regularizer_term.item(),
            "specific_metric_loss": specific_metric_term.item(),
            "total_loss": loss.item(),
        }
    )

    with torch.no_grad():
        # The loss has a lot of variance so let's just average over a few runs with the same seed
        rng_state = torch.random.get_rng_state()

        # Final training loss
        specific_metric_term = 0.0
        if args.zero_ablation:
            masked_model.do_zero_caching()
        else:
            masked_model.do_random_resample_caching(all_task_things.validation_patch_data)

        for _ in range(args.n_loss_average_runs):
            with masked_model.with_fwd_hooks() as hooked_model:
                specific_metric_term += all_task_things.validation_metric(
                    hooked_model(all_task_things.validation_data)
                ).item()
        print(f"Final train/validation metric: {specific_metric_term:.4f}")

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
                with masked_model.with_fwd_hooks() as hooked_model:
                    test_specific_metric_term += fn(hooked_model(all_task_things.test_data)).item()
            test_specific_metrics[f"test_{k}"] = test_specific_metric_term

        print(f"Final test metric: {test_specific_metrics}")

        to_log_dict = dict(
            number_of_nodes=number_of_nodes,
            specific_metric=specific_metric_term,
            nodes_to_mask=nodes_to_mask,
            **test_specific_metrics,
        )
    return masked_model, to_log_dict


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


parser = argparse.ArgumentParser("train_induction")
parser.add_argument("--wandb-name", type=str, required=True)
parser.add_argument("--wandb-project", type=str, default="subnetwork-probing")
parser.add_argument("--wandb-entity", type=str, required=True)
parser.add_argument("--wandb-group", type=str, required=True)
parser.add_argument("--wandb-dir", type=str, default="/tmp/wandb")
parser.add_argument("--wandb-mode", type=str, default="online")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--loss-type", type=str, required=True)
parser.add_argument("--epochs", type=int, default=3000)
parser.add_argument("--verbose", type=int, default=1)
parser.add_argument("--lambda-reg", type=float, default=100)
parser.add_argument("--zero-ablation", type=int, required=True)
parser.add_argument("--reset-subject", type=int, default=0)
parser.add_argument("--seed", type=int, default=random.randint(0, 2**31 - 1), help="Random seed (default: random)")
parser.add_argument("--num-examples", type=int, default=50)
parser.add_argument("--seq-len", type=int, default=300)
parser.add_argument("--n-loss-average-runs", type=int, default=4)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--torch-num-threads", type=int, default=0, help="How many threads to use for torch (0=all)")
parser.add_argument("--print-stats", type=int, default=1, required=False)

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
    elif args.task == "induction":
        all_task_things = get_all_induction_things(
            args.num_examples,
            args.seq_len,
            device=torch.device(args.device),
            metric=args.loss_type,
        )
    elif args.task == "tracr-reverse":
        all_task_things = get_all_tracr_things(
            task="reverse", metric_name=args.loss_type, num_examples=args.num_examples, device=torch.device(args.device)
        )
    elif args.task == "tracr-proportion":
        all_task_things = get_all_tracr_things(
            task="proportion",
            metric_name=args.loss_type,
            num_examples=args.num_examples,
            device=torch.device(args.device),
        )
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
    else:
        raise ValueError(f"Unknown task {args.task}")

    masked_model = MaskedTransformer(all_task_things.tl_model)
    masked_model = masked_model.to(args.device)

    masked_model.freeze_weights()
    print("Finding subnetwork...")
    masked_model, to_log_dict = train_sp(
        args=args,
        masked_model=masked_model,
        all_task_things=all_task_things,
    )

    corr, _ = iterative_correspondence_from_mask(masked_model.model, to_log_dict["nodes_to_mask"])
    percentage_binary = proportion_of_binary_scores(masked_model)

    # Update dict with some different things
    to_log_dict["nodes_to_mask"] = list(map(str, to_log_dict["nodes_to_mask"]))
    to_log_dict["number_of_edges"] = corr.count_no_edges()
    to_log_dict["percentage_binary"] = percentage_binary

    wandb.log(to_log_dict)
    if args.print_stats:
        canonical_circuit_subgraph = TLACDCCorrespondence.setup_from_model(masked_model.model, use_pos_embed=False)
        d_trues = set(get_true_edges())

        for (receiver_name, receiver_index, sender_name, sender_index), edge in canonical_circuit_subgraph.all_edges().items():
            key =(receiver_name, receiver_index.hashable_tuple, sender_name, sender_index.hashable_tuple)
            edge.present = (key in d_trues)

        stats = get_node_stats(ground_truth=canonical_circuit_subgraph, recovered=corr)
        tpr = stats["true positive"] / (stats["true positive"] + stats["false negative"])
        fpr = stats["false positive"] / (stats["false positive"] + stats["true negative"])
        print(f"Node TPR: {tpr:.3f}. Node FPR: {fpr:.3f}")

        stats = get_edge_stats(ground_truth=canonical_circuit_subgraph, recovered=corr)
        tpr = stats["true positive"] / (stats["true positive"] + stats["false negative"])
        fpr = stats["false positive"] / (stats["false positive"] + stats["true negative"])
        print(f"Edge TPR: {tpr:.3f}. Edge FPR: {fpr:.3f}")

    wandb.finish()
