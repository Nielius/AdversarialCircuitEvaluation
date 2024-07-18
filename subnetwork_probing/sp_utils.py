import wandb

from acdc.acdc_utils import get_edge_stats, get_node_stats
from acdc.TLACDCCorrespondence import TLACDCCorrespondence


def set_ground_truth_edges(canonical_circuit_subgraph: TLACDCCorrespondence, ground_truth_set: set):
    for (
        receiver_name,
        receiver_index,
        sender_name,
        sender_index,
    ), edge in canonical_circuit_subgraph.edge_dict().items():
        key = (
            receiver_name,
            receiver_index.hashable_tuple,
            sender_name,
            sender_index.hashable_tuple,
        )
        edge.present = key in ground_truth_set


def print_stats(recovered_corr, ground_truth_subgraph, do_print=True):
    """
    False positive = present in recovered_corr but not in ground_truth_set
    False negative = present in ground_truth_set but not in recovered_corr
    """
    # diff = set(recovered_corr.edge_dict().keys()) - set(ground_truth_subgraph.edge_dict().keys())
    # if diff: print(f"{len(diff)} key mismatches: {diff}")
    stats = get_node_stats(ground_truth=ground_truth_subgraph, recovered=recovered_corr)
    node_tpr = stats["true positive"] / (stats["true positive"] + stats["false negative"])
    node_fpr = stats["false positive"] / (stats["false positive"] + stats["true negative"])
    if do_print:
        print(f"Node TPR: {node_tpr:.3f}. Node FPR: {node_fpr:.3f}")

    stats = get_edge_stats(ground_truth=ground_truth_subgraph, recovered=recovered_corr)
    edge_tpr = stats["true positive"] / (stats["true positive"] + stats["false negative"])
    edge_fpr = stats["false positive"] / (stats["false positive"] + stats["true negative"])
    if do_print:
        print(f"Edge TPR: {edge_tpr:.3f}. Edge FPR: {edge_fpr:.3f}")

    return {
        "node_tpr": node_tpr,
        "node_fpr": node_fpr,
        "edge_tpr": edge_tpr,
        "edge_fpr": edge_fpr,
    }


def log_plotly_bar_chart(x: list[str], y: list[float]) -> None:
    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Bar(x=x, y=y)])
    wandb.log({"mask_scores": fig})
