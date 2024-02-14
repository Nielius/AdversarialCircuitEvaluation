import itertools

import wandb
import os
from collections import defaultdict
import pickle
import torch
import numpy as np
import huggingface_hub
import datetime
from typing import Dict, Union, Iterable
import wandb
import plotly.graph_objects as go
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import warnings
import networkx as nx
from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.TLACDCEdge import EdgeCollection, HookPointName, TorchIndex
from acdc.TLACDCInterpNode import TLACDCInterpNode
from acdc.acdc_utils import EdgeType
import pygraphviz as pgv
from pathlib import Path

EDGE_TYPE_COLORS = {
    EdgeType.ADDITION.value: "#FF0000",  # Red
    EdgeType.DIRECT_COMPUTATION.value: "#00FF00",  # Green
    EdgeType.PLACEHOLDER.value: "#0000FF",  # Blue
}


def generate_random_color(colorscheme: str) -> str:
    """
    https://stackoverflow.com/questions/28999287/generate-random-colors-rgb
    """
    import cmapy

    def rgb2hex(rgb):
        """
        https://stackoverflow.com/questions/3380726/converting-an-rgb-color-tuple-to-a-hexidecimal-string
        """
        return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

    return rgb2hex(cmapy.color("Pastel2", np.random.randint(0, 256), rgb_order=True))


def get_pretty_graph_name_for_interp_node(node: TLACDCInterpNode, show_full_index=True) -> str:
    """Node name for use in pretty graphs"""
    return get_pretty_graph_name_for_node(node.name, node.index, show_full_index=show_full_index)


def get_pretty_graph_name_for_node(node_name: HookPointName, node_index: TorchIndex, show_full_index=True) -> str:
    """Node name for use in pretty graphs"""

    if not show_full_index:
        name = ""
        qkv_substrings = [f"hook_{letter}" for letter in ["q", "k", "v"]]
        qkv_input_substrings = [f"hook_{letter}_input" for letter in ["q", "k", "v"]]

        # Handle embedz
        if "resid_pre" in node_name:
            assert "0" in node_name and not any([str(i) in node_name for i in range(1, 10)])
            name += "embed"
            if len(node_index.hashable_tuple) > 2:
                name += f"_[{node_index.hashable_tuple[2]}]"
            return name

        elif "embed" in node_name:
            name = "pos_embeds" if "pos" in node_name else "token_embeds"

        # Handle q_input and hook_q etc
        elif any([node_name.endswith(qkv_input_substring) for qkv_input_substring in qkv_input_substrings]):
            relevant_letter = None
            for letter, qkv_substring in zip(["q", "k", "v"], qkv_substrings):
                if qkv_substring in node_name:
                    assert relevant_letter is None
                    relevant_letter = letter
            name += "a" + node_name.split(".")[1] + "." + str(node_index.hashable_tuple[2]) + "_" + relevant_letter

        # Handle attention hook_result
        elif "hook_result" in node_name or any([qkv_substring in node_name for qkv_substring in qkv_substrings]):
            name = "a" + node_name.split(".")[1] + "." + str(node_index.hashable_tuple[2])

        # Handle MLPs
        elif node_name.endswith("resid_mid"):
            raise ValueError("We removed resid_mid annotations. Call these mlp_in now.")
        elif node_name.endswith("mlp_out") or node_name.endswith("mlp_in"):
            name = "m" + node_name.split(".")[1]

        # Handle resid_post
        elif "resid_post" in node_name:
            name += "resid_post"

        else:
            raise ValueError(f"Unrecognized node name {node_name}")

    else:
        name = node_name + str(node_index.graphviz_index(use_actual_colon=True))

    return "<" + name + ">"


def build_random_colorscheme_for_correspondence(
    correspondence: TLACDCCorrespondence, colorscheme: str = "Pastel2", show_full_index=True
) -> Dict[str, str]:
    colors = {}
    for node in correspondence.nodes_list():
        colors[get_pretty_graph_name_for_interp_node(node, show_full_index=show_full_index)] = generate_random_color(
            colorscheme
        )
    return colors


def build_random_colorscheme_for_edge_collection(
    edge_collection: EdgeCollection, colorscheme: str = "Pastel2", show_full_index=True
) -> dict[str, str]:
    return build_random_colorscheme_for_nodes(
        itertools.chain(
            ((node_name, node_index) for node_name, node_index, _, _ in edge_collection.keys()),
            ((node_name, node_index) for _, _, node_name, node_index in edge_collection.keys()),
        ),
        colorscheme,
        show_full_index=show_full_index,
    )


def build_random_colorscheme_for_nodes(
    nodes: Iterable[tuple[HookPointName, TorchIndex]], colorscheme: str = "Pastel2", show_full_index=True
) -> dict[str, str]:
    return {
        get_pretty_graph_name_for_node(node_name, node_index, show_full_index=show_full_index): generate_random_color(
            colorscheme
        )
        for node_name, node_index in nodes
    }


def show(
    correspondence: TLACDCCorrespondence,
    fname=None,
    colorscheme: Union[Dict, str] = "Pastel2",
    minimum_penwidth: float = 0.3,
    show_full_index: bool = True,
    remove_self_loops: bool = True,
    remove_qkv: bool = False,
    layout: str = "dot",
    edge_type_colouring: bool = False,
    show_placeholders: bool = False,
    seed: Optional[int] = None,
) -> pgv.AGraph:
    """
    Colorscheme: a color for each node name, or a string corresponding to a cmapy color scheme
    """
    return graph_from_edges(
        correspondence.edge_dict(),
        filename=fname,
        colorscheme=build_random_colorscheme_for_correspondence(
            correspondence, colorscheme, show_full_index=show_full_index
        ),
        minimum_penwidth=minimum_penwidth,
        show_full_index=show_full_index,
        remove_self_loops=remove_self_loops,
        remove_qkv=remove_qkv,
        layout=layout,
        edge_type_colouring=edge_type_colouring,
        show_placeholders=show_placeholders,
        seed=seed,
    )


def graph_from_edges(
    edge_collection: EdgeCollection,
    filename: str | None = None,
    colorscheme: Union[Dict, str] = "Pastel2",
    minimum_penwidth: float = 0.3,
    show_full_index: bool = True,
    remove_self_loops: bool = True,
    remove_qkv: bool = False,
    layout: str = "dot",
    edge_type_colouring: bool = False,
    show_placeholders: bool = False,
    show_everything: bool = False,
    seed: Optional[int] = None,
) -> pgv.AGraph:
    """
    Colorscheme: a color for each node name, or a string corresponding to a cmapy color scheme

    filename: if not None, save the graph to this filename
    """
    g = pgv.AGraph(directed=True, bgcolor="transparent", overlap="false", splines="true", layout=layout)

    if seed is not None:
        np.random.seed(seed)

    color_groups = {}
    if isinstance(colorscheme, str):
        colors = build_random_colorscheme_for_edge_collection(
            edge_collection,
            colorscheme,
            show_full_index=show_full_index,
        )
    else:
        colors = colorscheme
        for name, color in colors.items():
            if color not in color_groups:
                color_groups[color] = [name]
            else:
                color_groups[color].append(name)

    node_pos = {}
    if filename is not None:
        filename_without_extension = ".".join(str(filename).split(".")[:-1])

        dir_for_groups = Path(filename_without_extension)
        fpath = dir_for_groups / "layout.gv"
        if fpath.exists():
            g_pos = pgv.AGraph()
            g_pos.read(fpath)
            for node in g_pos.nodes():
                node_pos[node.name] = node.attr["pos"]

    # create all nodes and edges
    for (child_hook_name, child_index, parent_hook_name, parent_index), edge in edge_collection.items():
        parent_name = get_pretty_graph_name_for_node(parent_hook_name, parent_index, show_full_index=show_full_index)
        child_name = get_pretty_graph_name_for_node(child_hook_name, child_index, show_full_index=show_full_index)

        if remove_qkv:
            parent_name = parent_name.replace("_q>", ">").replace("_k>", ">").replace("_v>", ">")
            child_name = child_name.replace("_q>", ">").replace("_k>", ">").replace("_v>", ">")

        if remove_self_loops and parent_name == child_name:
            # Important this go after the qkv removal
            continue

        if not show_everything:
            if (
                (not edge.present)
                or edge.effect_size is None
                or ((not show_placeholders) and edge.edge_type == EdgeType.PLACEHOLDER)
            ):
                continue

        for node_name in [parent_name, child_name]:
            maybe_pos = {}
            if node_name in node_pos:
                maybe_pos["pos"] = node_pos[node_name]
            g.add_node(
                node_name,
                fillcolor=colors[node_name],
                color="black",
                style="filled, rounded",
                shape="box",
                fontname="Helvetica",
                **maybe_pos,
            )

        g.add_edge(
            parent_name,
            child_name,
            penwidth=str(max(minimum_penwidth, edge.effect_size or 0) * 2),
            color=colors[parent_name] if not edge_type_colouring else EDGE_TYPE_COLORS[edge.edge_type.value],
        )

    if filename is not None:
        assert len(filename.split(".")) > 1, "filename must have an extension"
        filename_without_extension = Path(filename).stem

        dir_for_groups = Path(filename_without_extension)
        dir_for_groups.mkdir(exist_ok=True)
        for k, s in color_groups.items():
            g2 = pgv.AGraph(directed=True, bgcolor="transparent", overlap="false", splines="true", layout="neato")
            for node_name in s:
                g2.add_node(
                    node_name,
                    style="filled, rounded",
                    shape="box",
                )
            for i in range(len(s)):
                for j in range(i + 1, len(s)):
                    g2.add_edge(s[i], s[j], style="invis", weight=200)
            g2.write(path=dir_for_groups / f"{k}.gv")

        g.write(path=filename_without_extension + ".gv")

        if not filename.endswith(".gv"):  # turn the .gv file into a .png file
            g.draw(path=filename, prog="dot")

    return g


# -------------------------------------------
# WANDB
# -------------------------------------------


def do_plotly_plot_and_log(
    experiment,
    x: List[int],
    y: List[float],
    plot_name: str,
    metadata: Optional[List[str]] = None,
) -> None:
    # Create a plotly plot with metadata
    fig = go.Figure(data=[go.Scatter(x=x, y=y, mode="lines+markers", text=metadata)])
    wandb.log({plot_name: fig})


def log_metrics_to_wandb(
    experiment,
    current_metric: Optional[float] = None,
    parent_name=None,
    child_name=None,
    evaluated_metric=None,
    result=None,
    picture_fname=None,
    times=None,
) -> None:
    """Arthur added Nones so that just some of the metrics can be plotted"""

    experiment.metrics_to_plot["new_metrics"].append(experiment.cur_metric)
    experiment.metrics_to_plot["list_of_nodes_evaluated"].append(str(experiment.current_node))
    if parent_name is not None:
        experiment.metrics_to_plot["list_of_parents_evaluated"].append(parent_name)
    if child_name is not None:
        experiment.metrics_to_plot["list_of_children_evaluated"].append(child_name)
    if evaluated_metric is not None:
        experiment.metrics_to_plot["evaluated_metrics"].append(evaluated_metric)
    if current_metric is not None:
        experiment.metrics_to_plot["current_metrics"].append(current_metric)
    if result is not None:
        experiment.metrics_to_plot["results"].append(result)
    if experiment.skip_edges != "yes":
        experiment.metrics_to_plot["num_edges"].append(experiment.count_num_edges())
    if times is not None:
        experiment.metrics_to_plot["times"].append(times)
        experiment.metrics_to_plot["times_diff"].append(  # hopefully fixes
            0
            if len(experiment.metrics_to_plot["times"]) == 1
            else (experiment.metrics_to_plot["times"][-1] - experiment.metrics_to_plot["times"][-2])
        )

    experiment.metrics_to_plot["acdc_step"] += 1
    list_of_timesteps = [i + 1 for i in range(experiment.metrics_to_plot["acdc_step"])]
    if experiment.metrics_to_plot["acdc_step"] > 1:
        if result is not None:
            for y_name in ["results"]:
                if len(list_of_timesteps) % 20 == 19:
                    do_plotly_plot_and_log(
                        experiment,
                        x=list_of_timesteps,
                        y=experiment.metrics_to_plot[y_name],
                        metadata=[
                            f"{parent_string} to {child_string}"
                            for parent_string, child_string in zip(
                                experiment.metrics_to_plot["list_of_parents_evaluated"],
                                experiment.metrics_to_plot["list_of_children_evaluated"],
                            )
                        ],
                        plot_name=y_name,
                    )

        if evaluated_metric is not None:
            do_plotly_plot_and_log(
                experiment,
                x=list_of_timesteps,
                y=experiment.metrics_to_plot["evaluated_metrics"],
                metadata=experiment.metrics_to_plot["list_of_nodes_evaluated"],
                plot_name="evaluated_metrics",
            )
            do_plotly_plot_and_log(
                experiment,
                x=list_of_timesteps,
                y=experiment.metrics_to_plot["current_metrics"],
                metadata=experiment.metrics_to_plot["list_of_nodes_evaluated"],
                plot_name="current_metrics",
            )

        # Arthur added... I think wandb graphs have a lot of desirable properties
        if experiment.skip_edges != "yes":
            wandb.log({"num_edges_total": experiment.metrics_to_plot["num_edges"][-1]})
        wandb.log({"experiment.cur_metric": experiment.metrics_to_plot["current_metrics"][-1]})
        if experiment.second_metric is not None:
            wandb.log({"experiment.second_metric": experiment.cur_second_metric})

        if picture_fname is not None:  # presumably this is more expensive_update_cur
            wandb.log(
                {
                    "acdc_graph": wandb.Image(picture_fname),
                }
            )


# -------------------------------------------
# utilities for ROC and AUC
# -------------------------------------------


def pessimistic_auc(xs, ys):
    # Sort indices based on 'x' and 'y'
    i = np.lexsort(
        (ys, xs)
    )  # lexsort sorts by the last column first, then the second last, etc., i.e we firstly sort by x and then y to break ties

    xs = np.array(xs, dtype=np.float64)[i]
    ys = np.array(ys, dtype=np.float64)[i]

    dys = np.diff(ys)
    assert np.all(np.diff(xs) >= 0), "not sorted"
    assert np.all(dys >= 0), "not monotonically increasing"

    # The slabs of the stairs
    area = np.sum((1 - xs)[1:] * dys)
    return area


assert pessimistic_auc([0, 1], [0, 1]) == 0.0
assert pessimistic_auc([0, 0.5, 1], [0, 0.5, 1]) == 0.5**2
assert pessimistic_auc([0, 0.25, 1], [0, 0.25, 1]) == 0.25 * 0.75
assert pessimistic_auc([0, 0.25, 0.5, 1], [0, 0.25, 0.5, 1]) == 5 / 16
assert pessimistic_auc([0, 0.25, 0.75, 1], [0, 0.25, 0.5, 1]) == 4 / 16


def dict_merge(dct, merge_dct):
    """Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.

    Copyright 2016-2022 Paul Durivage, licensed under Apache License https://gist.github.com/angstwad/bf22d1822c38a92ec0a9

    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    for k in merge_dct.keys():
        if k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], dict):  # noqa
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]
