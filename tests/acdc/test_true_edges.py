from acdc.ioi.ioi_data_fetchers import get_all_ioi_things
from acdc.ioi.utils import get_ioi_true_edges
from acdc.TLACDCEdge import Edge, IndexedHookPointName, TorchIndex


def test_true_edges_for_ioi():
    ioi_things = get_all_ioi_things(num_examples=6, device="cpu", metric_name="kl_div")
    ioi_true_edges = get_ioi_true_edges(ioi_things.tl_model)
    del ioi_things

    # check a few edges
    assert ioi_true_edges[("blocks.11.hook_resid_post", (None,), "blocks.10.attn.hook_result", (None, None, 7))]

    # check another edge, but using the Edge class
    child_name, child_index, parent_name, parent_index = Edge(
        child=IndexedHookPointName(hook_name="blocks.11.hook_q_input", index=TorchIndex([None, None, 10])),
        parent=IndexedHookPointName(hook_name="blocks.5.hook_mlp_out", index=TorchIndex([None])),
    ).to_tuple_format()
    assert ioi_true_edges[(child_name, child_index.hashable_tuple, parent_name, parent_index.hashable_tuple)]
