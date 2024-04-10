from acdc.TLACDCEdge import Edge, IndexedHookPointName, TorchIndex


def test_edge_tuple_conversion():
    edge = Edge(
        child=IndexedHookPointName(hook_name="blocks.11.hook_resid_post", index=TorchIndex([None])),
        parent=IndexedHookPointName(hook_name="blocks.10.attn.hook_result", index=TorchIndex([0, 0, 7])),
    )
    tuple_format = (
        "blocks.11.hook_resid_post",
        TorchIndex([None]),
        "blocks.10.attn.hook_result",
        TorchIndex([0, 0, 7]),
    )

    assert edge.to_tuple_format() == tuple_format
    assert edge == Edge.from_tuple_format(*edge.to_tuple_format())
    assert tuple_format == Edge.from_tuple_format(*tuple_format).to_tuple_format()
