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


def test_rich_comparison_operators():
    """I have now implemented this with the dataclass order=True parameter, but am still keeping these tests to make
    sure it does what we expect."""
    assert IndexedHookPointName(
        hook_name="blocks.11.hook_resid_post", index=TorchIndex([None])
    ) <= IndexedHookPointName(hook_name="blocks.12.hook_resid_post", index=TorchIndex([None]))

    assert IndexedHookPointName(
        hook_name="blocks.11.hook_resid_post", index=TorchIndex([None])
    ) <= IndexedHookPointName(hook_name="blocks.11.hook_resid_post", index=TorchIndex([None]))

    assert not IndexedHookPointName(
        hook_name="blocks.11.hook_resid_post", index=TorchIndex([None, 1])
    ) <= IndexedHookPointName(hook_name="blocks.11.hook_resid_post", index=TorchIndex([None]))

    # lexicographic ordering, with parent first
    assert Edge(
        child=IndexedHookPointName(hook_name="blocks.11.hook_resid_post", index=TorchIndex([None])),
        parent=IndexedHookPointName(hook_name="blocks.10.attn.hook_result", index=TorchIndex([0, 0, 7])),
    ) <= Edge(
        child=IndexedHookPointName(hook_name="blocks.10.hook_resid_post", index=TorchIndex([None])),
        parent=IndexedHookPointName(hook_name="blocks.10.attn.hook_result", index=TorchIndex([0, 0, 8])),
    )
