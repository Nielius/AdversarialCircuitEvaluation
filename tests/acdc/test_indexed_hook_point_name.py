from acdc.TLACDCEdge import IndexedHookPointName, TorchIndex


# want to test if you can have them in sets


def test_indexed_hook_point_name_set():
    hook_points = {
        IndexedHookPointName(
            hook_name="hook_mlp_in",
            index=TorchIndex([None,None, 2]),
        ),
        # deliberately the same as the above!
        IndexedHookPointName(
            hook_name="hook_mlp_in",
            index=TorchIndex([None,None, 2]),
        ),
        IndexedHookPointName(
            hook_name="hook_mlp_out",
            index=TorchIndex([None,None, 2]),
        ),
    }

    assert len(hook_points) == 2