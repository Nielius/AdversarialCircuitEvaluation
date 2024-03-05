import json

from acdc.nudb.adv_opt.edge_serdes import EdgeJSONDecoder, EdgeJSONEncoder
from acdc.TLACDCEdge import Edge, IndexedHookPointName, TorchIndex


def test_encode_then_decode():
    s = json.dumps(
        e := Edge(
            parent=IndexedHookPointName("parent_hook", TorchIndex([None, 1, None])),
            child=IndexedHookPointName("child_Hook", TorchIndex([None, 2, None])),
        ),
        cls=EdgeJSONEncoder,
    )

    assert e == json.loads(s, cls=EdgeJSONDecoder)


def test_decode_then_code():
    s = '{"_type": "edge", "parent": {"_type": "indexed_hook_point_name", "hook_name": "parent_hook", "index": [null, 1, null]}, "child": {"_type": "indexed_hook_point_name", "hook_name": "child_Hook", "index": [null, 2, null]}}'

    assert s == json.dumps(json.loads(s, cls=EdgeJSONDecoder), cls=EdgeJSONEncoder)
