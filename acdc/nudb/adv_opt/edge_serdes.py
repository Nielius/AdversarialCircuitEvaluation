import json

from acdc.TLACDCEdge import Edge, IndexedHookPointName, TorchIndex


class EdgeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Edge):
            return {
                "_type": "edge",
                "parent": obj.parent,
                "child": obj.child,
            }
        elif isinstance(obj, IndexedHookPointName):
            return {
                "_type": "indexed_hook_point_name",
                "hook_name": obj.hook_name,
                "index": obj.index.hashable_tuple,
            }
        return super(EdgeJSONEncoder, self).default(obj)


class EdgeJSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if "_type" not in obj:
            return obj
        type = obj["_type"]
        if type == "edge":
            return Edge(
                parent=IndexedHookPointName(obj["parent"]["hook_name"], TorchIndex(obj["parent"]["index"])),
                child=IndexedHookPointName(obj["child"]["hook_name"], TorchIndex(obj["child"]["index"])),
            )
        return obj
