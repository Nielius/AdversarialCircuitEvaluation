from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, TypeAlias


class EdgeType(Enum):
    """
    Property of edges in the computational graph - either

    ADDITION: the child (hook_name, index) is a sum of the parent (hook_name, index)s
    DIRECT_COMPUTATION The *single* child is a function of and only of the parent (e.g the value hooked by hook_q is a function of what hook_q_input saves).
    PLACEHOLDER generally like 2. but where there are generally multiple parents. Here in ACDC we just include these edges by default when we find them. Explained below?

    Q: Why do we do this?

    There are two answers to this question: A1 is an interactive notebook, see <a href="https://colab.research.google.com/github/ArthurConmy/Automatic-Circuit-Discovery/blob/main/notebooks/colabs/ACDC_Editing_Edges_Demo.ipynb">this Colab notebook</a>, which is in this repo at notebooks/implementation_demo.py. A2 is an answer that is written here below, but probably not as clear as A1 (though shorter).

    A2: We need something inside TransformerLens to represent the edges of a computational graph.
    The object we choose is pairs (hook_name, index). For example the output of Layer 11 Heads is a hook (blocks.11.attn.hook_result) and to sepcify the 3rd head we add the index [:, :, 3]. Then we can build a computational graph on these!

    However, when we do ACDC there turn out to be two conflicting things "removing edges" wants to do:
    i) for things in the residual stream, we want to remove the sum of the effects from previous hooks
    ii) for things that are not linear we want to *recompute* e.g the result inside the hook
    blocks.11.attn.hook_result from a corrupted Q and normal K and V

    The easiest way I thought of of reconciling these different cases, while also having a connected computational graph, is to have three types of edges: addition for the residual case, direct computation for easy cases where we can just replace hook_q with a cached value when we e.g cut it off from hook_q_input, and placeholder to make the graph connected (when hook_result is connected to hook_q and hook_k and hook_v)
    """

    ADDITION = 0
    DIRECT_COMPUTATION = 1
    PLACEHOLDER = 2

    def __eq__(self, other):
        """Necessary because of extremely frustrating error that arises with load_ext autoreload (because this uses importlib under the hood: https://stackoverflow.com/questions/66458864/enum-comparison-become-false-after-reloading-module)"""

        assert isinstance(other, EdgeType)
        return self.value == other.value


class EdgeInfo:
    def __init__(
        self,
        edge_type: EdgeType,
        present: bool = True,
        effect_size: Optional[float] = None,
    ):
        self.edge_type = edge_type
        self.present = present
        self.effect_size = effect_size

    def __repr__(self) -> str:
        return f"Edge({self.edge_type}, {self.present})"


HookPointName: TypeAlias = str


class TorchIndex:
    """There is not a clean bijection between things we
    want in the computational graph, and things that are hooked
    (e.g hook_result covers all heads in a layer)

    `TorchIndex`s are essentially indices that say which part of the tensor is being affected.

    EXAMPLES: Initialise [:, :, 3] with TorchIndex([None, None, 3]) and [:] with TorchIndex([None])

    Also we want to be able to call e.g `my_dictionary[my_torch_index]` hence the hashable tuple stuff

    Note: ideally this would be integrated with transformer_lens.utils.Slice in future; they are accomplishing similar but different things
    """

    def __init__(
        self,
        list_of_things_in_tuple: List,
    ):
        # check correct types
        for arg in list_of_things_in_tuple:
            if type(arg) in [type(None), int]:
                continue
            else:
                assert isinstance(arg, list)
                assert all([type(x) == int for x in arg])

        # make an object that can be indexed into a tensor
        self.as_index = tuple([slice(None) if x is None else x for x in list_of_things_in_tuple])

        # make an object that can be hashed (so used as a dictionary key)
        self.hashable_tuple = tuple(list_of_things_in_tuple)

    def __hash__(self):
        return hash(self.hashable_tuple)

    def __eq__(self, other):
        return self.hashable_tuple == other.hashable_tuple

    # some graphics things

    def __repr__(
        self, use_actual_colon=True
    ) -> (
        str
    ):  # graphviz, an old library used to dislike actual colons in strings, but this shouldn't be an issue anymore
        ret = "["
        for idx, x in enumerate(self.hashable_tuple):
            if idx > 0:
                ret += ", "
            if x is None:
                ret += ":" if use_actual_colon else "COLON"
            elif type(x) == int:
                ret += str(x)
            else:
                raise NotImplementedError(x)
        ret += "]"
        return ret

    def graphviz_index(self, use_actual_colon=True) -> str:
        return self.__repr__(use_actual_colon=use_actual_colon)


def is_attn_hook_point(name: HookPointName) -> bool:
    if "mlp" in name or "resid" in name or "embed" in name or name == "blocks.0.hook_resid_pre":
        return False
    return True


@dataclass(eq=True, frozen=True, slots=True)
class IndexedHookPointName:
    hook_name: HookPointName
    index: TorchIndex

    def __repr__(self) -> str:
        return f"IndexedHookID({self.hook_name}, {self.index})"

    def __str__(self) -> str:
        return f"{self.hook_name}{self.index}"

    @classmethod
    def list_from_hook_point(cls, name: HookPointName, n_heads: int) -> list["IndexedHookPointName"]:
        """Provides a list of all IndexedHookPointNames that are sub-components of the given HookPointName."""
        if is_attn_hook_point(name):
            return [
                IndexedHookPointName(
                    hook_name=name,
                    index=TorchIndex([None, None, i]),
                )
                for i in range(n_heads)
            ]
        else:
            return [
                IndexedHookPointName(
                    hook_name=name,
                    index=TorchIndex([None]),
                )
            ]


@dataclass(eq=True, frozen=True, slots=True)
class Edge:
    """An edge in the computational graph, pointing from parent to child."""

    child: IndexedHookPointName
    parent: IndexedHookPointName

    @classmethod
    def from_tuple_format(
        cls,
        child_hook_name: HookPointName,
        child_hook_index: TorchIndex | list | tuple,
        parent_hook_name: HookPointName,
        parent_hook_index: TorchIndex | list | tuple,
    ) -> "Edge":
        child_hook_torch_index = (
            TorchIndex(child_hook_index) if not isinstance(child_hook_index, TorchIndex) else child_hook_index
        )
        parent_hook_torch_index = (
            TorchIndex(parent_hook_index) if not isinstance(parent_hook_index, TorchIndex) else parent_hook_index
        )

        return cls(
            child=IndexedHookPointName(hook_name=child_hook_name, index=child_hook_torch_index),
            parent=IndexedHookPointName(hook_name=parent_hook_name, index=parent_hook_torch_index),
        )


@dataclass
class EdgeWithInfo:
    """An edge in the computational graph, pointing from parent to child."""

    child: IndexedHookPointName
    parent: IndexedHookPointName
    edge_info: EdgeInfo

    def __repr__(self) -> str:
        return f"Edge({self.child}, {self.parent}, {self.edge_info})"

    def __str__(self) -> str:
        return f"{self.child} -> {self.parent} ({self.edge_info})"

    def to_tuple_format(self) -> tuple[tuple[HookPointName, TorchIndex, HookPointName, TorchIndex], EdgeInfo]:
        """This is the format used by the TLACDCCorrespondence object. Might want to deprecate it."""
        return (self.child.hook_name, self.child.index, self.parent.hook_name, self.parent.index), self.edge_info


# I think i want to deprecate this kind of object
EdgeCollection: TypeAlias = dict[tuple[HookPointName, TorchIndex, HookPointName, TorchIndex], EdgeInfo]
