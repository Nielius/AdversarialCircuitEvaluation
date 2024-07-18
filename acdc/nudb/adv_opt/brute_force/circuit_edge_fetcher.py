import random
from dataclasses import dataclass
from enum import Enum

from acdc.nudb.adv_opt.data_fetchers import AdvOptExperimentData
from acdc.TLACDCEdge import Edge


class CircuitType(str, Enum):
    RANDOM = "random"
    CANONICAL = "canonical"
    CORRUPTED_CANONICAL = "corrupted_canonical"
    FULL_MODEL = "full_model"
    UNKNOWN = "unknown"


@dataclass(kw_only=True)
class CircuitSpec:
    circuit_type: CircuitType
    edges: list[Edge]


@dataclass(kw_only=True)
class CorruptedCanonicalCircuitSpec(CircuitSpec):
    removed_edges: list[Edge]  # removed from the canonical circuit
    circuit_type: CircuitType = CircuitType.CORRUPTED_CANONICAL


def get_circuit_edges(
    rng: random.Random, experiment_data: AdvOptExperimentData, circuit_type: CircuitType
) -> CircuitSpec:
    match circuit_type:
        case CircuitType.CANONICAL:
            return CircuitSpec(
                circuit_type=CircuitType.CANONICAL,
                edges=experiment_data.circuit_edges,
            )
        case CircuitType.FULL_MODEL:
            return CircuitSpec(
                circuit_type=CircuitType.FULL_MODEL,
                edges=list(experiment_data.masked_runner.all_ablatable_edges),
            )
        case CircuitType.RANDOM:
            ablatable_edges_sorted = sorted(
                experiment_data.masked_runner.all_ablatable_edges
            )  # sort so that iterating over it is deterministic
            return CircuitSpec(
                circuit_type=CircuitType.RANDOM, edges=[edge for edge in ablatable_edges_sorted if rng.random() < 0.4]
            )
        case CircuitType.CORRUPTED_CANONICAL:
            circuit = get_circuit_edges(rng, experiment_data, CircuitType.CANONICAL)
            removed_edges = list(rng.choices(circuit.edges, k=2))
            return CorruptedCanonicalCircuitSpec(
                removed_edges=removed_edges, edges=sorted(set(circuit.edges) - set(removed_edges))
            )
        case _:
            raise ValueError(f"Unknown circuit type: {circuit_type}")
