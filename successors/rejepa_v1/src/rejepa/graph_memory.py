"""Causal graph relations for engram values and cross-asset context."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TemporalRelation:
    source: str
    target: str
    weight: float
    available_at_ns: int
    relation_type: str


class EngramKnowledgeGraph:
    def __init__(self, nodes: tuple[str, ...], relations: tuple[TemporalRelation, ...]) -> None:
        if not nodes or len(set(nodes)) != len(nodes):
            raise ValueError("Graph nodes must be nonempty and unique")
        if any(
            relation.source not in nodes
            or relation.target not in nodes
            or not torch.isfinite(torch.tensor(relation.weight))
            for relation in relations
        ):
            raise ValueError("Graph relation is invalid")
        self.nodes = nodes
        self.relations = relations
        self.index = {node: index for index, node in enumerate(nodes)}

    def adjacency(self, as_of_ns: int, *, device: torch.device | None = None) -> torch.Tensor:
        matrix = torch.zeros(len(self.nodes), len(self.nodes), device=device)
        for relation in self.relations:
            if relation.available_at_ns <= as_of_ns:
                matrix[self.index[relation.source], self.index[relation.target]] += relation.weight
        matrix = matrix + torch.eye(len(self.nodes), device=device)
        scale = matrix.abs().sum(dim=1, keepdim=True).clamp_min(1.0)
        return matrix / scale

    def propagate(self, node_values: torch.Tensor, as_of_ns: int) -> torch.Tensor:
        if node_values.ndim != 2 or node_values.shape[0] != len(self.nodes):
            raise ValueError("Expected node_values [nodes, features]")
        if not torch.isfinite(node_values).all():
            raise ValueError("Node values must be finite")
        return self.adjacency(as_of_ns, device=node_values.device) @ node_values
