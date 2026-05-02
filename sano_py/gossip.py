# gossip.py
"""Simulated gossip layer (HashiCorp memberlist analogue).

Each agent broadcasts a 64-byte digest (replicas, EWMA mu, decision id)
every gossip_period_ms. We model this as a synchronous step that
exchanges state with k random peers per tick — convergence in O(log N).
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class GossipDigest:
    node: str
    replicas: int
    mu_rps: float
    decision_id: int


@dataclass
class GossipNode:
    name: str
    digest: GossipDigest
    peers_seen: dict = field(default_factory=dict)

    def receive(self, d: GossipDigest) -> None:
        prev = self.peers_seen.get(d.node)
        if prev is None or d.decision_id > prev.decision_id:
            self.peers_seen[d.node] = d


class GossipMesh:
    def __init__(self, fanout: int = 3, seed: int = 0):
        self.fanout = fanout
        self.nodes: dict[str, GossipNode] = {}
        self.rng = random.Random(seed)

    def join(self, node: GossipNode) -> None:
        self.nodes[node.name] = node

    def tick(self) -> None:
        names = list(self.nodes)
        for n in names:
            sender = self.nodes[n]
            others = [m for m in names if m != n]
            if not others:
                continue
            for peer in self.rng.sample(others, k=min(self.fanout, len(others))):
                self.nodes[peer].receive(sender.digest)

    def cluster_view(self, node: str) -> Iterable[GossipDigest]:
        return self.nodes[node].peers_seen.values()