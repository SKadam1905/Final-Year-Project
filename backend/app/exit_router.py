"""Exit routing over the Ramkund Ghat walkway graph (plan Layer 3).

A NetworkX DiGraph of monitored areas, ghats and exit gates, with every walkway in both
directions. Crowd pressure in an area raises the cost of every walkway *leaving* it, i.e.
of getting through or out of that area:

    weight(u -> v) = base_metres(u, v) * (1 + alpha * pressure(u)),   pressure clamped to 0..10

Dijkstra then ranks the exits by congestion-weighted distance. With alpha = 1 a RED area
(pressure 6) costs 7x its length, so routes detour around it whenever a reasonable
alternative exists. (A gentler 1 + pressure/10 would cost only 1.6x, and the router would
still send people through a RED ghat to save a few tens of metres.) The worker refreshes
pressures every 2 s.

The layout and distances are a schematic placeholder: replace them with walkways and
distances measured on the site map (e.g. traced from OpenStreetMap) before quoting routes.
"""

from __future__ import annotations

import threading

import networkx as nx

DEFAULT_NODES = [
    "RAMKUND_NORTH", "RAMKUND_SOUTH", "RAMKUND_EAST", "RAMKUND_WEST",
    "GHAT_1", "GHAT_2", "GHAT_3",
    "EXIT_GATE_A", "EXIT_GATE_B", "EXIT_GATE_C", "EXIT_GATE_D",
]

DEFAULT_EDGES = [  # (from, to, metres) - walkable both ways
    ("RAMKUND_NORTH", "GHAT_1", 50),
    ("RAMKUND_NORTH", "RAMKUND_EAST", 80),
    ("RAMKUND_SOUTH", "GHAT_2", 50),
    ("RAMKUND_SOUTH", "RAMKUND_WEST", 80),
    ("RAMKUND_EAST", "GHAT_3", 60),
    ("RAMKUND_EAST", "EXIT_GATE_A", 120),
    ("RAMKUND_WEST", "EXIT_GATE_B", 100),
    ("GHAT_1", "EXIT_GATE_A", 90),
    ("GHAT_1", "EXIT_GATE_C", 70),
    ("GHAT_2", "EXIT_GATE_B", 80),
    ("GHAT_2", "EXIT_GATE_D", 110),
    ("GHAT_3", "EXIT_GATE_C", 60),
    ("GHAT_3", "EXIT_GATE_D", 90),
    ("EXIT_GATE_A", "EXIT_GATE_B", 200),
    ("EXIT_GATE_C", "EXIT_GATE_D", 150),
]

EXIT_GATES = {"EXIT_GATE_A", "EXIT_GATE_B", "EXIT_GATE_C", "EXIT_GATE_D"}

MAX_PRESSURE = 10.0


class ExitRouter:
    def __init__(self, edges=DEFAULT_EDGES, exits=EXIT_GATES, alpha: float = 1.0) -> None:
        self.graph = nx.DiGraph()
        for u, v, metres in edges:
            for a, b in ((u, v), (v, u)):
                self.graph.add_edge(a, b, base_weight=float(metres), weight=float(metres))
        nx.set_node_attributes(self.graph, 0.0, "pressure")
        self.exits = frozenset(exits)
        self.alpha = alpha
        self._lock = threading.Lock()

    def update_pressure(self, zone_id: str, crowd_pressure: float) -> None:
        """Re-weight every walkway leaving `zone_id` from its base length (non-destructive)."""
        if zone_id not in self.graph:
            raise KeyError(f"{zone_id!r} is not a node of the exit graph")
        pressure = min(max(float(crowd_pressure), 0.0), MAX_PRESSURE)
        with self._lock:
            self.graph.nodes[zone_id]["pressure"] = pressure
            for _, _, data in self.graph.out_edges(zone_id, data=True):
                data["weight"] = data["base_weight"] * (1.0 + self.alpha * pressure)

    def get_best_exits(self, from_zone: str, top_k: int = 3) -> list[dict]:
        """The `top_k` cheapest exits as {"exit", "cost", "path"}; [] for an unknown node."""
        if from_zone not in self.graph:
            return []
        with self._lock:
            costs, paths = nx.single_source_dijkstra(self.graph, from_zone, weight="weight")
        ranked = sorted((costs[e], e) for e in self.exits if e in costs and e != from_zone)
        return [{"exit": e, "cost": round(c, 1), "path": paths[e]} for c, e in ranked[:top_k]]
