"""Gate actions per zone, derived from its risk level and the exit graph (plan Layer 3).

    GREEN   normal operation
    YELLOW  monitor; prepare marshals at the gates feeding the zone
    ORANGE  hold inflow at the main gate feeding the zone; open the best other exit
    RED     stop inflow at every gate feeding the zone; open all exits; announce the route out

"Gates feeding a zone" are the exit gates people walk in from: those within FEED_SLACK (1.5x) of
the nearest gate's walking distance on the uncongested graph. The exit to open is the cheapest
by live, congestion-weighted cost (ExitRouter.get_best_exits) that is not the gate being held.
The layout is the schematic placeholder of app.exit_router, so gate names are its node names.
"""

from __future__ import annotations

from typing import Sequence

import networkx as nx

from app.exit_router import ExitRouter
from app.risk_engine import RiskLevel

FEED_SLACK = 1.5


def place_name(node: str) -> str:
    """EXIT_GATE_A -> "Gate A", RAMKUND_NORTH -> "Ramkund north", GHAT_1 -> "Ghat 1"."""
    if node.startswith("EXIT_GATE_"):
        return f"Gate {node.removeprefix('EXIT_GATE_')}"
    words = node.lower().split("_")
    return " ".join([words[0].capitalize(), *words[1:]])


def feeding_gates(router: ExitRouter, node: str, slack: float = FEED_SLACK) -> list[str]:
    """The exit gates people walk in from, nearest first (by walking distance, ignoring crowding)."""
    if node not in router.graph:
        return []
    with router._lock:
        distances = nx.single_source_dijkstra_path_length(router.graph.reverse(copy=False), node, weight="base_weight")
    gates = sorted((distances[g], g) for g in router.exits if g in distances and g != node)
    if not gates:
        return []
    nearest = gates[0][0]
    return [g for d, g in gates if d <= nearest * slack]


def _route(path: Sequence[str]) -> str:
    return " > ".join(place_name(p) for p in path)


def gate_actions(level: str, router: ExitRouter | None, node: str | None, exits: Sequence[dict],
                 hotspot: dict | None = None) -> list[dict]:
    """[{kind, text, gates}] for a zone at `level`; `exits` are its best exits now (best first)."""
    level = RiskLevel(level)
    feeds = feeding_gates(router, node) if router is not None and node else []
    names = ", ".join(place_name(g) for g in feeds)
    where = f" ({hotspot['label']} of the view)" if hotspot else ""

    if level == RiskLevel.GREEN:
        return [{"kind": "normal", "text": "Normal operation: no action needed.", "gates": []}]
    if level == RiskLevel.YELLOW:
        return [
            {"kind": "monitor", "text": f"Monitor the crowd closely{where}.", "gates": []},
            {"kind": "prepare", "text": f"Prepare marshals at {names}." if feeds else "Prepare marshals at the entry points.",
             "gates": feeds},
        ]
    if level == RiskLevel.ORANGE:
        hold = feeds[:1]
        other = next((e for e in exits if e["exit"] not in hold), None)
        actions = [{"kind": "hold_inflow",
                    "text": f"Hold inflow at {place_name(hold[0])}." if hold else "Hold inflow into this area.",
                    "gates": hold}]
        if other:
            actions.append({"kind": "open_exit", "gates": [other["exit"]],
                            "text": f"Open alternate exit {place_name(other['exit'])} (route: {_route(other['path'])})."})
        else:
            actions.append({"kind": "open_exit", "text": "Open alternate exits.", "gates": []})
        return actions
    all_exits = sorted(router.exits) if router is not None else []
    best = exits[0] if exits else None
    return [
        {"kind": "stop_inflow", "gates": feeds,
         "text": f"STOP inflow at all gates feeding this zone: {names}." if feeds else "STOP all inflow into this area."},
        {"kind": "open_all_exits", "gates": all_exits,
         "text": f"Open all exits: {', '.join(place_name(g) for g in all_exits)}." if all_exits else "Open all exits."},
        {"kind": "announce", "gates": [best["exit"]] if best else [],
         "text": (f"Announce: move away from the crush{where}, leave via {place_name(best['exit'])} "
                  f"({_route(best['path'])}).") if best else f"Announce: move away from the crush{where} calmly."},
    ]
