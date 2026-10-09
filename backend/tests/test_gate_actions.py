"""P4 gate actions from the risk level and the exit graph."""

from app.exit_router import ExitRouter
from app.gate_actions import feeding_gates, gate_actions, place_name
from tests.fakes import make_services

HOTSPOT = {"row": 0, "col": 2, "label": "top-right", "count": 120.0, "density": 6.5}


def texts(level, node="RAMKUND_NORTH", router=None, hotspot=None):
    router = router or ExitRouter()
    return [a["text"] for a in gate_actions(level, router, node, router.get_best_exits(node) if node else [], hotspot)]


def test_place_names():
    assert [place_name(n) for n in ("EXIT_GATE_A", "RAMKUND_NORTH", "GHAT_1")] == ["Gate A", "Ramkund north", "Ghat 1"]


def test_gates_feeding_a_zone_are_the_nearest_ones():
    router = ExitRouter()
    # Ramkund north: Gate C 120 m (via Ghat 1), Gate A 140 m, Gate D 230 m (> 1.5 x 120), Gate B 340 m
    assert feeding_gates(router, "RAMKUND_NORTH") == ["EXIT_GATE_C", "EXIT_GATE_A"]
    assert feeding_gates(router, "RAMKUND_SOUTH") == ["EXIT_GATE_B", "EXIT_GATE_D"]
    assert feeding_gates(router, "NOT_A_NODE") == []


def test_actions_for_each_level():
    assert texts("GREEN") == ["Normal operation: no action needed."]
    assert texts("YELLOW", hotspot=HOTSPOT) == ["Monitor the crowd closely (top-right of the view).",
                                               "Prepare marshals at Gate C, Gate A."]
    assert texts("ORANGE") == ["Hold inflow at Gate C.",
                               "Open alternate exit Gate A (route: Ramkund north > Ghat 1 > Gate A)."]
    assert texts("RED", hotspot=HOTSPOT) == [
        "STOP inflow at all gates feeding this zone: Gate C, Gate A.",
        "Open all exits: Gate A, Gate B, Gate C, Gate D.",
        "Announce: move away from the crush (top-right of the view), leave via Gate C (Ramkund north > Ghat 1 > Gate C).",
    ]


def test_the_exit_to_open_follows_live_congestion():
    router = ExitRouter()
    router.update_pressure("GHAT_1", 6.0)  # Ghat 1 is RED: routes through it cost 7x
    actions = texts("ORANGE", router=router)
    assert actions[0] == "Hold inflow at Gate C."
    assert actions[1].startswith("Open alternate exit Gate A (route: Ramkund north > Ramkund east > Gate A)")


def test_a_zone_off_the_exit_graph_gets_generic_actions():
    assert texts("ORANGE", node=None) == ["Hold inflow into this area.", "Open alternate exits."]
    assert texts("RED", node=None)[0] == "STOP all inflow into this area."


def test_gate_actions_reach_the_dashboard_and_the_alert():
    services = make_services(count=700)  # 7 /m²: RED
    events = services.worker.step()
    update = next(p for name, p in events if name == "density_update" and p["zone_id"] == "RAMKUND_NORTH")
    assert [a["kind"] for a in update["gate_actions"]] == ["stop_inflow", "open_all_exits", "announce"]
    alert = next(p for name, p in events if name == "alert" and p["zone_id"] == "RAMKUND_NORTH")
    assert "\nGate actions:\n- STOP inflow at all gates feeding this zone: Gate C, Gate A." in alert["message"]
    off_graph = next(p for name, p in events if name == "alert" and p["zone_id"] == "CAMERA_X")
    assert "- STOP all inflow into this area." in off_graph["message"]
