import pytest

from app.exit_router import DEFAULT_NODES, EXIT_GATES, ExitRouter


@pytest.fixture
def router():
    return ExitRouter()


def test_default_graph_has_all_nodes(router):
    assert set(router.graph.nodes) == set(DEFAULT_NODES)


def test_walkways_go_both_ways(router):
    assert all(router.graph.has_edge(v, u) for u, v in router.graph.edges)


def test_get_best_exits_returns_top3(router):
    exits = router.get_best_exits("RAMKUND_NORTH")
    assert len(exits) == 3
    assert [e["cost"] for e in exits] == sorted(e["cost"] for e in exits)


def test_exits_are_exit_gates(router):
    assert all(e["exit"] in EXIT_GATES for e in router.get_best_exits("RAMKUND_NORTH", top_k=4))


def test_paths_run_from_zone_to_exit(router):
    for e in router.get_best_exits("RAMKUND_SOUTH"):
        assert e["path"][0] == "RAMKUND_SOUTH" and e["path"][-1] == e["exit"]


def test_cost_is_positive(router):
    assert all(e["cost"] > 0 for e in router.get_best_exits("RAMKUND_NORTH", top_k=4))


def test_unknown_zone_returns_empty(router):
    assert router.get_best_exits("UNKNOWN_ZONE") == []


def test_update_pressure_increases_cost(router):
    baseline = router.get_best_exits("RAMKUND_NORTH")[0]["cost"]
    router.update_pressure("RAMKUND_NORTH", 8.0)
    assert router.get_best_exits("RAMKUND_NORTH")[0]["cost"] > baseline


def test_update_pressure_is_non_destructive(router):
    baseline = router.get_best_exits("RAMKUND_NORTH")
    router.update_pressure("RAMKUND_NORTH", 8.0)
    router.update_pressure("RAMKUND_NORTH", 0.0)
    assert router.get_best_exits("RAMKUND_NORTH") == baseline


def test_pressure_is_clamped_to_ten(router):
    router.update_pressure("GHAT_1", 50.0)
    assert router.graph.edges["GHAT_1", "EXIT_GATE_C"]["weight"] == pytest.approx(70.0 * 11)


def test_crowded_ghat_is_avoided(router):
    # Calm: north steps -> GHAT_1 -> EXIT_GATE_C (50 + 70 m) is best.
    assert router.get_best_exits("RAMKUND_NORTH")[0]["path"] == ["RAMKUND_NORTH", "GHAT_1", "EXIT_GATE_C"]
    router.update_pressure("GHAT_1", 6.0)  # RED
    assert "GHAT_1" not in router.get_best_exits("RAMKUND_NORTH")[0]["path"]


def test_update_unknown_node_raises(router):
    with pytest.raises(KeyError):
        router.update_pressure("UNKNOWN_ZONE", 3.0)
