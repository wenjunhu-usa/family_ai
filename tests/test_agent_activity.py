from family_ai.agent_activity import active_agent_ids, agent_activity

def test_agent_activity_tracks_nested_work_and_cleans_up():
    assert "builtin-search" not in active_agent_ids()
    with agent_activity("builtin-search"):
        assert "builtin-search" in active_agent_ids()
        with agent_activity("builtin-search"):
            assert "builtin-search" in active_agent_ids()
        assert "builtin-search" in active_agent_ids()
    assert "builtin-search" not in active_agent_ids()
