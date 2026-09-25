from unittest.mock import Mock
import pytest

from family_ai.calendar_agent import CalendarAgent
from family_ai.gmail_backup import GmailBackupError


def test_calendar_reads_and_normalizes_upcoming_events():
    google = Mock()
    google._select_account.return_value = "first@example.com"
    google._access_token.return_value = "token"
    google.transport.return_value = {"items": [{
        "id": "event-1", "summary": "Doctor",
        "start": {"dateTime": "2026-08-14T10:00:00-05:00"},
        "end": {"dateTime": "2026-08-14T11:00:00-05:00"},
            "location": "*",
    }]}
    result = CalendarAgent(google).upcoming("member", "first@example.com", 3)
    assert result["verified"] is True
    assert result["events"][0]["summary"] == "Doctor"
    assert "calendar/v3/calendars/primary/events" in google.transport.call_args.args[0]


def test_calendar_request_routes_without_model():
    from family_ai.planner import ModelPlanner
    plan = ModelPlanner.__new__(ModelPlanner).plan("查看 first@example.com 未来3天日程")
    assert (plan.tool, plan.account, plan.days) == ("calendar", "first@example.com", 3)


def test_calendar_all_accounts_routes_and_combines_events():
    from family_ai.planner import ModelPlanner
    plan = ModelPlanner.__new__(ModelPlanner).plan("查看所有 Gmail 账户未来 7 天的 Calendar 日程")
    assert (plan.tool, plan.account, plan.days) == ("calendar", "*", 7)

    google = Mock()
    google.accounts.return_value = ["first@example.com", "second@example.com"]
    google._select_account.side_effect = lambda _member, account: account
    google._access_token.return_value = "token"
    google.transport.side_effect = [
        {"items": [{"id": "a", "summary": "A", "start": {"date": "2026-08-15"}}]},
        {"items": [{"id": "b", "summary": "B", "start": {"date": "2026-08-14"}}]},
    ]
    result = CalendarAgent(google).upcoming("member", "*", 7)
    assert [item["account"] for item in result["events"]] == ["second@example.com", "first@example.com"]


def test_calendar_scope_error_identifies_account():
    google = Mock()
    google._select_account.return_value = "first@example.com"
    google._access_token.return_value = "token"
    google.transport.side_effect = GmailBackupError("Google API returned HTTP 403")
    with pytest.raises(GmailBackupError, match="first@example.com"):
        CalendarAgent(google).upcoming("member", "first@example.com", 7)
