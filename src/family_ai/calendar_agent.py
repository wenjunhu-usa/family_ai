from __future__ import annotations

import urllib.parse
from datetime import datetime, timedelta, timezone

from .gmail_backup import GmailBackupAgent, GmailBackupError

CALENDAR_API = "https://www.googleapis.com/calendar/v3"


class CalendarAgent:
    """Read-only Google Calendar specialist sharing an approved Google token."""

    def __init__(self, google: GmailBackupAgent):
        self.google = google

    def upcoming(self, member_id: str, account: str | None = None, days: int = 7) -> dict:
        days = max(1, min(int(days), 31))
        if account == "*":
            accounts = self.google.accounts(member_id)
            if not accounts:
                raise GmailBackupError("Gmail is not connected for this family member")
            events = []
            for selected_account in accounts:
                result = self.upcoming(member_id, selected_account, days)
                events.extend(dict(item, account=selected_account) for item in result["events"])
            events.sort(key=lambda item: str(item.get("start") or ""))
            return {"account": "*", "accounts": accounts, "days": days,
                    "events": events, "verified": True}
        selected = self.google._select_account(member_id, account)
        token = self.google._access_token(member_id, selected)
        start = datetime.now(timezone.utc)
        params = urllib.parse.urlencode({
            "timeMin": start.isoformat().replace("+00:00", "Z"),
            "timeMax": (start + timedelta(days=days)).isoformat().replace("+00:00", "Z"),
            "singleEvents": "true", "orderBy": "startTime", "maxResults": "100",
        })
        try:
            data = self.google.transport(
                f"{CALENDAR_API}/calendars/primary/events?{params}",
                headers={"Authorization": f"Bearer {token}"},
            )
        except GmailBackupError as exc:
            if "403" in str(exc):
                raise GmailBackupError(
                    f"Calendar read permission is missing for {selected}; reconnect this Google account"
                ) from exc
            raise
        events = []
        for item in data.get("items", []):
            events.append({
                "id": str(item.get("id", "")),
                "summary": str(item.get("summary") or "(untitled)"),
                "start": item.get("start", {}).get("dateTime") or item.get("start", {}).get("date"),
                "end": item.get("end", {}).get("dateTime") or item.get("end", {}).get("date"),
                "location": str(item.get("location") or ""),
            })
        return {"account": selected, "days": days, "events": events, "verified": True}
