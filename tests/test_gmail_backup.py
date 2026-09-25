import base64
import http.client
import json
import sqlite3
from unittest.mock import patch

import pytest

from family_ai.gmail_backup import (
    GmailBackupAgent, GmailBackupError, GmailBackupInterrupted,
    GmailBackupWorker, GmailDailyScheduler, _http_json,
)
from datetime import datetime


class MemoryTokenStore:
    def __init__(self):
        self.tokens = {}

    def load(self, member_id, email_address):
        return self.tokens.get((member_id, email_address))

    def save(self, member_id, email_address, token):
        self.tokens[(member_id, email_address)] = dict(token)

    def delete(self, member_id, email_address):
        self.tokens.pop((member_id, email_address), None)


class MemoryRegistry:
    def __init__(self):
        self.items = {}

    def register_google_account(self, member_id, email_address):
        self.items.setdefault(member_id, set()).add(email_address.casefold())

    def list_google_accounts(self, member_id):
        return sorted(self.items.get(member_id, set()))

    def remove_google_account(self, member_id, email_address):
        self.items.get(member_id, set()).discard(email_address.casefold())
        return True


def test_worker_survives_temporary_storage_failure():
    class FlakyStorage:
        def __init__(self):
            self.claims = 0
            self.recoveries = 0

        def recover_gmail_backup_jobs(self):
            self.recoveries += 1

        def claim_gmail_backup_job(self):
            self.claims += 1
            if self.claims == 1:
                raise ConnectionError("temporary database outage")
            return None

    storage = FlakyStorage()
    worker = GmailBackupWorker(object(), storage, poll_seconds=0.01)
    worker.start()
    for _ in range(100):
        if storage.claims >= 2:
            break
        __import__("time").sleep(0.01)
    worker.stop()
    assert storage.claims >= 2
    assert storage.recoveries >= 2


def test_daily_scheduler_submits_all_accounts_once_per_day(tmp_path):
    class Agent:
        def accounts(self, member_id):
            assert member_id == "member"
            return ["first@example.com", "second@example.com"]

    class Worker:
        def __init__(self):
            self.calls = []

        def submit(self, member_id, accounts):
            self.calls.append((member_id, accounts))
            return "job-1"

    worker = Worker()
    scheduler = GmailDailyScheduler(Agent(), worker, tmp_path / "schedule.sqlite",
                                    hour=3)
    now = datetime(2026, 8, 13, 4, 0)
    assert scheduler.run_due(now) == "job-1"
    assert scheduler.run_due(now) is None
    assert worker.calls == [("member", ["first@example.com", "second@example.com"])]

class MemoryArchive:
    def __init__(self):
        self.messages = {}

    def store_email_backup(self, member_id, account, message, raw_email, sha256):
        self.messages[(member_id, account, message["id"])] = {
            "gmail_message_id": message["id"], "thread_id": message.get("threadId"),
            "internal_date": message.get("internalDate"), "labels": message.get("labelIds", []),
            "raw_email": raw_email, "sha256": sha256,
        }

    def has_email_backup(self, member_id, account, message_id, sha256=None):
        row = self.messages.get((member_id, account, message_id))
        return bool(row and (sha256 is None or row["sha256"] == sha256))

    def email_backup_count(self, member_id, account):
        return sum(1 for key in self.messages if key[:2] == (member_id, account))

    def email_backup_ids(self, member_id, account):
        return {key[2] for key in self.messages if key[:2] == (member_id, account)}

    def iter_email_backups(self, member_id, account):
        for key, row in self.messages.items():
            if key[:2] == (member_id, account):
                yield row


def build_test_agent(tmp_path, transport):
    volume = tmp_path / "Volumes" / "USB"
    volume.mkdir(parents=True)
    store = MemoryTokenStore()
    registry = MemoryRegistry()
    registry.register_google_account("member", "one@example.com")
    store.save("member", "one@example.com", {"access_token": "old", "refresh_token": "refresh"})
    return GmailBackupAgent(
        "client", "secret", "http://localhost/callback", volume / "Gmail",
        token_store=store, transport=transport, volume_base=tmp_path / "Volumes",
        account_registry=registry,
    )


def test_backup_writes_eml_manifest_and_resumes(tmp_path):
    raw = b"From: sender@example.com\r\nSubject: Test\r\n\r\nHello"

    def transport(url, **kwargs):
        if "oauth2.googleapis.com" in url:
            return {"access_token": "fresh"}
        if "?format=raw" in url:
            return {"id": "abc123", "threadId": "thread1", "labelIds": ["INBOX"],
                    "internalDate": "1", "raw": base64.urlsafe_b64encode(raw).decode()}
        return {"messages": [{"id": "abc123", "threadId": "thread1"}]}

    agent = build_test_agent(tmp_path, transport)
    first = agent.backup("member")
    second = agent.backup("member")

    assert (first.target / "messages" / "ab" / "abc123.eml").read_bytes() == raw
    assert first.downloaded == 1 and second.downloaded == 0 and second.skipped == 1
    with sqlite3.connect(first.target / "manifest.sqlite") as db:
        assert db.execute("SELECT message_id FROM messages").fetchone()[0] == "abc123"
    assert json.loads((first.target / "last-backup.json").read_text())["downloaded"] == 0


def test_database_first_sync_does_not_write_disk_then_exports_on_request(tmp_path):
    raw = b"From: sender@example.com\r\nSubject: DB first\r\n\r\nHello"

    def transport(url, **kwargs):
        if "oauth2.googleapis.com" in url:
            return {"access_token": "fresh"}
        if "?format=raw" in url:
            return {"id": "db123", "threadId": "thread", "internalDate": "1",
                    "labelIds": ["INBOX"],
                    "raw": base64.urlsafe_b64encode(raw).decode()}
        return {"messages": [{"id": "db123"}]}

    agent = build_test_agent(tmp_path, transport)
    archive = MemoryArchive()
    agent.secondary_archive = archive
    synced = agent.sync_database("member")
    account_root = agent._member_root("member", "one@example.com")
    assert synced.downloaded == 1
    assert not account_root.exists()

    exported = agent.export_database_to_disk("member")
    assert exported.exported == 1
    assert (exported.target / "messages" / "db" / "db123.eml").read_bytes() == raw
    with sqlite3.connect(exported.target / "manifest.sqlite") as db:
        assert db.execute("SELECT postgres_stored FROM messages").fetchone()[0] == 1


def test_database_sync_resumes_from_latest_postgres_timestamp(tmp_path):
    requested_urls = []

    def transport(url, **kwargs):
        requested_urls.append(url)
        if "oauth2.googleapis.com" in url:
            return {"access_token": "fresh"}
        return {"messages": []}

    class Archive:
        def email_backup_ids(self, *_args):
            return {"existing"}

        def email_backup_latest_internal_date(self, *_args):
            return 1_700_000_000_000

    agent = build_test_agent(tmp_path, transport)
    agent.secondary_archive = Archive()
    agent.sync_database("member", "one@example.com")
    gmail_list_url = next(url for url in requested_urls if "/messages?" in url)
    assert "q=after%3A1697408000" in gmail_list_url


def test_backup_rejects_target_outside_volume_base(tmp_path):
    agent = GmailBackupAgent("id", "secret", "http://localhost", tmp_path / "unsafe",
                             token_store=MemoryTokenStore(), volume_base=tmp_path / "Volumes")
    with pytest.raises(GmailBackupError, match="removable-volume"):
        agent._member_root("member", "one@example.com")


def test_backup_stores_second_copy_and_repairs_old_manifest(tmp_path):
    raw = b"Subject: Dual copy\r\n\r\nBody"
    archive = MemoryArchive()

    def transport(url, **kwargs):
        if "oauth2.googleapis.com" in url:
            return {"access_token": "fresh"}
        if "format=" in url:
            return {"id": "dual1", "threadId": "t", "labelIds": ["INBOX"],
                    "internalDate": "2", "raw": base64.urlsafe_b64encode(raw).decode()}
        return {"messages": [{"id": "dual1"}]}

    agent = build_test_agent(tmp_path, transport)
    agent.secondary_archive = archive
    agent.require_secondary = True
    result = agent.backup("member")
    assert archive.messages[("member", "one@example.com", "dual1")]["raw_email"] == raw
    assert result.downloaded == 1

    with sqlite3.connect(result.target / "manifest.sqlite") as db:
        db.execute("UPDATE messages SET postgres_stored = 0")
        db.commit()
    archive.messages.clear()
    repaired = agent.backup("member")
    assert repaired.postgres_repaired == 1
    assert archive.messages[("member", "one@example.com", "dual1")]["raw_email"] == raw


def test_dual_copy_mode_requires_postgres_archive(tmp_path):
    agent = build_test_agent(tmp_path, lambda *args, **kwargs: {"access_token": "fresh"})
    agent.require_secondary = True
    with pytest.raises(GmailBackupError, match="PostgreSQL"):
        agent.backup("member")


def test_oauth_uses_readonly_scope_and_validates_state(tmp_path):
    store = MemoryTokenStore()
    agent = GmailBackupAgent("id", "secret", "http://localhost/callback",
                             tmp_path / "Volumes" / "USB" / "Gmail",
                             token_store=store, transport=lambda *args, **kwargs: {
                                 "access_token": "token", "refresh_token": "refresh"
                             }, volume_base=tmp_path / "Volumes", account_registry=MemoryRegistry())
    url = agent.authorization_url("member")
    assert "gmail.readonly" in url
    state = next(iter(agent._states))
    agent.transport = lambda url, **kwargs: (
        {"emailAddress": "second@example.com"} if url.endswith("/profile") else
        {"access_token": "token", "refresh_token": "refresh"}
    )
    assert agent.complete_authorization(state, "code") == "second@example.com"
    assert store.load("member", "second@example.com")["refresh_token"] == "refresh"
    with pytest.raises(GmailBackupError, match="state"):
        agent.complete_authorization(state, "code")


def test_two_accounts_are_isolated(tmp_path):
    calls = []

    def transport(url, **kwargs):
        if "oauth2.googleapis.com" in url:
            return {"access_token": "fresh"}
        if "?format=raw" in url:
            message_id = "one1" if "Bearer fresh" in kwargs["headers"]["Authorization"] else "two1"
            return {"id": message_id, "raw": base64.urlsafe_b64encode(b"mail").decode()}
        calls.append(url)
        return {"messages": []}

    agent = build_test_agent(tmp_path, transport)
    agent.account_registry.register_google_account("member", "two@example.com")
    agent.token_store.save("member", "two@example.com", {"access_token": "two"})
    assert agent.accounts("member") == ["one@example.com", "two@example.com"]
    with pytest.raises(GmailBackupError, match="Multiple"):
        agent.backup("member")
    first = agent.backup("member", "one@example.com")
    second = agent.backup("member", "two@example.com")
    assert first.target != second.target
    assert first.target.name == "one@example.com"
    assert second.target.name == "two@example.com"


def test_long_backup_refreshes_and_retries_after_401(tmp_path):
    raw = base64.urlsafe_b64encode(b"mail").decode()
    refreshes = 0
    raw_calls = 0

    def transport(url, **kwargs):
        nonlocal refreshes, raw_calls
        if "oauth2.googleapis.com" in url:
            refreshes += 1
            return {"access_token": f"fresh-{refreshes}"}
        if "?format=raw" in url:
            raw_calls += 1
            if raw_calls == 1:
                raise GmailBackupError("Google API returned HTTP 401: expired")
            assert kwargs["headers"]["Authorization"] == "Bearer fresh-2"
            return {"id": "renew1", "raw": raw}
        return {"messages": [{"id": "renew1"}]}

    agent = build_test_agent(tmp_path, transport)
    result = agent.backup("member")
    assert result.downloaded == 1
    assert refreshes == 2
    assert raw_calls == 2


def test_http_transport_retries_direct_read_timeout():
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"ok": true}'

    with patch("family_ai.gmail_backup.urllib.request.urlopen",
               side_effect=[TimeoutError("read timed out"), Response()]) as request, \
         patch("family_ai.gmail_backup.time.sleep"):
        assert _http_json("https://example.invalid") == {"ok": True}
    assert request.call_count == 2


def test_http_transport_retries_remote_disconnect():
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"ok": true}'

    with patch("family_ai.gmail_backup.urllib.request.urlopen",
               side_effect=[http.client.RemoteDisconnected(), Response()]) as request, \
         patch("family_ai.gmail_backup.time.sleep"):
        assert _http_json("https://example.invalid") == {"ok": True}
    assert request.call_count == 2


def test_backup_skips_message_that_fails_precondition(tmp_path):
    def transport(url, **_kwargs):
        if "oauth2.googleapis.com" in url:
            return {"access_token": "fresh"}
        if "?format=raw" in url:
            raise GmailBackupError(
                'Google API returned HTTP 400: {"status":"FAILED_PRECONDITION"}'
            )
        return {"messages": [{"id": "gone1"}]}

    result = build_test_agent(tmp_path, transport).backup("member")
    assert result.downloaded == 0
    assert result.skipped == 1


def test_backup_uses_unique_temporary_email_file(tmp_path):
    raw = b"Subject: Unique temp\r\n\r\nBody"

    def transport(url, **_kwargs):
        if "oauth2.googleapis.com" in url:
            return {"access_token": "fresh"}
        if "?format=raw" in url:
            return {"id": "unique1", "raw": base64.urlsafe_b64encode(raw).decode()}
        return {"messages": [{"id": "unique1"}]}

    agent = build_test_agent(tmp_path, transport)
    result = agent.backup("member")
    message = result.target / "messages" / "un" / "unique1.eml"
    assert message.read_bytes() == raw
    assert not list(message.parent.glob("*.part"))


def test_worker_shutdown_requeues_instead_of_pausing():
    class Storage:
        def __init__(self):
            self.values = {}

        def update_gmail_backup_job(self, _job_id, **values):
            self.values.update(values)

    class Agent:
        def database_count(self, *_args):
            return 0

        def mailbox_total(self, *_args):
            return 1

        def sync_database(self, *_args, **_kwargs):
            raise GmailBackupInterrupted()

    storage = Storage()
    GmailBackupWorker(Agent(), storage)._process(
        {"id": "job", "member_id": "member", "accounts": ["first@example.com"]}
    )
    assert storage.values["status"] == "queued"


def test_worker_automatically_requeues_transient_failure():
    class Storage:
        def __init__(self):
            self.values = {}

        def update_gmail_backup_job(self, _job_id, **values):
            self.values.update(values)

    class Agent:
        def database_count(self, *_args):
            return 0

        def mailbox_total(self, *_args):
            return 1

        def sync_database(self, *_args, **_kwargs):
            raise GmailBackupError("Google API network error: RemoteDisconnected")

    storage = Storage()
    worker = GmailBackupWorker(Agent(), storage)
    with patch.object(worker._stop, "wait", return_value=False):
        worker._process({"id": "job", "member_id": "member", "accounts": ["first@example.com"]})
    assert storage.values["status"] == "queued"
    assert "Auto-retry 1/8" in storage.values["error"]


def test_worker_automatically_requeues_gmail_quota_failure():
    class Storage:
        def __init__(self):
            self.values = {}

        def update_gmail_backup_job(self, _job_id, **values):
            self.values.update(values)

    class Agent:
        def database_count(self, *_args):
            return 0

        def mailbox_total(self, *_args):
            return 1

        def sync_database(self, *_args, **_kwargs):
            raise GmailBackupError("Google API returned HTTP 403: Quota exceeded")

    storage = Storage()
    worker = GmailBackupWorker(Agent(), storage)
    with patch.object(worker._stop, "wait", return_value=False):
        worker._process({"id": "job", "member_id": "member", "accounts": ["first@example.com"]})
    assert storage.values["status"] == "queued"
    assert "Auto-retry 1/8" in storage.values["error"]
