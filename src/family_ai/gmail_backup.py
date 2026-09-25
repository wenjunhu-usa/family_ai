import base64
import hashlib
import http.client
import json
import logging
from email.message import EmailMessage
import secrets
import sqlite3
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


logger = logging.getLogger(__name__)

GMAIL_READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
GMAIL_SEND_SCOPE = "https://www.googleapis.com/auth/gmail.send"
CALENDAR_READONLY_SCOPE = "https://www.googleapis.com/auth/calendar.readonly"
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GMAIL_API = "https://gmail.googleapis.com/gmail/v1/users/me"


class GmailBackupError(RuntimeError):
    pass


class GmailMessageUnavailable(Exception):
    """A listed message disappeared or became unreadable during traversal."""


class GmailBackupPaused(Exception):
    pass


class GmailBackupInterrupted(Exception):
    """Worker shutdown/rebuild interruption; requeue the durable job."""


class GmailBackupCancelled(Exception):
    pass


class MacOSKeychainTokenStore:
    """Keep OAuth refresh tokens out of the backup disk and project files."""

    service = "Family AI Gmail Backup"

    @staticmethod
    def _account(member_id: str, email_address: str) -> str:
        return f"{member_id}:{email_address.casefold()}"

    def load(self, member_id: str, email_address: str) -> dict | None:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", self.service,
             "-a", self._account(member_id, email_address), "-w"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise GmailBackupError("The Gmail token in macOS Keychain is invalid") from exc

    def save(self, member_id: str, email_address: str, token: dict) -> None:
        subprocess.run(
            [
                "security", "add-generic-password", "-U", "-s", self.service,
                "-a", self._account(member_id, email_address),
                "-w", json.dumps(token, separators=(",", ":")),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

    def delete(self, member_id: str, email_address: str) -> None:
        subprocess.run(
            ["security", "delete-generic-password", "-s", self.service,
             "-a", self._account(member_id, email_address)],
            capture_output=True,
            text=True,
            check=False,
        )


def _http_json(url: str, *, method: str = "GET", headers: dict | None = None,
               form: dict | None = None, json_body: dict | None = None) -> dict:
    body = None
    request_headers = dict(headers or {})
    if form is not None:
        body = urllib.parse.urlencode(form).encode()
        request_headers["Content-Type"] = "application/x-www-form-urlencoded"
    elif json_body is not None:
        body = json.dumps(json_body).encode()
        request_headers["Content-Type"] = "application/json"
    for attempt in range(5):
        request = urllib.request.Request(url, data=body, headers=request_headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            quota_limited = exc.code == 403 and (
                "Quota exceeded" in detail
                or "rateLimitExceeded" in detail
                or "userRateLimitExceeded" in detail
            )
            if (exc.code in {429, 500, 502, 503, 504} or quota_limited) and attempt < 4:
                time.sleep((15 * (2 ** attempt)) if quota_limited else (2 ** attempt))
                continue
            raise GmailBackupError(f"Google API returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            if attempt < 4:
                time.sleep(2 ** attempt)
                continue
            raise GmailBackupError(f"Google API network error: {type(exc.reason).__name__}") from exc
        except (TimeoutError, http.client.RemoteDisconnected, ConnectionResetError) as exc:
            # urlopen may surface a socket read timeout directly instead of
            # wrapping it in URLError. Treat both forms as transient.
            if attempt < 4:
                time.sleep(2 ** attempt)
                continue
            raise GmailBackupError(f"Google API network error: {type(exc).__name__}") from exc
    raise GmailBackupError("Google API retry limit exceeded")


@dataclass(frozen=True)
class BackupResult:
    account: str
    downloaded: int
    skipped: int
    postgres_repaired: int
    total_seen: int
    target: Path


@dataclass(frozen=True)
class DatabaseSyncResult:
    account: str
    downloaded: int
    skipped: int
    postgres_repaired: int
    total_seen: int


@dataclass(frozen=True)
class DiskExportResult:
    account: str
    exported: int
    skipped: int
    target: Path


class GmailBackupAgent:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        backup_root: Path,
        token_store=None,
        transport: Callable[..., dict] = _http_json,
        volume_base: Path = Path("/Volumes"),
        secondary_archive=None,
        require_secondary: bool = False,
        account_registry=None,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.backup_root = backup_root
        self.token_store = token_store or MacOSKeychainTokenStore()
        self.transport = transport
        self.volume_base = volume_base
        self.secondary_archive = secondary_archive
        self.require_secondary = require_secondary
        self.account_registry = account_registry
        self._states: dict[str, tuple[str, str, str | None]] = {}

    @property
    def configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def authorization_url(self, member_id: str, account_hint: str | None = None,
                          allow_send: bool = False) -> str:
        if not self.configured:
            raise GmailBackupError("Google OAuth client ID/secret are not configured")
        state = secrets.token_urlsafe(24)
        verifier = secrets.token_urlsafe(48)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
        self._states[state] = (member_id, verifier, account_hint)
        scopes = [GMAIL_READONLY_SCOPE, CALENDAR_READONLY_SCOPE]
        if allow_send:
            scopes.append(GMAIL_SEND_SCOPE)
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": " ".join(scopes),
            "access_type": "offline",
            "prompt": "consent select_account",
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        if account_hint:
            params["login_hint"] = account_hint
        return GOOGLE_AUTH_URL + "?" + urllib.parse.urlencode(params)

    def send_message(self, member_id: str, email_address: str, recipient: str,
                     subject: str, text_body: str, html_body: str | None = None) -> str:
        """Send one explicitly addressed message; requires the gmail.send OAuth scope."""
        account = self._select_account(member_id, email_address)
        saved = self.token_store.load(member_id, account) or {}
        granted = set(str(saved.get("scope", "")).split())
        if GMAIL_SEND_SCOPE not in granted:
            raise GmailBackupError(
                f"{account} has not granted Gmail send permission; reconnect it with send enabled"
            )
        message = EmailMessage()
        message["To"] = recipient
        message["From"] = account
        message["Subject"] = subject
        message.set_content(text_body)
        if html_body:
            message.add_alternative(html_body, subtype="html")
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode().rstrip("=")
        token = self._access_token(member_id, account)
        result = self.transport(
            GMAIL_API + "/messages/send", method="POST",
            headers={"Authorization": f"Bearer {token}"}, json_body={"raw": raw},
        )
        message_id = str(result.get("id", ""))
        if not message_id:
            raise GmailBackupError("Gmail did not confirm the sent message")
        return message_id

    def complete_authorization(self, state: str, code: str) -> str:
        pending = self._states.pop(state, None)
        if pending is None:
            raise GmailBackupError("OAuth state is invalid or expired")
        member_id, verifier, _account_hint = pending
        token = self.transport(GOOGLE_TOKEN_URL, method="POST", form={
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "code_verifier": verifier,
            "grant_type": "authorization_code",
            "redirect_uri": self.redirect_uri,
        })
        token["obtained_at"] = datetime.now(timezone.utc).timestamp()
        profile = self.transport(
            GMAIL_API + "/profile", headers={"Authorization": f"Bearer {token['access_token']}"}
        )
        email_address = str(profile.get("emailAddress", "")).strip().casefold()
        if not email_address or "@" not in email_address:
            raise GmailBackupError("Google did not return the authorized Gmail address")
        self.token_store.save(member_id, email_address, token)
        if self.account_registry is not None:
            self.account_registry.register_google_account(member_id, email_address)
        return email_address

    def accounts(self, member_id: str) -> list[str]:
        if self.account_registry is None:
            return []
        return self.account_registry.list_google_accounts(member_id)

    def _select_account(self, member_id: str, email_address: str | None) -> str:
        accounts = self.accounts(member_id)
        if email_address:
            selected = email_address.casefold()
            if selected not in accounts:
                raise GmailBackupError(f"Gmail account is not connected: {selected}")
            return selected
        if not accounts:
            raise GmailBackupError("Gmail is not connected for this family member")
        if len(accounts) > 1:
            raise GmailBackupError("Multiple Gmail accounts are connected; specify one account or ask to back up all Gmail accounts")
        return accounts[0]

    def _access_token(self, member_id: str, email_address: str) -> str:
        token = self.token_store.load(member_id, email_address)
        if not token:
            raise GmailBackupError("Gmail is not connected for this family member")
        if token.get("refresh_token"):
            refreshed = self.transport(GOOGLE_TOKEN_URL, method="POST", form={
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "refresh_token": token["refresh_token"],
                "grant_type": "refresh_token",
            })
            token.update(refreshed)
            token["obtained_at"] = datetime.now(timezone.utc).timestamp()
            self.token_store.save(member_id, email_address, token)
        access_token = token.get("access_token")
        if not access_token:
            raise GmailBackupError("Google did not provide an access token")
        return str(access_token)

    def _member_root(self, member_id: str, email_address: str) -> Path:
        try:
            relative = self.backup_root.resolve(strict=False).relative_to(
                self.volume_base.resolve(strict=False)
            )
        except ValueError as exc:
            raise GmailBackupError("Backup target must be on the configured removable-volume base") from exc
        if not self.backup_root.is_absolute() or len(relative.parts) < 2:
            raise GmailBackupError("Backup target must be an absolute path on a mounted volume")
        volume_root = self.volume_base / relative.parts[0]
        if not volume_root.is_dir():
            raise GmailBackupError(f"Backup drive is not mounted: {volume_root}")
        safe_member = "".join(c for c in member_id if c.isalnum() or c in "-_")
        if safe_member != member_id or not safe_member:
            raise GmailBackupError("Invalid family member ID")
        safe_account = "".join(c for c in email_address.casefold() if c.isalnum() or c in "@._-+")
        if safe_account != email_address.casefold() or "@" not in safe_account:
            raise GmailBackupError("Invalid Gmail account address")
        return self.backup_root / safe_member / safe_account

    @staticmethod
    def _setup_manifest(path: Path) -> sqlite3.Connection:
        db = sqlite3.connect(path)
        db.execute("PRAGMA journal_mode=DELETE")
        db.execute("""CREATE TABLE IF NOT EXISTS messages (
            message_id TEXT PRIMARY KEY, thread_id TEXT, internal_date TEXT,
            labels_json TEXT NOT NULL, eml_path TEXT NOT NULL, sha256 TEXT NOT NULL,
            backed_up_at TEXT NOT NULL, postgres_stored INTEGER NOT NULL DEFAULT 0
        )""")
        columns = {row[1] for row in db.execute("PRAGMA table_info(messages)")}
        if "postgres_stored" not in columns:
            db.execute("ALTER TABLE messages ADD COLUMN postgres_stored INTEGER NOT NULL DEFAULT 0")
        db.commit()
        return db

    def mailbox_total(self, member_id: str, email_address: str) -> int:
        account = self._select_account(member_id, email_address)
        token = self._access_token(member_id, account)
        profile = self.transport(
            GMAIL_API + "/profile", headers={"Authorization": f"Bearer {token}"}
        )
        return int(profile.get("messagesTotal", 0))

    def completed_count(self, member_id: str, email_address: str) -> int:
        root = self._member_root(member_id, email_address)
        manifest = root / "manifest.sqlite"
        if not manifest.exists():
            return 0
        with sqlite3.connect(manifest) as db:
            row = db.execute("SELECT COUNT(*) FROM messages").fetchone()
        return int(row[0])

    def database_count(self, member_id: str, email_address: str) -> int:
        if self.secondary_archive is None:
            return 0
        return self.secondary_archive.email_backup_count(member_id, email_address)

    def sync_database(self, member_id: str, email_address: str | None = None,
                      progress: Callable[[int, int, int], None] | None = None,
                      control: Callable[[], str | None] | None = None) -> DatabaseSyncResult:
        """Incrementally archive Gmail into PostgreSQL without touching disk."""
        if self.secondary_archive is None:
            raise GmailBackupError("PostgreSQL email archive is not configured")
        account = self._select_account(member_id, email_address)
        headers = {"Authorization": f"Bearer {self._access_token(member_id, account)}"}
        archived_ids = self.secondary_archive.email_backup_ids(member_id, account)
        latest_internal_date = None
        latest_reader = getattr(
            self.secondary_archive, "email_backup_latest_internal_date", None
        )
        if latest_reader is not None:
            latest_internal_date = latest_reader(member_id, account)
        downloaded = skipped = total = 0
        page_token = None
        while True:
            params = {"maxResults": 500, "includeSpamTrash": "true"}
            if latest_internal_date is not None:
                # Gmail accepts epoch seconds in `after:`. Re-scan the most
                # recent 30 days so a previous interruption or delayed mail is
                # repaired; archived IDs make the overlap idempotent.
                overlap_seconds = 30 * 24 * 60 * 60
                after_seconds = max(0, latest_internal_date // 1000 - overlap_seconds)
                params["q"] = f"after:{after_seconds}"
            if page_token:
                params["pageToken"] = page_token
            listing = self.transport(
                GMAIL_API + "/messages?" + urllib.parse.urlencode(params), headers=headers
            )
            for item in listing.get("messages", []):
                requested = control() if control else None
                if requested == "interrupt":
                    raise GmailBackupInterrupted()
                if requested == "pause":
                    raise GmailBackupPaused()
                if requested == "cancel":
                    raise GmailBackupCancelled()
                total += 1
                message_id = str(item["id"])
                if message_id in archived_ids:
                    skipped += 1
                else:
                    message = self.transport(
                        f"{GMAIL_API}/messages/{message_id}?format=raw", headers=headers
                    )
                    raw = base64.urlsafe_b64decode(str(message["raw"]) + "===")
                    digest = hashlib.sha256(raw).hexdigest()
                    self.secondary_archive.store_email_backup(
                        member_id, account, message, raw, digest
                    )
                    if not self.secondary_archive.has_email_backup(
                        member_id, account, message_id, digest
                    ):
                        raise GmailBackupError("PostgreSQL verification failed after email write")
                    archived_ids.add(message_id)
                    downloaded += 1
                if progress:
                    progress(total, downloaded, 0)
            page_token = listing.get("nextPageToken")
            if not page_token:
                break
        return DatabaseSyncResult(account, downloaded, skipped, 0, total)

    def export_database_to_disk(self, member_id: str, email_address: str | None = None,
                                progress: Callable[[int, int], None] | None = None) -> DiskExportResult:
        """Export PostgreSQL email rows to removable disk and verify both hashes."""
        if self.secondary_archive is None:
            raise GmailBackupError("PostgreSQL email archive is not configured")
        account = self._select_account(member_id, email_address)
        root = self._member_root(member_id, account)
        (root / "messages").mkdir(parents=True, exist_ok=True)
        manifest = self._setup_manifest(root / "manifest.sqlite")
        exported = skipped = seen = 0
        try:
            for row in self.secondary_archive.iter_email_backups(member_id, account):
                seen += 1
                message_id = str(row["gmail_message_id"])
                raw = bytes(row["raw_email"])
                digest = hashlib.sha256(raw).hexdigest()
                if digest != str(row["sha256"]):
                    raise GmailBackupError(f"PostgreSQL email hash mismatch: {message_id}")
                relative = Path("messages") / message_id[:2] / f"{message_id}.eml"
                destination = root / relative
                existing = manifest.execute(
                    "SELECT sha256 FROM messages WHERE message_id=?", (message_id,)
                ).fetchone()
                if existing and existing[0] == digest and destination.is_file():
                    skipped += 1
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_name(
                        f"{destination.name}.{threading.get_ident()}.{secrets.token_hex(4)}.part"
                    )
                    try:
                        temporary.write_bytes(raw)
                        if hashlib.sha256(temporary.read_bytes()).hexdigest() != digest:
                            raise GmailBackupError(f"Disk verification failed: {message_id}")
                        temporary.replace(destination)
                    finally:
                        temporary.unlink(missing_ok=True)
                    manifest.execute("""INSERT OR REPLACE INTO messages
                        (message_id, thread_id, internal_date, labels_json, eml_path,
                         sha256, backed_up_at, postgres_stored)
                        VALUES (?, ?, ?, ?, ?, ?, ?, 1)""", (
                        message_id, row.get("thread_id"), row.get("internal_date"),
                        json.dumps(row.get("labels") or []), str(relative), digest,
                        datetime.now(timezone.utc).isoformat(),
                    ))
                    manifest.commit()
                    exported += 1
                if progress:
                    progress(seen, exported)
        finally:
            manifest.close()
        return DiskExportResult(account, exported, skipped, root)

    def backup(self, member_id: str, email_address: str | None = None,
               progress: Callable[[int, int, int], None] | None = None,
               control: Callable[[], str | None] | None = None) -> BackupResult:
        if self.require_secondary and self.secondary_archive is None:
            raise GmailBackupError("PostgreSQL is required for dual-copy Gmail backup but is not configured")
        account = self._select_account(member_id, email_address)
        token = self._access_token(member_id, account)
        root = self._member_root(member_id, account)
        messages_dir = root / "messages"
        messages_dir.mkdir(parents=True, exist_ok=True)
        db = self._setup_manifest(root / "manifest.sqlite")
        headers = {"Authorization": f"Bearer {token}"}

        def gmail_request(url: str) -> dict:
            try:
                return self.transport(url, headers=headers)
            except GmailBackupError as exc:
                detail = str(exc)
                if ("HTTP 400" in detail and "FAILED_PRECONDITION" in detail
                        and "/messages/" in url):
                    raise GmailMessageUnavailable(url) from exc
                if "HTTP 401" not in detail:
                    raise
                # Access tokens normally expire after about an hour. A large
                # mailbox can run much longer, so refresh and retry the exact
                # request once without losing manifest progress.
                headers["Authorization"] = (
                    f"Bearer {self._access_token(member_id, account)}"
                )
                return self.transport(url, headers=headers)

        downloaded = skipped = postgres_repaired = total = 0
        page_token = None
        try:
            while True:
                params = {"maxResults": 500, "includeSpamTrash": "true"}
                if page_token:
                    params["pageToken"] = page_token
                listing = gmail_request(GMAIL_API + "/messages?" + urllib.parse.urlencode(params))
                for item in listing.get("messages", []):
                    requested_control = control() if control else None
                    if requested_control == "interrupt":
                        raise GmailBackupInterrupted()
                    if requested_control == "pause":
                        raise GmailBackupPaused()
                    if requested_control == "cancel":
                        raise GmailBackupCancelled()
                    total += 1
                    message_id = str(item["id"])
                    existing = db.execute(
                        "SELECT eml_path, sha256, postgres_stored FROM messages WHERE message_id = ?",
                        (message_id,),
                    ).fetchone()
                    if existing:
                        existing_file = root / existing[0]
                        if self.secondary_archive is not None and not existing[2]:
                            if not existing_file.is_file():
                                db.execute("DELETE FROM messages WHERE message_id = ?", (message_id,))
                                db.commit()
                            else:
                                raw = existing_file.read_bytes()
                                try:
                                    metadata = gmail_request(
                                        f"{GMAIL_API}/messages/{message_id}?format=metadata"
                                    )
                                except GmailMessageUnavailable:
                                    skipped += 1
                                    if progress:
                                        progress(total, downloaded, postgres_repaired)
                                    continue
                                self.secondary_archive.store_email_backup(
                                    member_id, account, metadata, raw, existing[1]
                                )
                                db.execute(
                                    "UPDATE messages SET postgres_stored = 1 WHERE message_id = ?",
                                    (message_id,),
                                )
                                db.commit()
                                postgres_repaired += 1
                                skipped += 1
                                if progress:
                                    progress(total, downloaded, postgres_repaired)
                                continue
                        else:
                            skipped += 1
                            if progress:
                                progress(total, downloaded, postgres_repaired)
                            continue
                    try:
                        message = gmail_request(f"{GMAIL_API}/messages/{message_id}?format=raw")
                    except GmailMessageUnavailable:
                        skipped += 1
                        if progress:
                            progress(total, downloaded, postgres_repaired)
                        continue
                    raw = base64.urlsafe_b64decode(str(message["raw"]) + "===")
                    relative = Path("messages") / message_id[:2] / f"{message_id}.eml"
                    destination = root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    # Agent rebuilds can briefly overlap workers. A fixed
                    # `.eml.part` name lets one worker rename another's temp
                    # file on exFAT. Use a unique sibling and atomically commit.
                    temporary = destination.with_name(
                        f"{destination.name}.{threading.get_ident()}.{secrets.token_hex(4)}.part"
                    )
                    try:
                        temporary.write_bytes(raw)
                        temporary.replace(destination)
                    finally:
                        temporary.unlink(missing_ok=True)
                    digest = hashlib.sha256(raw).hexdigest()
                    postgres_stored = 0
                    if self.secondary_archive is not None:
                        self.secondary_archive.store_email_backup(member_id, account, message, raw, digest)
                        postgres_stored = 1
                    db.execute(
                        """INSERT OR REPLACE INTO messages
                        (message_id, thread_id, internal_date, labels_json, eml_path, sha256,
                         backed_up_at, postgres_stored) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (message_id, message.get("threadId"), message.get("internalDate"),
                         json.dumps(message.get("labelIds", [])), str(relative), digest,
                         datetime.now(timezone.utc).isoformat(), postgres_stored),
                    )
                    db.commit()
                    downloaded += 1
                    if progress:
                        progress(total, downloaded, postgres_repaired)
                page_token = listing.get("nextPageToken")
                if not page_token:
                    break
        finally:
            db.close()
        report = {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "downloaded": downloaded, "skipped": skipped,
            "postgres_repaired": postgres_repaired, "total_seen": total,
        }
        (root / "last-backup.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return BackupResult(account, downloaded, skipped, postgres_repaired, total, root)

    def status(self, member_id: str, email_address: str | None = None) -> dict:
        account = self._select_account(member_id, email_address)
        root = self._member_root(member_id, account)
        relative = self.backup_root.resolve(strict=False).relative_to(self.volume_base.resolve(strict=False))
        report = root / "last-backup.json"
        return {
            "configured": self.configured,
            "account": account,
            "connected": self.token_store.load(member_id, account) is not None,
            "drive_mounted": (self.volume_base / relative.parts[0]).is_dir(),
            "target": str(root),
            "last_backup": json.loads(report.read_text()) if report.exists() else None,
        }

    def disconnect(self, member_id: str, email_address: str | None = None) -> str:
        account = self._select_account(member_id, email_address)
        self.token_store.delete(member_id, account)
        if self.account_registry is not None:
            self.account_registry.remove_google_account(member_id, account)
        return account


class GmailBackupWorker:
    """Durable single-worker queue that runs independently of the model task queue."""

    def __init__(self, agent: GmailBackupAgent, storage, poll_seconds: float = 2.0):
        self.agent = agent
        self.storage = storage
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._retry_attempts: dict[str, int] = {}
        self.max_auto_retries = 8

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        detail = f"{type(exc).__name__}: {exc}"
        markers = (
            "Timeout", "timed out", "RemoteDisconnected", "ConnectionReset",
            "FileNotFoundError",
            "network error", "Network is down", "Quota exceeded", "rateLimitExceeded",
            "userRateLimitExceeded", "HTTP 429", "HTTP 500",
            "HTTP 502", "HTTP 503", "HTTP 504", "FAILED_PRECONDITION",
        )
        return any(marker in detail for marker in markers)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.storage.recover_gmail_backup_jobs()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="gmail-backup-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=10)

    def submit(self, member_id: str, accounts: list[str]) -> str:
        current = self.storage.get_gmail_backup_job(member_id)
        if current and current["status"] in {"queued", "running", "paused"}:
            raise GmailBackupError(f"A Gmail backup job is already {current['status']}: {current['id'][:8]}")
        job_id = self.storage.create_gmail_backup_job(member_id, accounts)
        self.start()
        self._wake.set()
        return job_id

    def control(self, member_id: str, action: str) -> bool:
        changed = self.storage.control_gmail_backup_job(member_id, action)
        if changed and action == "resume":
            self.start()
        self._wake.set()
        return changed

    def status(self, member_id: str):
        return self.storage.get_gmail_backup_job(member_id)

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _run(self) -> None:
        idle_polls = 0
        consecutive_failures = 0
        while not self._stop.is_set():
            try:
                job = self.storage.claim_gmail_backup_job()
                if job:
                    idle_polls = 0
                    consecutive_failures = 0
                    self._process(job)
                    continue
                awakened = self._wake.wait(self.poll_seconds)
                self._wake.clear()
                idle_polls = 0 if awakened else idle_polls + 1
                if idle_polls >= 2:
                    return
            except Exception:
                # PostgreSQL can disappear briefly when the network changes.
                # Bound retry noise and let the daily scheduler wake us later.
                logger.exception("Gmail backup worker cycle failed; retrying")
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    logger.error("Gmail backup worker paused after repeated storage failures")
                    return
                self._wake.wait(self.poll_seconds)
                self._wake.clear()
                try:
                    self.storage.recover_gmail_backup_jobs()
                except Exception:
                    logger.exception("Gmail backup job recovery failed; retrying")

    def _process(self, job: dict) -> None:
        from .agent_activity import agent_activity
        with agent_activity("builtin-gmail-backup"):
            self._process_active(job)

    def _process_active(self, job: dict) -> None:
        job_id = str(job["id"])
        member_id = str(job["member_id"])
        accounts = [str(item) for item in job["accounts"]]
        try:
            initial_counts = {
                account: self.agent.database_count(member_id, account) for account in accounts
            }
            totals = {account: self.agent.mailbox_total(member_id, account) for account in accounts}
            total_messages = sum(totals.values())
            completed_base = 0
            self.storage.update_gmail_backup_job(
                job_id, total_messages=total_messages,
                completed_messages=sum(initial_counts.values()), error=None,
            )
            repaired_total = int(job.get("postgres_repaired", 0))
            for account in accounts:
                self.storage.update_gmail_backup_job(job_id, current_account=account)
                last_update = [0.0]
                last_control = [0.0, None]

                def control():
                    now = time.monotonic()
                    if now - last_control[0] < 1.0:
                        return last_control[1]
                    current = self.storage.get_gmail_backup_job(member_id, job_id)
                    last_control[0] = now
                    if self._stop.is_set():
                        last_control[1] = "interrupt"
                    elif current and current["pause_requested"]:
                        last_control[1] = "pause"
                    elif current and current["cancel_requested"]:
                        last_control[1] = "cancel"
                    else:
                        last_control[1] = None
                    return last_control[1]

                def progress(seen: int, _downloaded: int, repaired: int):
                    now = time.monotonic()
                    if now - last_update[0] < 2.0 and seen % 25:
                        return
                    account_done = max(initial_counts[account], seen)
                    self.storage.update_gmail_backup_job(
                        job_id,
                        completed_messages=min(total_messages, completed_base + account_done),
                        postgres_repaired=repaired_total + repaired,
                    )
                    last_update[0] = now

                result = self.agent.sync_database(
                    member_id, account, progress=progress, control=control
                )
                completed_base += result.total_seen
                repaired_total += result.postgres_repaired
                self.storage.update_gmail_backup_job(
                    job_id, completed_messages=min(total_messages, completed_base),
                    postgres_repaired=repaired_total,
                )
            self.storage.update_gmail_backup_job(
                job_id, status="completed", current_account=None,
                completed_messages=total_messages, pause_requested=False,
            )
            self._retry_attempts.pop(job_id, None)
        except GmailBackupPaused:
            self.storage.update_gmail_backup_job(job_id, status="paused", pause_requested=False)
        except GmailBackupInterrupted:
            self.storage.update_gmail_backup_job(
                job_id, status="queued", pause_requested=False, cancel_requested=False
            )
        except GmailBackupCancelled:
            self.storage.update_gmail_backup_job(job_id, status="cancelled", cancel_requested=False)
        except Exception as exc:
            attempt = self._retry_attempts.get(job_id, 0) + 1
            if self._retryable(exc) and attempt <= self.max_auto_retries:
                self._retry_attempts[job_id] = attempt
                delay = min(2 ** attempt, 300)
                try:
                    self.storage.update_gmail_backup_job(
                        job_id, status="queued",
                        error=(f"Auto-retry {attempt}/{self.max_auto_retries} in {delay}s: "
                               f"{type(exc).__name__}: {str(exc)[:220]}"),
                    )
                except Exception:
                    logger.exception("Could not requeue Gmail backup job %s", job_id)
                    raise
                self._stop.wait(delay)
                return
            self._retry_attempts.pop(job_id, None)
            try:
                self.storage.update_gmail_backup_job(
                    job_id, status="failed", error=f"{type(exc).__name__}: {str(exc)[:300]}"
                )
            except Exception:
                logger.exception("Could not persist Gmail backup failure for job %s", job_id)
                raise


class GmailDailyScheduler:
    """Lightweight daily trigger; the on-demand worker performs the sync."""

    def __init__(self, agent: GmailBackupAgent, worker: GmailBackupWorker,
                 state_path: Path, member_id: str = "member", hour: int = 3,
                 poll_seconds: float = 900):
        self.agent, self.worker = agent, worker
        self.state_path, self.member_id = state_path, member_id
        self.hour = max(0, min(int(hour), 23))
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(state_path) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS daily_sync (
                member_id TEXT PRIMARY KEY, last_submitted_date TEXT NOT NULL)""")

    def run_due(self, now: datetime | None = None) -> str | None:
        now = now or datetime.now().astimezone()
        if now.hour < self.hour:
            return None
        today = now.date().isoformat()
        with sqlite3.connect(self.state_path) as db:
            row = db.execute("SELECT last_submitted_date FROM daily_sync WHERE member_id=?",
                             (self.member_id,)).fetchone()
            if row and row[0] == today:
                return None
        accounts = self.agent.accounts(self.member_id)
        if not accounts:
            return None
        job_id = self.worker.submit(self.member_id, accounts)
        with sqlite3.connect(self.state_path) as db:
            db.execute("""INSERT INTO daily_sync(member_id,last_submitted_date) VALUES(?,?)
                ON CONFLICT(member_id) DO UPDATE SET
                last_submitted_date=excluded.last_submitted_date""",
                (self.member_id, today))
        return job_id

    def _run(self):
        while not self._stop.is_set():
            try:
                # Recover queued work periodically without leaving a polling
                # worker resident when there is nothing to do.
                self.worker.start()
                self.run_due()
            except Exception:
                logger.exception("daily Gmail database sync scheduling failed")
            self._stop.wait(self.poll_seconds)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run,
            name="gmail-daily-scheduler", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
