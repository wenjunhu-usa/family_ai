from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class TaskRegistry:
    """Durable Agent lifecycle and completion notifications."""

    def __init__(self, path: Path, database_url: str | None = None):
        self.path = path
        self.database_url = database_url
        if database_url:
            import psycopg
            from psycopg.rows import dict_row
            self._psycopg, self._dict_row = psycopg, dict_row
            self._setup_postgres()
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS agent_tasks (
                id TEXT PRIMARY KEY, agent TEXT NOT NULL, member_id TEXT NOT NULL,
                status TEXT NOT NULL, summary TEXT NOT NULL, error_type TEXT,
                error_summary TEXT, result_summary TEXT,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )""")
            columns = {row[1] for row in db.execute("PRAGMA table_info(agent_tasks)")}
            if "result_summary" not in columns:
                db.execute("ALTER TABLE agent_tasks ADD COLUMN result_summary TEXT")

    def _pg(self):
        return self._psycopg.connect(self.database_url, row_factory=self._dict_row, connect_timeout=3)

    def _setup_postgres(self):
        with self._pg() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS family_agent_runs (
                id TEXT PRIMARY KEY, agent TEXT NOT NULL, member_id TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('queued','running','completed','failed')),
                summary TEXT NOT NULL, error_type TEXT, error_summary TEXT, result_summary TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS family_notifications (
                id TEXT PRIMARY KEY, member_id TEXT NOT NULL, kind TEXT NOT NULL,
                title TEXT NOT NULL, body TEXT NOT NULL, related_run_id TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), read_at TIMESTAMPTZ
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS family_agent_runs_member_updated_idx ON family_agent_runs(member_id,updated_at DESC)")
            db.execute("CREATE INDEX IF NOT EXISTS family_notifications_member_created_idx ON family_notifications(member_id,created_at DESC)")
            self._prune_pg(db)

    @staticmethod
    def _prune_pg(db):
        db.execute("DELETE FROM family_agent_runs WHERE updated_at < NOW() - INTERVAL '6 months'")
        db.execute("DELETE FROM family_notifications WHERE created_at < NOW() - INTERVAL '6 months'")

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        return db

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat()

    def create(self, task_id: str, agent: str, member_id: str, summary: str):
        if self.database_url:
            with self._pg() as db:
                self._prune_pg(db)
                db.execute("""INSERT INTO family_agent_runs
                    (id,agent,member_id,status,summary) VALUES (%s,%s,%s,'queued',%s)
                    ON CONFLICT(id) DO UPDATE SET agent=EXCLUDED.agent,
                    member_id=EXCLUDED.member_id,status='queued',summary=EXCLUDED.summary,
                    error_type=NULL,error_summary=NULL,result_summary=NULL,updated_at=NOW()""",
                    (task_id, agent, member_id, summary[:2000]))
            return
        now = self._now()
        with self._connect() as db:
            db.execute("""INSERT OR REPLACE INTO agent_tasks
                (id, agent, member_id, status, summary, created_at, updated_at)
                VALUES (?, ?, ?, 'queued', ?, ?, ?)""",
                (task_id, agent, member_id, summary[:500], now, now))

    def update(self, task_id: str, status: str, error: BaseException | None = None,
               result: str | None = None):
        if self.database_url:
            with self._pg() as db:
                db.execute("""UPDATE family_agent_runs SET status=%s,error_type=%s,
                    error_summary=%s,result_summary=%s,updated_at=NOW() WHERE id=%s""", (
                    status, type(error).__name__ if error else None,
                    str(error)[:1000] if error else None,
                    result[:12000] if result else None, task_id))
            return
        with self._connect() as db:
            db.execute("""UPDATE agent_tasks SET status=?, error_type=?,
                error_summary=?, result_summary=?, updated_at=? WHERE id=?""", (
                status, type(error).__name__ if error else None,
                str(error)[:500] if error else None,
                result if result else None, self._now(), task_id))

    def latest(self, member_id: str, agent: str | None = None):
        if self.database_url:
            sql, values = "SELECT * FROM family_agent_runs WHERE member_id=%s", [member_id]
            if agent:
                sql, values = sql + " AND agent=%s", values + [agent]
            with self._pg() as db:
                row = db.execute(sql + " ORDER BY updated_at DESC LIMIT 1", values).fetchone()
            return dict(row) if row else None
        sql, values = "SELECT * FROM agent_tasks WHERE member_id=?", [member_id]
        if agent:
            sql, values = sql + " AND agent=?", values + [agent]
        with self._connect() as db:
            row = db.execute(sql + " ORDER BY updated_at DESC LIMIT 1", values).fetchone()
        return dict(row) if row else None

    def get(self, task_id: str, member_id: str):
        if self.database_url:
            with self._pg() as db:
                row = db.execute("SELECT * FROM family_agent_runs WHERE id=%s AND member_id=%s",
                                 (task_id, member_id)).fetchone()
            return dict(row) if row else None
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM agent_tasks WHERE id=? AND member_id=?",
                (task_id, member_id),
            ).fetchone()
        return dict(row) if row else None

    def active_count(self, agent: str | None = None):
        if self.database_url:
            sql, values = "SELECT COUNT(*) count FROM family_agent_runs WHERE status IN ('queued','running')", []
            if agent:
                sql, values = sql + " AND agent=%s", [agent]
            with self._pg() as db:
                return int(db.execute(sql, values).fetchone()["count"])
        sql, values = "SELECT COUNT(*) FROM agent_tasks WHERE status IN ('queued','running')", []
        if agent:
            sql, values = sql + " AND agent=?", [agent]
        with self._connect() as db:
            return int(db.execute(sql, values).fetchone()[0])

    def notify(self, member_id: str, kind: str, title: str, body: str,
               related_run_id: str | None = None) -> str:
        notification_id = str(uuid4())
        if not self.database_url:
            return notification_id
        with self._pg() as db:
            self._prune_pg(db)
            db.execute("""INSERT INTO family_notifications
                (id,member_id,kind,title,body,related_run_id) VALUES (%s,%s,%s,%s,%s,%s)""",
                (notification_id, member_id, kind, title[:300], body[:12000], related_run_id))
        return notification_id

    def unread_notifications(self, member_id: str, limit: int = 20) -> list[dict]:
        if not self.database_url:
            return []
        with self._pg() as db:
            rows = db.execute("""SELECT * FROM family_notifications
                WHERE member_id=%s AND read_at IS NULL ORDER BY created_at LIMIT %s""",
                (member_id, limit)).fetchall()
        return [dict(row) for row in rows]

    def mark_notification_read(self, notification_id: str, member_id: str) -> bool:
        if not self.database_url:
            return False
        with self._pg() as db:
            cursor = db.execute("""UPDATE family_notifications SET read_at=NOW()
                WHERE id=%s AND member_id=%s AND read_at IS NULL""",
                (notification_id, member_id))
        return cursor.rowcount == 1
