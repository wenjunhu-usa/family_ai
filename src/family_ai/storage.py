import sqlite3
import json
from threading import Lock
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class FamilyStorage:
    def __init__(self, path: Path):
        self.path = path
        self._setup()

    def healthcheck(self) -> dict:
        with self.connect() as db:
            db.execute("SELECT 1").fetchone()
        return {"database": "sqlite", "reachable": True}

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _setup(self):
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    member_id TEXT NOT NULL,
                    scope TEXT NOT NULL CHECK(scope IN ('private', 'family')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS todos (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    completed INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS agent_registry (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    purpose TEXT NOT NULL,
                    permissions TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('active', 'pending', 'rejected')),
                    requested_by TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS codex_requests (
                    id TEXT PRIMARY KEY,
                    member_id TEXT NOT NULL,
                    task TEXT NOT NULL,
                    workspace TEXT NOT NULL CHECK(workspace IN ('isolated', 'family_project')),
                    reason TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending', 'approved', 'rejected')),
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS google_accounts (
                    member_id TEXT NOT NULL,
                    email_address TEXT NOT NULL,
                    connected_at TEXT NOT NULL,
                    PRIMARY KEY (member_id, email_address)
                );
                CREATE TABLE IF NOT EXISTS gmail_backup_jobs (
                    id TEXT PRIMARY KEY,
                    member_id TEXT NOT NULL,
                    accounts TEXT NOT NULL,
                    status TEXT NOT NULL,
                    current_account TEXT,
                    total_messages INTEGER NOT NULL DEFAULT 0,
                    completed_messages INTEGER NOT NULL DEFAULT 0,
                    postgres_repaired INTEGER NOT NULL DEFAULT 0,
                    pause_requested INTEGER NOT NULL DEFAULT 0,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            db.executemany(
                """INSERT INTO agent_registry
                (id, name, purpose, permissions, status, requested_by, created_at)
                VALUES (?, ?, ?, ?, 'active', 'system', ?)
                ON CONFLICT(id) DO UPDATE SET name=excluded.name,
                purpose=excluded.purpose, permissions=excluded.permissions,
                status='active', requested_by='system'""",
                [
                    ("builtin-main", "Main Family Agent", "理解、监管和分配家庭任务", "[]", self.now()),
                    ("builtin-voice-turn", "Voice Turn Agent", "判断语音是否面向助手并交给主 Agent", "[]", self.now()),
                    ("builtin-camera", "Camera Agent", "按主 Agent 请求读取指定摄像头窗口并进行本地视觉分析", '["camera_window_read"]', self.now()),
                    ("builtin-search", "Search Agent", "多策略公开网页搜索", '["network_public"]', self.now()),
                    ("builtin-weather", "Weather Agent", "获取实时天气并验证结构化预报", '["network_public"]', self.now()),
                    ("builtin-memory", "Memory Agent", "按成员权限读取记忆，写入后重新读取验证", '["family_memory_read", "database_write"]', self.now()),
                    ("builtin-tasks", "Tasks Agent", "管理个人和家庭待办，写入与完成后验证", '["database_read", "database_write"]', self.now()),
                    ("builtin-diagnostics", "Diagnostics Agent", "隔离检查数据库、硬盘、Gmail、Ollama 与 Codex 健康状态", '["database_read", "files_read"]', self.now()),
                    ("builtin-codex", "Codex Specialist Agent", "经批准处理复杂问题和只读工程分析", '["cloud_codex", "files_read"]', self.now()),
                    ("builtin-curator", "Model & Capability Curator Agent", "每月检查适合本机的新模型与能力工具；先测试、可回滚、按需提议 Codex", '["network_public", "ollama_model_manage"]', self.now()),
                    ("builtin-financial-video", "Financial Video Agent", "制作金融教育短视频审核稿；默认 Ollama，用户可明确指定 Codex", '["files_read", "files_write", "cloud_codex"]', self.now()),
                    ("builtin-gmail-backup", "Gmail & Calendar Agent", "每天只读增量同步 Gmail 到 PostgreSQL；用户明确要求时从 PostgreSQL 导出硬盘；只读查询 Calendar", '["email", "calendar", "database_write", "files_write"]', self.now()),
                ],
            )

    @staticmethod
    def now():
        return datetime.now(timezone.utc).isoformat()

    def add_memory(self, member_id: str, content: str, scope: str = "private"):
        memory_id = str(uuid4())
        with self.connect() as db:
            db.execute(
                "INSERT INTO memories VALUES (?, ?, ?, ?, ?)",
                (memory_id, member_id, scope, content, self.now()),
            )
        return memory_id

    def list_memories(self, member_id: str, limit: int = 20):
        with self.connect() as db:
            rows = db.execute(
                """SELECT id, member_id, scope, content, created_at FROM memories
                WHERE member_id = ? OR scope = 'family'
                ORDER BY created_at DESC LIMIT ?""",
                (member_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def add_todo(self, owner_id: str, content: str):
        todo_id = str(uuid4())
        with self.connect() as db:
            db.execute(
                "INSERT INTO todos VALUES (?, ?, ?, 0, ?)",
                (todo_id, owner_id, content, self.now()),
            )
        return todo_id

    def list_todos(self, owner_id: str):
        with self.connect() as db:
            rows = db.execute(
                """SELECT id, owner_id, content, completed, created_at FROM todos
                WHERE owner_id IN (?, 'family') ORDER BY completed, created_at DESC""",
                (owner_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def complete_todo(self, owner_id: str, todo_id: str):
        with self.connect() as db:
            cursor = db.execute(
                "UPDATE todos SET completed = 1 WHERE id LIKE ? AND owner_id IN (?, 'family')",
                (f"{todo_id}%", owner_id),
            )
        return cursor.rowcount == 1

    def propose_agent(self, member_id: str, name: str, purpose: str, permissions: list[str]):
        proposal_id = str(uuid4())
        with self.connect() as db:
            db.execute(
                "INSERT INTO agent_registry VALUES (?, ?, ?, ?, 'pending', ?, ?)",
                (proposal_id, name, purpose, json.dumps(permissions), member_id, self.now()),
            )
        return proposal_id

    def list_agents(self):
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM agent_registry ORDER BY status, created_at"
            ).fetchall()
        result = [dict(row) for row in rows]
        for item in result:
            item["permissions"] = json.loads(item["permissions"])
        return result

    def decide_agent(self, proposal_id: str, approved: bool):
        status = "active" if approved else "rejected"
        with self.connect() as db:
            cursor = db.execute(
                "UPDATE agent_registry SET status = ? WHERE id LIKE ? AND status = 'pending'",
                (status, f"{proposal_id}%"),
            )
        return cursor.rowcount == 1

    def propose_codex(self, member_id: str, task: str, workspace: str, reason: str):
        request_id = str(uuid4())
        with self.connect() as db:
            db.execute(
                "INSERT INTO codex_requests VALUES (?, ?, ?, ?, ?, 'pending', ?)",
                (request_id, member_id, task, workspace, reason, self.now()),
            )
        return request_id

    def approve_codex(self, member_id: str, request_id: str):
        with self.connect() as db:
            row = db.execute(
                """SELECT * FROM codex_requests WHERE id LIKE ? AND member_id = ?
                AND status = 'pending'""",
                (f"{request_id}%", member_id),
            ).fetchone()
            if not row:
                return None
            db.execute("UPDATE codex_requests SET status = 'approved' WHERE id = ?", (row["id"],))
        return dict(row)

    def reject_codex(self, member_id: str, request_id: str):
        with self.connect() as db:
            cursor = db.execute(
                """UPDATE codex_requests SET status = 'rejected'
                WHERE id LIKE ? AND member_id = ? AND status = 'pending'""",
                (f"{request_id}%", member_id),
            )
        return cursor.rowcount == 1

    def list_codex_requests(self, member_id: str, status: str = "pending"):
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM codex_requests WHERE member_id = ? AND status = ? ORDER BY created_at",
                (member_id, status),
            ).fetchall()
        return [dict(row) for row in rows]

    def register_google_account(self, member_id: str, email_address: str) -> None:
        with self.connect() as db:
            db.execute(
                """INSERT OR REPLACE INTO google_accounts
                (member_id, email_address, connected_at) VALUES (?, ?, ?)""",
                (member_id, email_address.casefold(), self.now()),
            )

    def list_google_accounts(self, member_id: str) -> list[str]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT email_address FROM google_accounts WHERE member_id = ? ORDER BY email_address",
                (member_id,),
            ).fetchall()
        return [str(row["email_address"]) for row in rows]

    def remove_google_account(self, member_id: str, email_address: str) -> bool:
        with self.connect() as db:
            cursor = db.execute(
                "DELETE FROM google_accounts WHERE member_id = ? AND email_address = ?",
                (member_id, email_address.casefold()),
            )
        return cursor.rowcount == 1

    def create_gmail_backup_job(self, member_id: str, accounts: list[str]) -> str:
        job_id = str(uuid4())
        now = self.now()
        with self.connect() as db:
            db.execute(
                """INSERT INTO gmail_backup_jobs
                (id, member_id, accounts, status, created_at, updated_at)
                VALUES (?, ?, ?, 'queued', ?, ?)""",
                (job_id, member_id, json.dumps(accounts), now, now),
            )
        return job_id

    def get_gmail_backup_job(self, member_id: str, job_id: str | None = None):
        with self.connect() as db:
            if job_id:
                row = db.execute(
                    "SELECT * FROM gmail_backup_jobs WHERE member_id = ? AND id LIKE ?",
                    (member_id, f"{job_id}%"),
                ).fetchone()
            else:
                row = db.execute(
                    """SELECT * FROM gmail_backup_jobs WHERE member_id = ?
                    ORDER BY created_at DESC LIMIT 1""", (member_id,)
                ).fetchone()
        if not row:
            return None
        item = dict(row)
        item["accounts"] = json.loads(item["accounts"])
        return item

    def get_gmail_backup_job_for_account(self, member_id: str, email_address: str):
        with self.connect() as db:
            rows = db.execute(
                """SELECT * FROM gmail_backup_jobs WHERE member_id = ?
                ORDER BY created_at DESC""", (member_id,)
            ).fetchall()
        for row in rows:
            item = dict(row)
            item["accounts"] = json.loads(item["accounts"])
            if email_address.casefold() in {str(x).casefold() for x in item["accounts"]}:
                return item
        return None

    def claim_gmail_backup_job(self):
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM gmail_backup_jobs WHERE status = 'queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if not row:
                return None
            db.execute(
                "UPDATE gmail_backup_jobs SET status='running', updated_at=? WHERE id=?",
                (self.now(), row["id"]),
            )
        item = dict(row)
        item["accounts"] = json.loads(item["accounts"])
        return item

    def update_gmail_backup_job(self, job_id: str, **fields) -> None:
        allowed = {"status", "current_account", "total_messages", "completed_messages",
                   "postgres_repaired", "pause_requested", "cancel_requested", "error"}
        values = {key: value for key, value in fields.items() if key in allowed}
        if not values:
            return
        values["updated_at"] = self.now()
        assignments = ", ".join(f"{key} = ?" for key in values)
        with self.connect() as db:
            db.execute(
                f"UPDATE gmail_backup_jobs SET {assignments} WHERE id = ?",
                (*values.values(), job_id),
            )

    def control_gmail_backup_job(self, member_id: str, action: str) -> bool:
        job = self.get_gmail_backup_job(member_id)
        if not job:
            return False
        with self.connect() as db:
            if action == "pause" and job["status"] in {"queued", "running"}:
                db.execute("UPDATE gmail_backup_jobs SET pause_requested=1, updated_at=? WHERE id=?",
                           (self.now(), job["id"]))
            elif action == "resume" and job["status"] == "paused":
                db.execute("""UPDATE gmail_backup_jobs SET status='queued', pause_requested=0,
                    cancel_requested=0, error=NULL, updated_at=? WHERE id=?""", (self.now(), job["id"]))
            elif action == "cancel" and job["status"] in {"queued", "running", "paused"}:
                db.execute("UPDATE gmail_backup_jobs SET cancel_requested=1, updated_at=? WHERE id=?",
                           (self.now(), job["id"]))
            else:
                return False
        return True

    def recover_gmail_backup_jobs(self) -> None:
        with self.connect() as db:
            db.execute("""UPDATE gmail_backup_jobs SET status='queued', updated_at=?
                WHERE status='running'""", (self.now(),))


class PostgresFamilyStorage:
    def __init__(self, connection_string: str, lazy: bool = False):
        import psycopg
        from psycopg.rows import dict_row

        self.connection_string = connection_string
        self.psycopg = psycopg
        self.dict_row = dict_row
        self._setup_lock = Lock()
        self._setup_complete = False
        if not lazy:
            self._setup()
            self._setup_complete = True

    def ensure_setup(self) -> None:
        if self._setup_complete:
            return
        with self._setup_lock:
            if not self._setup_complete:
                self._setup()
                self._setup_complete = True

    def healthcheck(self) -> dict:
        started = __import__("time").monotonic()
        with self.connect() as db:
            db.execute("SELECT 1").fetchone()
        return {
            "database": "postgresql", "reachable": True,
            "latency_ms": round((__import__("time").monotonic() - started) * 1000, 1),
        }

    @contextmanager
    def connect(self):
        with self.psycopg.connect(
            self.connection_string, row_factory=self.dict_row, connect_timeout=3
        ) as connection:
            yield connection

    @staticmethod
    def now():
        return datetime.now(timezone.utc).isoformat()

    def _setup(self):
        with self.connect() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS family_memories (
                    id TEXT PRIMARY KEY,
                    member_id TEXT NOT NULL,
                    scope TEXT NOT NULL CHECK(scope IN ('private', 'family')),
                    content TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS family_todos (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    completed BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS family_agent_registry (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    purpose TEXT NOT NULL,
                    permissions JSONB NOT NULL DEFAULT '[]'::jsonb,
                    status TEXT NOT NULL CHECK(status IN ('active', 'pending', 'rejected')),
                    requested_by TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS family_codex_requests (
                    id TEXT PRIMARY KEY,
                    member_id TEXT NOT NULL,
                    task TEXT NOT NULL,
                    workspace TEXT NOT NULL CHECK(workspace IN ('isolated', 'family_project')),
                    reason TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending', 'approved', 'rejected')),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS family_email_backups (
                    member_id TEXT NOT NULL,
                    google_account_id TEXT NOT NULL,
                    gmail_message_id TEXT NOT NULL,
                    thread_id TEXT,
                    internal_date BIGINT,
                    labels JSONB NOT NULL DEFAULT '[]'::jsonb,
                    raw_email BYTEA NOT NULL,
                    sha256 TEXT NOT NULL,
                    backed_up_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (member_id, google_account_id, gmail_message_id)
                )
            """)
            email_columns = {
                row["column_name"] for row in db.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = 'family_email_backups'
                """).fetchall()
            }
            if "google_account_id" not in email_columns:
                db.execute("ALTER TABLE family_email_backups ADD COLUMN google_account_id TEXT NOT NULL DEFAULT 'legacy'")
                db.execute("ALTER TABLE family_email_backups DROP CONSTRAINT IF EXISTS family_email_backups_pkey")
                db.execute("""ALTER TABLE family_email_backups ADD PRIMARY KEY
                    (member_id, google_account_id, gmail_message_id)""")
                db.execute("ALTER TABLE family_email_backups ALTER COLUMN google_account_id DROP DEFAULT")
            db.execute("""
                CREATE TABLE IF NOT EXISTS family_google_accounts (
                    member_id TEXT NOT NULL,
                    email_address TEXT NOT NULL,
                    connected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (member_id, email_address)
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS family_gmail_backup_jobs (
                    id TEXT PRIMARY KEY,
                    member_id TEXT NOT NULL,
                    accounts JSONB NOT NULL,
                    status TEXT NOT NULL CHECK(status IN
                        ('queued', 'running', 'paused', 'completed', 'failed', 'cancelled')),
                    current_account TEXT,
                    total_messages BIGINT NOT NULL DEFAULT 0,
                    completed_messages BIGINT NOT NULL DEFAULT 0,
                    postgres_repaired BIGINT NOT NULL DEFAULT 0,
                    pause_requested BOOLEAN NOT NULL DEFAULT FALSE,
                    cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
                    error TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            db.execute("""
                INSERT INTO family_agent_registry
                    (id, name, purpose, permissions, status, requested_by)
                VALUES
                    ('builtin-main', 'Main Family Agent', '理解、监管和分配家庭任务', '[]', 'active', 'system'),
                    ('builtin-voice-turn', 'Voice Turn Agent', '判断语音是否面向助手并交给主 Agent', '[]', 'active', 'system'),
                    ('builtin-camera', 'Camera Agent', '按主 Agent 请求读取指定摄像头窗口并进行本地视觉分析', '["camera_window_read"]', 'active', 'system'),
                    ('builtin-search', 'Search Agent', '多策略公开网页搜索', '["network_public"]', 'active', 'system')
                    ,('builtin-weather', 'Weather Agent', '获取实时天气并验证结构化预报', '["network_public"]', 'active', 'system')
                    ,('builtin-memory', 'Memory Agent', '按成员权限读取记忆，写入后重新读取验证', '["family_memory_read", "database_write"]', 'active', 'system')
                    ,('builtin-tasks', 'Tasks Agent', '管理个人和家庭待办，写入与完成后验证', '["database_read", "database_write"]', 'active', 'system')
                    ,('builtin-diagnostics', 'Diagnostics Agent', '隔离检查数据库、硬盘、Gmail、Ollama 与 Codex 健康状态', '["database_read", "files_read"]', 'active', 'system')
                    ,('builtin-codex', 'Codex Specialist Agent', '经批准处理复杂问题和只读工程分析', '["cloud_codex", "files_read"]', 'active', 'system')
                    ,('builtin-curator', 'Model & Capability Curator Agent', '每月检查适合本机的新模型与能力工具；先测试、可回滚、按需提议 Codex', '["network_public", "ollama_model_manage"]', 'active', 'system')
                    ,('builtin-financial-video', 'Financial Video Agent', '制作金融教育短视频审核稿；默认 Ollama，用户可明确指定 Codex', '["files_read", "files_write", "cloud_codex"]', 'active', 'system')
                    ,('builtin-gmail-backup', 'Gmail & Calendar Agent', '每天只读增量同步 Gmail 到 PostgreSQL；用户明确要求时从 PostgreSQL 导出硬盘；只读查询 Calendar', '["email", "calendar", "database_write", "files_write"]', 'active', 'system')
                    ,('builtin-school-email', '* School Agent', '从 PostgreSQL 检索并总结 * 学校邮件，每日发送重要事项表格', '["email_read", "email_send", "database_read"]', 'active', 'system')
                ON CONFLICT (id) DO UPDATE SET name=EXCLUDED.name,
                    purpose=EXCLUDED.purpose, permissions=EXCLUDED.permissions,
                    status='active', requested_by='system'
            """)

    def add_memory(self, member_id: str, content: str, scope: str = "private"):
        memory_id = str(uuid4())
        with self.connect() as db:
            db.execute(
                "INSERT INTO family_memories (id, member_id, scope, content) VALUES (%s, %s, %s, %s)",
                (memory_id, member_id, scope, content),
            )
        return memory_id

    def list_memories(self, member_id: str, limit: int = 20):
        with self.connect() as db:
            rows = db.execute(
                """SELECT id, member_id, scope, content, created_at FROM family_memories
                WHERE member_id = %s OR scope = 'family'
                ORDER BY created_at DESC LIMIT %s""",
                (member_id, limit),
            ).fetchall()
        return list(rows)

    def add_todo(self, owner_id: str, content: str):
        todo_id = str(uuid4())
        with self.connect() as db:
            db.execute(
                "INSERT INTO family_todos (id, owner_id, content) VALUES (%s, %s, %s)",
                (todo_id, owner_id, content),
            )
        return todo_id

    def list_todos(self, owner_id: str):
        with self.connect() as db:
            rows = db.execute(
                """SELECT id, owner_id, content, completed, created_at FROM family_todos
                WHERE owner_id IN (%s, 'family') ORDER BY completed, created_at DESC""",
                (owner_id,),
            ).fetchall()
        return list(rows)

    def complete_todo(self, owner_id: str, todo_id: str):
        with self.connect() as db:
            cursor = db.execute(
                "UPDATE family_todos SET completed = TRUE WHERE id LIKE %s AND owner_id IN (%s, 'family')",
                (f"{todo_id}%", owner_id),
            )
        return cursor.rowcount == 1

    def store_email_backup(self, member_id: str, google_account_id: str,
                           message: dict, raw_email: bytes, sha256: str) -> None:
        with self.connect() as db:
            db.execute(
                """INSERT INTO family_email_backups
                (member_id, google_account_id, gmail_message_id, thread_id, internal_date,
                 labels, raw_email, sha256)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                ON CONFLICT (member_id, google_account_id, gmail_message_id) DO UPDATE SET
                    thread_id = EXCLUDED.thread_id,
                    internal_date = EXCLUDED.internal_date,
                    labels = EXCLUDED.labels,
                    raw_email = EXCLUDED.raw_email,
                    sha256 = EXCLUDED.sha256,
                    backed_up_at = NOW()""",
                (member_id, google_account_id.casefold(), str(message["id"]), message.get("threadId"),
                 int(message["internalDate"]) if message.get("internalDate") else None,
                 json.dumps(message.get("labelIds", [])), raw_email, sha256),
            )

    def has_email_backup(self, member_id: str, google_account_id: str,
                         message_id: str, sha256: str | None = None) -> bool:
        with self.connect() as db:
            if sha256:
                row = db.execute(
                    """SELECT 1 FROM family_email_backups
                    WHERE member_id = %s AND google_account_id = %s
                    AND gmail_message_id = %s AND sha256 = %s""",
                    (member_id, google_account_id.casefold(), message_id, sha256),
                ).fetchone()
            else:
                row = db.execute(
                    """SELECT 1 FROM family_email_backups
                    WHERE member_id = %s AND google_account_id = %s AND gmail_message_id = %s""",
                    (member_id, google_account_id.casefold(), message_id),
                ).fetchone()
        return row is not None

    def email_backup_count(self, member_id: str, google_account_id: str) -> int:
        with self.connect() as db:
            row = db.execute(
                """SELECT COUNT(*) AS count FROM family_email_backups
                WHERE member_id = %s AND google_account_id = %s""",
                (member_id, google_account_id.casefold()),
            ).fetchone()
        return int(row["count"])

    def email_backup_latest_internal_date(
        self, member_id: str, google_account_id: str
    ) -> int | None:
        """Return the newest archived Gmail internal timestamp in milliseconds."""
        with self.connect() as db:
            row = db.execute(
                """SELECT MAX(internal_date) AS latest FROM family_email_backups
                WHERE member_id = %s AND google_account_id = %s""",
                (member_id, google_account_id.casefold()),
            ).fetchone()
        return int(row["latest"]) if row and row["latest"] is not None else None

    def email_backup_ids(self, member_id: str, google_account_id: str) -> set[str]:
        """Load lightweight Gmail IDs once for an incremental mailbox traversal."""
        with self.connect() as db:
            rows = db.execute(
                """SELECT gmail_message_id FROM family_email_backups
                WHERE member_id = %s AND google_account_id = %s""",
                (member_id, google_account_id.casefold()),
            ).fetchall()
        return {str(row["gmail_message_id"]) for row in rows}

    def iter_email_backups(self, member_id: str, google_account_id: str):
        """Stream archived messages without loading an entire mailbox into RAM."""
        with self.connect() as db:
            with db.cursor(name=f"family_ai_email_export_{uuid4().hex}") as cursor:
                cursor.itersize = 100
                cursor.execute(
                    """SELECT gmail_message_id, thread_id, internal_date, labels,
                    raw_email, sha256, backed_up_at FROM family_email_backups
                    WHERE member_id = %s AND google_account_id = %s
                    ORDER BY internal_date NULLS LAST, gmail_message_id""",
                    (member_id, google_account_id.casefold()),
                )
                for row in cursor:
                    yield dict(row)

    def recent_email_backups(self, member_id: str, since_ms: int,
                             limit: int = 2000) -> list[dict]:
        """Return a bounded recent window across all connected accounts."""
        with self.connect() as db:
            rows = db.execute(
                """SELECT google_account_id, gmail_message_id, thread_id, internal_date,
                labels, raw_email, backed_up_at FROM family_email_backups
                WHERE member_id = %s AND internal_date >= %s
                ORDER BY internal_date DESC LIMIT %s""",
                (member_id, since_ms, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def register_google_account(self, member_id: str, email_address: str) -> None:
        with self.connect() as db:
            db.execute(
                """INSERT INTO family_google_accounts (member_id, email_address)
                VALUES (%s, %s) ON CONFLICT (member_id, email_address)
                DO UPDATE SET connected_at = NOW()""",
                (member_id, email_address.casefold()),
            )

    def list_google_accounts(self, member_id: str) -> list[str]:
        with self.connect() as db:
            rows = db.execute(
                """SELECT email_address FROM family_google_accounts
                WHERE member_id = %s ORDER BY email_address""",
                (member_id,),
            ).fetchall()
        return [str(row["email_address"]) for row in rows]

    def remove_google_account(self, member_id: str, email_address: str) -> bool:
        with self.connect() as db:
            cursor = db.execute(
                """DELETE FROM family_google_accounts
                WHERE member_id = %s AND email_address = %s""",
                (member_id, email_address.casefold()),
            )
        return cursor.rowcount == 1

    def create_gmail_backup_job(self, member_id: str, accounts: list[str]) -> str:
        job_id = str(uuid4())
        with self.connect() as db:
            db.execute(
                """INSERT INTO family_gmail_backup_jobs (id, member_id, accounts, status)
                VALUES (%s, %s, %s::jsonb, 'queued')""",
                (job_id, member_id, json.dumps(accounts)),
            )
        return job_id

    def get_gmail_backup_job(self, member_id: str, job_id: str | None = None):
        with self.connect() as db:
            if job_id:
                row = db.execute(
                    """SELECT * FROM family_gmail_backup_jobs
                    WHERE member_id=%s AND id LIKE %s""", (member_id, f"{job_id}%")
                ).fetchone()
            else:
                row = db.execute(
                    """SELECT * FROM family_gmail_backup_jobs WHERE member_id=%s
                    ORDER BY created_at DESC LIMIT 1""", (member_id,)
                ).fetchone()
        return dict(row) if row else None

    def get_gmail_backup_job_for_account(self, member_id: str, email_address: str):
        with self.connect() as db:
            row = db.execute(
                """SELECT * FROM family_gmail_backup_jobs
                WHERE member_id=%s AND accounts ? %s
                ORDER BY created_at DESC LIMIT 1""",
                (member_id, email_address.casefold()),
            ).fetchone()
        return dict(row) if row else None

    def claim_gmail_backup_job(self):
        with self.connect() as db:
            # A process crash or overlapping agent rebuild can leave a job
            # marked running after its worker is gone. Active jobs update
            # progress every few seconds, so ten minutes without a heartbeat
            # is safe to reclaim.
            db.execute("""UPDATE family_gmail_backup_jobs SET status='queued', updated_at=NOW()
                WHERE status='running' AND updated_at < NOW() - INTERVAL '10 minutes'""")
            row = db.execute("""SELECT * FROM family_gmail_backup_jobs
                WHERE status='queued' ORDER BY created_at LIMIT 1 FOR UPDATE SKIP LOCKED""").fetchone()
            if not row:
                return None
            db.execute("""UPDATE family_gmail_backup_jobs SET status='running', updated_at=NOW()
                WHERE id=%s""", (row["id"],))
        return dict(row)

    def update_gmail_backup_job(self, job_id: str, **fields) -> None:
        allowed = {"status", "current_account", "total_messages", "completed_messages",
                   "postgres_repaired", "pause_requested", "cancel_requested", "error"}
        values = {key: value for key, value in fields.items() if key in allowed}
        if not values:
            return
        assignments = ", ".join(f"{key} = %s" for key in values)
        with self.connect() as db:
            db.execute(
                f"UPDATE family_gmail_backup_jobs SET {assignments}, updated_at=NOW() WHERE id=%s",
                (*values.values(), job_id),
            )

    def control_gmail_backup_job(self, member_id: str, action: str) -> bool:
        job = self.get_gmail_backup_job(member_id)
        if not job:
            return False
        with self.connect() as db:
            if action == "pause" and job["status"] in {"queued", "running"}:
                db.execute("""UPDATE family_gmail_backup_jobs SET pause_requested=TRUE,
                    updated_at=NOW() WHERE id=%s""", (job["id"],))
            elif action == "resume" and job["status"] == "paused":
                db.execute("""UPDATE family_gmail_backup_jobs SET status='queued',
                    pause_requested=FALSE, cancel_requested=FALSE, error=NULL, updated_at=NOW()
                    WHERE id=%s""", (job["id"],))
            elif action == "cancel" and job["status"] in {"queued", "running", "paused"}:
                db.execute("""UPDATE family_gmail_backup_jobs SET cancel_requested=TRUE,
                    updated_at=NOW() WHERE id=%s""", (job["id"],))
            else:
                return False
        return True

    def recover_gmail_backup_jobs(self) -> None:
        with self.connect() as db:
            db.execute("""UPDATE family_gmail_backup_jobs SET status='queued', updated_at=NOW()
                WHERE status='running'""")

    def propose_agent(self, member_id: str, name: str, purpose: str, permissions: list[str]):
        proposal_id = str(uuid4())
        with self.connect() as db:
            db.execute(
                """INSERT INTO family_agent_registry
                (id, name, purpose, permissions, status, requested_by)
                VALUES (%s, %s, %s, %s::jsonb, 'pending', %s)""",
                (proposal_id, name, purpose, json.dumps(permissions), member_id),
            )
        return proposal_id

    def list_agents(self):
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM family_agent_registry ORDER BY status, created_at"
            ).fetchall()
        return list(rows)

    def decide_agent(self, proposal_id: str, approved: bool):
        status = "active" if approved else "rejected"
        with self.connect() as db:
            cursor = db.execute(
                """UPDATE family_agent_registry SET status = %s
                WHERE id LIKE %s AND status = 'pending'""",
                (status, f"{proposal_id}%"),
            )
        return cursor.rowcount == 1

    def propose_codex(self, member_id: str, task: str, workspace: str, reason: str):
        request_id = str(uuid4())
        with self.connect() as db:
            db.execute(
                """INSERT INTO family_codex_requests
                (id, member_id, task, workspace, reason, status)
                VALUES (%s, %s, %s, %s, %s, 'pending')""",
                (request_id, member_id, task, workspace, reason),
            )
        return request_id

    def approve_codex(self, member_id: str, request_id: str):
        with self.connect() as db:
            row = db.execute(
                """SELECT * FROM family_codex_requests WHERE id LIKE %s AND member_id = %s
                AND status = 'pending'""",
                (f"{request_id}%", member_id),
            ).fetchone()
            if not row:
                return None
            db.execute(
                "UPDATE family_codex_requests SET status = 'approved' WHERE id = %s",
                (row["id"],),
            )
        return row

    def reject_codex(self, member_id: str, request_id: str):
        with self.connect() as db:
            cursor = db.execute(
                """UPDATE family_codex_requests SET status = 'rejected'
                WHERE id LIKE %s AND member_id = %s AND status = 'pending'""",
                (f"{request_id}%", member_id),
            )
        return cursor.rowcount == 1

    def list_codex_requests(self, member_id: str, status: str = "pending"):
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM family_codex_requests WHERE member_id = %s AND status = %s ORDER BY created_at",
                (member_id, status),
            ).fetchall()
        return list(rows)
