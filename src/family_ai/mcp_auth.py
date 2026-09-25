from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import secrets
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock


@dataclass(frozen=True)
class DeviceIdentity:
    device_id: str
    name: str
    member_id: str
    permissions: tuple[str, ...]


DEFAULT_PERMISSIONS = (
    "system:read", "agents:read", "weather:read", "tasks:read", "tasks:write",
    "memory:read", "memory:write", "calendar:read", "gmail:read", "gmail:backup",
    "knowledge:read", "assistant:chat",
)
ADMIN_PERMISSIONS = DEFAULT_PERMISSIONS + ("camera:inspect", "video:create")

current_device: ContextVar[DeviceIdentity | None] = ContextVar(
    "family_ai_mcp_device", default=None
)


class DeviceRegistry:
    """File-backed LAN device credentials. Only token hashes are persisted."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = Lock()

    @staticmethod
    def _digest(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def _read(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"version": 1, "devices": []}
        if data.get("version") != 1 or not isinstance(data.get("devices"), list):
            raise ValueError("Invalid MCP device registry")
        return data

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(self.path)

    def issue(self, name: str, member_id: str,
              permissions: tuple[str, ...] = DEFAULT_PERMISSIONS) -> tuple[DeviceIdentity, str]:
        name, member_id = name.strip(), member_id.strip()
        if not name or not member_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in member_id):
            raise ValueError("Device name and a simple member ID are required")
        token = secrets.token_urlsafe(32)
        device_id = secrets.token_hex(8)
        identity = DeviceIdentity(device_id, name, member_id, tuple(sorted(set(permissions))))
        record = {
            **asdict(identity), "permissions": list(identity.permissions),
            "token_sha256": self._digest(token),
            "created_at": datetime.now(timezone.utc).isoformat(), "revoked": False,
        }
        with self._lock:
            data = self._read()
            data["devices"].append(record)
            self._write(data)
        return identity, token

    def authenticate(self, token: str) -> DeviceIdentity | None:
        if not token:
            return None
        supplied = self._digest(token)
        with self._lock:
            records = self._read()["devices"]
        for item in records:
            if not item.get("revoked") and hmac.compare_digest(
                str(item.get("token_sha256", "")), supplied
            ):
                return DeviceIdentity(
                    str(item["device_id"]), str(item["name"]), str(item["member_id"]),
                    tuple(str(value) for value in item.get("permissions", [])),
                )
        return None

    def list_devices(self) -> list[dict]:
        with self._lock:
            records = self._read()["devices"]
        return [{key: value for key, value in item.items() if key != "token_sha256"}
                for item in records]

    def revoke(self, device_id: str) -> bool:
        changed = False
        with self._lock:
            data = self._read()
            for item in data["devices"]:
                if item.get("device_id") == device_id and not item.get("revoked"):
                    item["revoked"], changed = True, True
            if changed:
                self._write(data)
        return changed


def require_device(permission: str) -> DeviceIdentity:
    identity = current_device.get()
    if identity is None:
        raise PermissionError("MCP device authentication is required")
    if permission not in identity.permissions:
        raise PermissionError(f"This device is not allowed to use {permission}")
    return identity


class LANBearerAuthMiddleware:
    """Pure ASGI bearer gate, avoiding request-body middleware around MCP streams."""

    def __init__(self, app, registry: DeviceRegistry):
        self.app, self.registry = app, registry

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        host = (scope.get("client") or ("", 0))[0]
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        if address is None or not (address.is_private or address.is_loopback or address.is_link_local):
            await self._reject(send, 403, "LAN access only")
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        authorization = headers.get(b"authorization", b"").decode("latin-1")
        token = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
        identity = self.registry.authenticate(token)
        if identity is None:
            await self._reject(send, 401, "Valid Family AI device token required")
            return
        marker = current_device.set(identity)
        try:
            await self.app(scope, receive, send)
        finally:
            current_device.reset(marker)

    @staticmethod
    async def _reject(send, status: int, message: str):
        body = json.dumps({"error": message}).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode()),
                                (b"www-authenticate", b"Bearer")]})
        await send({"type": "http.response.body", "body": body})
