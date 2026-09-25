from __future__ import annotations

import argparse
import sys
from functools import lru_cache

from .config import settings
from .mcp_auth import (
    ADMIN_PERMISSIONS, DEFAULT_PERMISSIONS, DeviceRegistry, LANBearerAuthMiddleware,
    require_device,
)
from .mcp_services import FamilyMCPServices


@lru_cache
def services() -> FamilyMCPServices:
    return FamilyMCPServices()


def create_server():
    try:
        from mcp.server.fastmcp import FastMCP
        from mcp.server.transport_security import TransportSecuritySettings
    except ImportError as exc:
        raise RuntimeError("Install the project dependencies to run the MCP server") from exc

    server = FastMCP(
        "Family AI", instructions=(
            "Private family tools hosted on the household server. The authenticated device "
            "determines member identity; never request or invent a member_id."
        ), stateless_http=True,
        # Keep SDK DNS-rebinding protection enabled while accepting only the
        # explicitly configured names and addresses of this host.
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[
                item
                for host in settings.family_ai_mcp_allowed_hosts.split(",")
                for item in (host.strip(), f"{host.strip()}:*")
                if host.strip()
            ],
        ),
    )

    @server.tool()
    def whoami() -> dict:
        """Return the authenticated device, family member, and granted permissions."""
        identity = require_device("system:read")
        return {"device_id": identity.device_id, "device_name": identity.name,
                "member_id": identity.member_id, "permissions": list(identity.permissions)}

    @server.tool()
    def system_status() -> dict:
        """Read Family AI, Ollama, loaded-model, and database health."""
        require_device("system:read")
        return services().system_status()

    @server.tool()
    def agents_list() -> list[dict]:
        """List built-in and approved Family AI agents without private content."""
        require_device("agents:read")
        return services().agents_list()

    @server.tool()
    def tasks_list() -> list[dict]:
        """List this member's private tasks and shared family tasks."""
        identity = require_device("tasks:read")
        return services().tasks_list(identity.member_id)

    @server.tool()
    def tasks_add(content: str, owner: str = "me") -> dict:
        """Add a task for this member or the family. Owner must be 'me' or 'family'."""
        identity = require_device("tasks:write")
        return services().tasks_add(identity.member_id, content, owner)

    @server.tool()
    def tasks_complete(task_id: str) -> dict:
        """Complete one task visible to this member using its ID or unique ID prefix."""
        identity = require_device("tasks:write")
        return services().tasks_complete(identity.member_id, task_id)

    @server.tool()
    def memory_list() -> list[dict]:
        """List this member's private memories and shared family memories."""
        identity = require_device("memory:read")
        return services().memory_list(identity.member_id)

    @server.tool()
    def memory_add(content: str, scope: str = "private") -> dict:
        """Save a memory with scope 'private' or 'family'."""
        identity = require_device("memory:write")
        return services().memory_add(identity.member_id, content, scope)

    @server.tool()
    def weather_forecast(location: str, days: int = 1, language: str = "zh") -> list[dict]:
        """Get one to seven days of current Open-Meteo forecast data."""
        require_device("weather:read")
        return services().weather(location, days, language)

    @server.tool()
    def calendar_upcoming(account: str | None = None, days: int = 7) -> dict:
        """Read upcoming Google Calendar events for this member; use '*' for all accounts."""
        identity = require_device("calendar:read")
        return services().calendar_upcoming(identity.member_id, account, days)

    @server.tool()
    def gmail_status(account: str | None = None) -> dict:
        """Read Gmail connection and backup status without returning message content."""
        identity = require_device("gmail:read")
        return services().gmail_status(identity.member_id, account)

    @server.tool()
    def gmail_backup_start(account: str | None = None) -> dict:
        """Queue a durable Gmail backup for one account; use '*' for every connected account."""
        identity = require_device("gmail:backup")
        return services().gmail_backup_start(identity.member_id, account)

    @server.tool()
    def gmail_backup_status() -> dict | None:
        """Read this member's current or most recent Gmail backup job."""
        identity = require_device("gmail:read")
        return services().gmail_backup_status(identity.member_id)

    @server.tool()
    def knowledge_search(query: str, limit: int = 5) -> list[dict]:
        """Search private and family knowledge available to this member."""
        identity = require_device("knowledge:read")
        return services().knowledge_search(identity.member_id, query, limit)

    @server.tool()
    def family_assistant_chat(message: str, thread_id: str = "mcp") -> dict:
        """Use the full LangGraph Family AI workflow for a complex or conversational request."""
        identity = require_device("assistant:chat")
        safe_thread = "".join(c for c in thread_id if c.isalnum() or c in "-_")[:60] or "mcp"
        return services().assistant_chat(identity.member_id, message, f"mcp-{identity.device_id}-{safe_thread}")

    @server.tool()
    def camera_inspect(provider: str, question: str = "描述当前画面",
                       language: str = "zh") -> dict:
        """Inspect a current camera-app window locally on the host."""
        require_device("camera:inspect")
        return services().camera_inspect(provider, question, language)

    @server.tool()
    def financial_video_create(request: str) -> dict:
        """Create a local Ollama-generated financial education review video."""
        require_device("video:create")
        return services().video_create(request)

    return server


def issue_device(registry: DeviceRegistry, name: str, member_id: str, admin: bool) -> int:
    identity, token = registry.issue(
        name, member_id, ADMIN_PERMISSIONS if admin else DEFAULT_PERMISSIONS
    )
    print(f"Device ID: {identity.device_id}")
    print(f"Member: {identity.member_id}")
    print(f"Bearer token (shown once): {token}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Family AI LAN MCP server")
    subparsers = parser.add_subparsers(dest="command")
    issue = subparsers.add_parser("issue-device", help="issue a bearer token locally")
    issue.add_argument("--name", required=True)
    issue.add_argument("--member", required=True)
    issue.add_argument("--admin", action="store_true")
    listing = subparsers.add_parser("list-devices")
    revoke = subparsers.add_parser("revoke-device")
    revoke.add_argument("device_id")
    args = parser.parse_args()
    registry = DeviceRegistry(settings.mcp_devices_file)
    if args.command == "issue-device":
        return issue_device(registry, args.name, args.member, args.admin)
    if args.command == "list-devices":
        import json
        print(json.dumps(registry.list_devices(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "revoke-device":
        return 0 if registry.revoke(args.device_id) else 1

    server = create_server()
    server.settings.streamable_http_path = "/mcp"
    app = LANBearerAuthMiddleware(server.streamable_http_app(), registry)
    try:
        import uvicorn
        uvicorn.run(app, host=settings.family_ai_mcp_host,
                    port=settings.family_ai_mcp_port, log_level="info")
    finally:
        if services.cache_info().currsize:
            current = services()
            if "gmail_worker" in current.__dict__:
                current.gmail_worker.stop()
            if "family_agent" in current.__dict__:
                current.family_agent.gmail_backup_worker.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
