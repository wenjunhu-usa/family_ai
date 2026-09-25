import asyncio

from family_ai.mcp_auth import (
    DEFAULT_PERMISSIONS, DeviceRegistry, LANBearerAuthMiddleware, current_device,
    require_device,
)


def test_device_tokens_are_hashed_and_map_to_member(tmp_path):
    registry = DeviceRegistry(tmp_path / "devices.json")
    identity, token = registry.issue("Test-Laptop", "member")
    stored = (tmp_path / "devices.json").read_text()
    assert token not in stored
    assert registry.authenticate(token) == identity
    assert registry.authenticate("wrong") is None
    assert "tasks:read" in identity.permissions


def test_revoked_device_cannot_authenticate(tmp_path):
    registry = DeviceRegistry(tmp_path / "devices.json")
    identity, token = registry.issue("Tablet", "member_two", DEFAULT_PERMISSIONS)
    assert registry.revoke(identity.device_id)
    assert registry.authenticate(token) is None


def test_tool_permission_comes_from_authenticated_device():
    from family_ai.mcp_auth import DeviceIdentity

    identity = DeviceIdentity("one", "Reader", "member", ("tasks:read",))
    marker = current_device.set(identity)
    try:
        assert require_device("tasks:read") == identity
        try:
            require_device("tasks:write")
        except PermissionError:
            pass
        else:
            raise AssertionError("write permission should be rejected")
    finally:
        current_device.reset(marker)


def test_lan_bearer_middleware_sets_request_identity(tmp_path):
    registry = DeviceRegistry(tmp_path / "devices.json")
    identity, token = registry.issue("Laptop", "member")
    observed = []

    async def downstream(scope, receive, send):
        observed.append(current_device.get())
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    messages = []

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http", "client": ("192.0.2.20", 50000),
        "headers": [(b"authorization", f"Bearer {token}".encode())],
    }
    asyncio.run(LANBearerAuthMiddleware(downstream, registry)(scope, None, send))
    assert observed == [identity]
    assert messages[0]["status"] == 200
    assert current_device.get() is None


def test_lan_bearer_middleware_rejects_missing_token(tmp_path):
    called = []

    async def downstream(scope, receive, send):
        called.append(True)

    messages = []

    async def send(message):
        messages.append(message)

    scope = {"type": "http", "client": ("192.0.2.20", 50000), "headers": []}
    registry = DeviceRegistry(tmp_path / "devices.json")
    asyncio.run(LANBearerAuthMiddleware(downstream, registry)(scope, None, send))
    assert not called
    assert messages[0]["status"] == 401
