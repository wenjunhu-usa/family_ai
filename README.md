# Family AI Assistant

Family AI is a local-first household assistant built with FastAPI, LangGraph,
Ollama, React, Tauri, and an optional database backend. The Mac acts as the central server. Other
trusted devices on the same LAN can use its tools through an authenticated MCP
endpoint.

Personal values in this public copy are replaced with `*`, including member
names, wake words, email addresses, local paths, host names, LAN addresses,
school names, camera names, camera applications, device names, and tokens.
Replace each required `*` locally before running the corresponding feature.

## Architecture

```text
Windows / macOS MCP client
          |
          | Streamable HTTP + bearer token (LAN only)
          v
Mac Family AI MCP server :8001
          |
          +-- LangGraph supervisor and specialist agents
          +-- Ollama :11434 (loopback only)
          +-- optional database (private host only)
          +-- memory, tasks, calendar, email backup, RAG, camera tools
```

MCP adds a remote tool interface; it does not replace LangGraph. LangGraph
continues to plan and coordinate the Family AI workflow on the Mac. Ollama,
The database, files, schedules, and private data remain on the Mac.

## Main capabilities

- Local chat and reasoning through Ollama
- LangGraph supervisor with specialist agents
- Private and family-scoped memory
- Shared and personal tasks
- Read-only calendar access
- Read-only email backup
- Optional database-backed document RAG
- Local voice input and speech output
- Permission-limited local camera inspection
- Authenticated Streamable HTTP MCP server
- React/Tauri desktop interface

## Requirements

- macOS host with Python 3.11+, `uv`, Ollama, Node.js, and Rust
- An optional private database when persistent shared storage is required
- Camera applications required by the local configuration: `*`
- A Windows or macOS client with Cline when remote MCP access is needed

Keep `.env`, databases, OAuth credentials, MCP device files, model files,
backups, logs, build output, and generated artifacts out of Git.

## Configuration

Copy the example environment file and replace the required placeholders:

```sh
cp .env.example .env
```

Important settings include:

```dotenv
FAMILY_AI_DATA_DIR=*
DATABASE_URL=*
VOICE_WAKE_WORD=*
GOOGLE_OAUTH_CLIENT_ID=*
GOOGLE_OAUTH_CLIENT_SECRET=*
GMAIL_BACKUP_ROOT=*
FAMILY_AI_MCP_ALLOWED_HOSTS=localhost,127.0.0.1,*,*
```

Use a unique secret for every installation. Do not commit the completed `.env`.

## Start locally on the Mac

```sh
uv sync
ollama serve
./scripts/start.sh
```

The local web interface is available at `http://127.0.0.1:8000`. Start the MCP
server for development with:

```sh
./scripts/start-mcp.sh
```

The packaged runtime includes launchd definitions for the application, Ollama,
MCP server, and scheduled curator. Update every `*` path before installing them.

## Create an MCP device token

Run this command locally on the Mac:

```sh
family-ai-mcp issue-device --name "*" --member "*" --admin
```

The bearer token is displayed once. Store it only on the intended client. The
server stores a SHA-256 token hash in its ignored device registry. Revoke a
client with:

```sh
family-ai-mcp revoke-device "*"
```

The MCP listener accepts loopback, private, and link-local client addresses.
Keep Ollama and the database private; expose only the authenticated MCP port to
the trusted LAN.

## Connect from Windows with Cline

This project has been tested with Cline on another Windows computer using the
Mac as the Family AI server. Both computers must be on the same trusted LAN.

1. Start Family AI, Ollama, and the MCP server on the Mac.
2. In Cline, open **MCP**, then select **Add MCP Server**.
3. Enter a server name such as `family-ai`.
4. Select **Remote** and **Streamable HTTP**.
5. Enter the URL `http://*:8001/mcp`.
6. Add header `Authorization` with value `Bearer *`.
7. Enable the server and refresh its tool list.

Use the Mac LAN IP address or LAN host name in place of the URL's `*`. Use the
per-device token in place of the header's `*`.

Verify the connection from Cline:

```text
Use the family-ai MCP server. Call whoami and system_status.
```

Expected results are an authenticated device identity and overall status
`online`. `loaded_models: []` is normal while Ollama is idle; the configured
model loads on the first local-model request.

If Cline receives `Valid Family AI device token required`, verify the complete
`Authorization: Bearer *` header and ensure the token belongs to an active
device. If the connection is refused, verify port `8001`, the Mac firewall, and
the MCP allowed-host configuration.

## Camera configuration

All camera and camera-application names are intentionally shown as `*` in this
public copy. Configure the local provider mapping and allowlisted window owner
names before enabling camera monitoring. Camera pixels are captured only from
approved application windows and analyzed by the local vision model.

Example public command form:

```text
/camera * Describe the current view
```

The camera agent does not identify people. Automatic monitoring is disabled by
default and must be explicitly configured locally.

## Email and calendar

Email backup uses read-only OAuth access. Tokens remain in the macOS Keychain.
Backups can be stored on a configured removable drive and optionally mirrored
in the database. Calendar access is read-only.

Configure the OAuth client ID, secret, and redirect URI in `.env`. Supported
commands include:

```text
/gmail connect
/gmail backup
/gmail status
/gmail disconnect
```

## Private knowledge RAG

An optional database backend can store document embeddings and application
state. Its host, schema, credentials, and deployment details are intentionally
omitted from this public copy. Private documents are scoped to one member;
family documents are shared. Retrieved excerpts are added only to the local
model context.

## Security notes

- MCP requires a separate bearer token for every client device.
- MCP device tokens are stored as hashes by the server.
- Ollama should listen on loopback only.
- The configured database should remain private and require authentication.
- Credentials and personal data belong only in ignored runtime files.
- Camera access is limited to configured local application windows.
- Codex requests exclude credentials, databases, memories, and conversation
  history from sanitized project copies.

## Tests

```sh
uv run pytest -q
```

Desktop build configuration is under `desktop/`; macOS launchd definitions are
under `launchd/`.
