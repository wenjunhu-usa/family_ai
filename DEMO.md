# Family AI demo recording plan

Record a real 60–90 second session with synthetic household data. This is a
script for a future recording, not evidence that a recording already exists.

## Before recording

Follow the README local setup. Use a separate data directory and demo member
identifiers. Check each command before recording and keep credentials, personal
email, camera footage, and real household details off screen. State the Mac
hardware and Ollama model used in the recording description.

## Main sequence

| Time | On-screen action | Point to explain |
| --- | --- | --- |
| 0–10 s | Show the local interface with member `demo-parent` | One local assistant for household information |
| 10–25 s | Run `/remember family Recycling pickup is Thursday.` then `/memory` | A shared household memory |
| 25–40 s | Run `/todo add family Put the recycling bins out Wednesday evening.` then `/todo list` | A shared task with a concrete use |
| 40–55 s | Switch to `demo-partner`, run `/memory` and `/todo list` | The shared information is available to another member |
| 55–70 s | Save `/remember private My gift idea is a telescope.` and show `/memory`; switch back to `demo-parent` and show `/memory` | Private notes are scoped to a member; this does not demonstrate identity authentication |
| 70–90 s | Show the architecture diagram and repository URL | Local models, shared tasks, member-scoped memory, optional MCP access |

Suggested closing caption: **Family data stored locally. Core AI inference on
your Mac.**

## Optional follow-up clips

- After configuring MCP, call `whoami` and `system_status` from a trusted client.
- After configuring an allowlisted camera window, show local person detection,
  person counting, and a description of a staged, non-sensitive scene. The
  public implementation does not attach a name or identity to a detected person.
- Demonstrate calendar, email, or document retrieval only after configuring the
  relevant integration and preparing synthetic source content.

## GitHub About copy

Description:

> A local-first household AI assistant with shared tasks, member-scoped memory, local models, and authenticated MCP tools.

Suggested topics:

```text
family-ai local-ai ai-agent multi-agent langgraph mcp ollama local-llm privacy self-hosted
```

Add a real recording link to the README once it is available. Do not use an
unverified competitor comparison or advertise integrations that are only planned.
