import asyncio
import hmac
import ipaddress
import json
import logging
import re
import secrets
from threading import Lock
from functools import lru_cache

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from .graph import build_agent
from .config import settings
from .curator import CuratorAgent
from .codex_cli import CodexCLI
from .task_registry import TaskRegistry
from .camera_agent import CameraAgent
from .camera_monitor import CameraMonitor
from .speech import LocalSpeech
from .voice_turn import VoiceTurnAgent
from .rag import RagStore
from .gmail_backup import GmailBackupAgent, GmailBackupError
from .agent_activity import active_agent_ids, agent_activity
from .storage import FamilyStorage, PostgresFamilyStorage


app = FastAPI(title="Family AI Assistant", version="0.1.0")
logger = logging.getLogger(__name__)
task_registry = TaskRegistry(settings.family_ai_data_dir / "agent-tasks.sqlite3")
BUILTIN_AGENT_STATUS = [
    {"id": "builtin-main", "name": "Main Family Agent", "status": "active"},
    {"id": "builtin-voice-turn", "name": "Voice Turn Agent", "status": "active"},
    {"id": "builtin-camera", "name": "Camera Agent", "status": "active"},
    {"id": "builtin-search", "name": "Search Agent", "status": "active"},
    {"id": "builtin-weather", "name": "Weather Agent", "status": "active"},
    {"id": "builtin-memory", "name": "Memory Agent", "status": "active"},
    {"id": "builtin-tasks", "name": "Tasks Agent", "status": "active"},
    {"id": "builtin-diagnostics", "name": "Diagnostics Agent", "status": "active"},
    {"id": "builtin-codex", "name": "Codex Specialist Agent", "status": "active"},
    {"id": "builtin-curator", "name": "Model & Capability Curator Agent", "status": "active"},
    {"id": "builtin-financial-video", "name": "Financial Video Agent", "status": "active"},
    {"id": "builtin-gmail-backup", "name": "Gmail & Calendar Agent", "status": "active"},
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "tauri://localhost", "http://tauri.localhost", "https://tauri.localhost",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    member_id: str = Field(min_length=1, max_length=40, pattern=r"^[\w-]+$")
    thread_id: str = Field(default="home", min_length=1, max_length=80)
    message: str = Field(min_length=1, max_length=8000)


class ChatResponse(BaseModel):
    reply: str


@lru_cache
def agent():
    return build_agent()


_agent_reset_lock = Lock()
_approval_cache_lock = Lock()
_approval_cache: dict[str, list[dict]] = {}
_database_reachable: bool | None = None


def lan_access_token() -> str:
    """Create one local pairing token without placing it in source control."""
    if settings.family_ai_lan_access_token:
        return settings.family_ai_lan_access_token
    path = settings.family_ai_data_dir / "lan-access-token.txt"
    try:
        token = path.read_text().strip()
    except OSError:
        token = ""
    if not token:
        token = secrets.token_hex(4)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(token + "\n")
        path.chmod(0o600)
    return token


@app.middleware("http")
async def protect_lan_api(request: Request, call_next):
    """Keep private APIs inaccessible to unpaired devices on the LAN."""
    host = request.client.host if request.client else ""
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    local = host in {"127.0.0.1", "::1", "localhost"}
    if request.url.path.startswith("/api/") and not local:
        if address is None or not (address.is_private or address.is_link_local):
            return Response("LAN access only", status_code=403)
        supplied = request.headers.get("X-Family-AI-Key", "")
        if not hmac.compare_digest(supplied, lan_access_token()):
            return Response("Pairing code required", status_code=401)
    return await call_next(request)


@lru_cache
def approval_storage():
    """Approval metadata without constructing models, workers, or the main graph."""
    if settings.database_url:
        return PostgresFamilyStorage(settings.database_url, lazy=True)
    return FamilyStorage(settings.family_ai_data_dir / "family.db")


def database_connection_failed(exc: BaseException) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if type(current).__name__ in {"AdminShutdown", "OperationalError", "InterfaceError", "ConnectionException", "ConnectionTimeout"}:
            return True
        current = current.__cause__ or current.__context__
    return False


def rebuild_agent() -> None:
    with _agent_reset_lock:
        previous = agent()
        if previous.gmail_backup_worker is not None:
            previous.gmail_backup_worker.stop()
        agent.cache_clear()
        if previous.checkpoint_resource is not None:
            try:
                previous.checkpoint_resource.__exit__(None, None, None)
            except Exception:
                pass


@lru_cache
def speech():
    return LocalSpeech(
        settings.whisper_model_path,
        settings.tts_model_path,
        settings.tts_chinese_voice,
        settings.tts_english_voice,
    )


@lru_cache
def voice_turn_agent():
    return VoiceTurnAgent()


@lru_cache
def rag_store():
    if not settings.database_url:
        raise RuntimeError("RAG requires PostgreSQL")
    return RagStore(settings.database_url, settings.ollama_base_url, settings.rag_embedding_model)


@lru_cache
def camera_monitor():
    return CameraMonitor(
        CameraAgent(
            settings.camera_capture_helper,
            settings.ollama_base_url,
            settings.ollama_model,
        ),
        settings.camera_monitor_interval_seconds,
        settings.camera_alert_cooldown_seconds,
        settings.camera_gate_interval_seconds,
    )


@lru_cache
def gmail_backup_agent():
    storage = agent().storage
    return GmailBackupAgent(
        settings.google_oauth_client_id,
        settings.google_oauth_client_secret,
        settings.google_oauth_redirect_uri,
        settings.gmail_backup_root,
        account_registry=storage,
    )


@app.get("/api/gmail/oauth/start")
def gmail_oauth_start(member_id: str = "member", account_hint: str | None = None,
                      allow_send: bool = False):
    try:
        return RedirectResponse(gmail_backup_agent().authorization_url(member_id, account_hint, allow_send))
    except GmailBackupError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/gmail/oauth/callback", response_class=HTMLResponse)
def gmail_oauth_callback(state: str, code: str):
    try:
        email_address = gmail_backup_agent().complete_authorization(state, code)
    except GmailBackupError as exc:
        raise HTTPException(400, str(exc)) from exc
    return HTMLResponse(f"<h1>Gmail connected</h1><p>Account: {email_address}</p><p>You may close this window.</p>")


@app.on_event("startup")
async def start_requested_background_services():
    global _database_reachable
    current = agent()
    if current.gmail_daily_scheduler is not None:
        try:
            current.storage.healthcheck()
        except Exception as exc:
            _database_reachable = False
            logger.warning(
                "Gmail daily scheduler disabled until restart because storage is unavailable: %s",
                type(exc).__name__,
            )
        else:
            _database_reachable = True
            current.gmail_daily_scheduler.start()
            if settings.school_digest_enabled and current.school_daily_scheduler is not None:
                current.school_daily_scheduler.start()
    if settings.camera_monitor_enabled:
        camera_monitor().start()


@app.on_event("shutdown")
def stop_background_services():
    if settings.camera_monitor_enabled:
        camera_monitor().stop()
    if agent.cache_info().currsize:
        current = agent()
        if current.gmail_daily_scheduler is not None:
            current.gmail_daily_scheduler.stop()
        if current.school_daily_scheduler is not None:
            current.school_daily_scheduler.stop()
        if current.gmail_backup_worker is not None:
            current.gmail_backup_worker.stop()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "speech": settings.whisper_model_path.is_dir(),
        "camera_monitor": settings.camera_monitor_enabled,
        "camera_health": camera_monitor().status() if settings.camera_monitor_enabled else {},
    }


@app.get("/api/system/status")
def system_status():
    # The desktop polls this endpoint. Never initialize the graph or connect
    # to PostgreSQL solely to repaint status cards.
    queue = (agent().task_queue.snapshot() if agent.cache_info().currsize
             else {"active": False, "waiting": 0})
    database_status = (
        "postgresql" if settings.database_url and _database_reachable is not False
        else "postgresql-offline" if settings.database_url
        else "sqlite"
    )
    rag_status = {
        "ready": database_status == "postgresql", "documents": 0,
        "model": settings.rag_embedding_model,
    }
    active = active_agent_ids()
    if queue["active"]:
        active.add("builtin-main")
    agents = [dict(item, status="active" if item["id"] in active else "idle")
              for item in BUILTIN_AGENT_STATUS]
    return {
        "status": "online",
        "model": settings.ollama_model,
        "reasoning": True,
        "queue": queue,
        "speech": {
            "ready": settings.whisper_model_path.is_dir(),
            "wake_word": settings.voice_wake_word,
        },
        "camera_monitor": False,
        "camera_mode": "on_demand",
        "agents": agents,
        "database": database_status,
        "rag": rag_status,
    }


@app.get("/api/curator/status")
def curator_status():
    return CuratorAgent(settings.family_ai_data_dir).status()


@app.get("/api/approvals")
def pending_approvals(member_id: str = "member"):
    items = []
    stale = False
    try:
        storage = approval_storage()
        ensure_setup = getattr(storage, "ensure_setup", None)
        if ensure_setup is not None:
            ensure_setup()
        for row in storage.list_agents():
            if row["status"] == "pending":
                items.append({
                    "kind": "agent", "id": row["id"], "title": f"Approve Agent: {row['name']}",
                    "body": f"{row['purpose']} · Permissions: {', '.join(row['permissions']) or 'none'}",
                })
        for row in storage.list_codex_requests(member_id, "pending"):
            items.append({
                "kind": "codex", "id": row["id"], "title": "Approve Codex request",
                "body": f"{row['reason']} · {row['task'][:180]}",
            })
    except Exception:
        stale = True
        with _approval_cache_lock:
            items.extend(_approval_cache.get(member_id, []))
    if not stale:
        with _approval_cache_lock:
            _approval_cache[member_id] = list(items)
    curator = CuratorAgent(settings.family_ai_data_dir)
    for row in curator.status()["pending_proposals"]:
        items.append({
            "kind": "curator", "id": row["id"], "title": "Approve Family AI upgrade",
            "body": row.get("title", "Review a capability upgrade"),
        })
    return {"items": items, "stale": stale}


def _run_codex_request(request: dict) -> None:
    with agent_activity("builtin-codex"):
        task_registry.update(request["id"], "running")
        try:
            result = CodexCLI(settings.codex_cli_path, settings.family_ai_data_dir / "codex-workspace", settings.family_ai_project_dir).run(
                request["task"], request["workspace"])
        except Exception as exc:
            task_registry.update(request["id"], "failed", exc)
            return
        task_registry.update(request["id"], "completed", result=result)


def _run_curator_upgrade(proposal: dict) -> None:
    CodexCLI(settings.codex_cli_path, settings.family_ai_data_dir / "codex-workspace", settings.family_ai_project_dir).run_approved_upgrade(
        proposal["task"]
    )


@app.post("/api/approvals/{kind}/{approval_id}/{decision}")
def decide_approval(kind: str, approval_id: str, decision: str, background: BackgroundTasks, member_id: str = "member"):
    if kind not in {"agent", "codex", "curator"} or decision not in {"approve", "deny"}:
        raise HTTPException(400, "Invalid approval decision")
    approved = decision == "approve"
    storage = approval_storage()
    ensure_setup = getattr(storage, "ensure_setup", None)
    if ensure_setup is not None:
        ensure_setup()
    if kind == "agent":
        ok = storage.decide_agent(approval_id, approved)
    elif kind == "codex":
        if approved:
            request = storage.approve_codex(member_id, approval_id)
            ok = request is not None
            if request:
                task_registry.create(request["id"], "codex", member_id, request["task"])
                background.add_task(_run_codex_request, request)
        else:
            ok = storage.reject_codex(member_id, approval_id)
    else:
        curator = CuratorAgent(settings.family_ai_data_dir)
        if approved:
            proposal = curator.approve_proposal(approval_id)
            ok = proposal is not None
            if proposal:
                background.add_task(_run_curator_upgrade, proposal)
        else:
            ok = curator.deny_proposal(approval_id)
    if not ok:
        raise HTTPException(409, "Approval is no longer pending or does not match")
    return {"status": "accepted", "decision": decision, "kind": kind, "id": approval_id}


@app.get("/api/tasks/status")
def task_status(member_id: str = "member", agent_name: str | None = None):
    return {"active_count": task_registry.active_count(agent_name),
            "latest": task_registry.latest(member_id, agent_name)}


@app.get("/api/tasks/{task_id}")
def task_detail(task_id: str, member_id: str = "member"):
    task = task_registry.get(task_id, member_id)
    if task is None:
        raise HTTPException(404, "Task was not found")
    return task


@app.get("/api/camera/events")
def camera_events(after: int = 0):
    if after < 0:
        raise HTTPException(400, "after must be zero or greater")
    events = camera_monitor().events_after(after) if settings.camera_monitor_enabled else []
    return {"enabled": settings.camera_monitor_enabled, "health": camera_monitor().status() if settings.camera_monitor_enabled else {}, "events": events}


@app.get("/api/knowledge")
def knowledge_documents(member_id: str = "member"):
    return {"items": rag_store().list_documents(member_id), "embedding_model": settings.rag_embedding_model}


@app.post("/api/knowledge/upload")
async def knowledge_upload(file: UploadFile, member_id: str = "member", scope: str = "private"):
    if scope not in {"private", "family"}:
        raise HTTPException(400, "scope must be private or family")
    data = await file.read()
    if len(data) > 25 * 1024 * 1024:
        raise HTTPException(413, "File is larger than 25 MB")
    try:
        return await asyncio.to_thread(rag_store().ingest, member_id, scope, file.filename or "document.txt", file.content_type or "application/octet-stream", data)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/api/knowledge/{document_id}")
def knowledge_delete(document_id: str, member_id: str = "member"):
    if not rag_store().delete_document(member_id, document_id):
        raise HTTPException(404, "Document not found")
    return {"status": "deleted"}


@app.post("/api/speech/transcribe")
async def transcribe(audio: UploadFile):
    data = await audio.read()
    if len(data) > 16 * 1024 * 1024:
        raise HTTPException(413, "The recording is too large.")
    try:
        text = await asyncio.to_thread(speech().transcribe_wav, data)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Local speech recognition failed: {type(exc).__name__}") from exc
    if not text:
        raise HTTPException(422, "No speech was recognized.")
    return {"text": text}


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=3000)


class VoiceRouteRequest(BaseModel):
    transcript: str = Field(min_length=1, max_length=1000)
    follow_up_active: bool = False


@app.post("/api/speech/route")
def route_voice_turn(request: VoiceRouteRequest):
    decision = voice_turn_agent().decide(request.transcript, request.follow_up_active)
    return {
        "should_respond": decision.should_respond,
        "message": decision.message,
        "reason": decision.reason,
        "agent": voice_turn_agent().name,
    }


@app.post("/api/speech/synthesize")
async def synthesize(request: SpeechRequest):
    try:
        audio = await asyncio.to_thread(speech().synthesize, request.text)
    except Exception as exc:
        raise HTTPException(500, f"Local speech synthesis failed: {type(exc).__name__}") from exc
    return Response(audio, media_type="audio/wav")


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        reply = agent().invoke(request.member_id, request.thread_id, request.message)
    except Exception as exc:
        if not settings.database_url or not database_connection_failed(exc):
            raise
        rebuild_agent()
        reply = agent().invoke(request.member_id, request.thread_id, request.message)
    return ChatResponse(reply=reply)


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    async def events():
        is_weather = bool(re.search(r"(天气|气温|温度|降水|下雨|风力|预报|weather|forecast)", request.message, re.I))
        is_web = request.message.startswith("/web ") or bool(re.search(r"(最新|新闻|实时|推荐|餐馆|公园|latest|news|recommend)", request.message, re.I))
        is_codex = request.message.startswith("/codex")
        family_agent = agent()
        ticket = family_agent.reserve()
        initial_position = family_agent.queue_position(ticket)
        first_status = "Request received. Identifying the right tool…" if is_weather or is_web else "Request received. Loading conversation and family memory…"
        if initial_position > 1:
            first_status = f"Queued. {initial_position - 1} model task(s) ahead…"
        yield json.dumps({"type": "status", "message": first_status}, ensure_ascii=False) + "\n"
        task = asyncio.create_task(
            asyncio.to_thread(
                family_agent.execute_reserved,
                ticket,
                request.member_id,
                request.thread_id,
                request.message,
            )
        )
        seconds = 0
        while not task.done():
            await asyncio.sleep(1)
            seconds += 1
            position = family_agent.queue_position(ticket)
            if position > 0:
                message = f"Queued. {position - 1} model task(s) ahead…"
            elif is_codex:
                message = f"Codex specialist is processing the read-only task… {seconds}s"
            elif is_weather:
                message = f"Connecting to the live weather service… {seconds}s"
            elif is_web:
                message = f"Searching the web and checking sources… {seconds}s"
            elif seconds < 3:
                message = "Preparing context…"
            elif seconds < 8:
                message = "The local model is reasoning…"
            else:
                message = f"The local model is still working… {seconds}s"
            yield json.dumps({"type": "status", "message": message}, ensure_ascii=False) + "\n"
        try:
            reply = await task
            yield json.dumps({"type": "answer", "reply": reply}, ensure_ascii=False) + "\n"
        except Exception as exc:
            if settings.database_url and database_connection_failed(exc):
                yield json.dumps({"type": "status", "message": "Database connection changed. Reconnecting…"}) + "\n"
                try:
                    rebuild_agent()
                    reply = await asyncio.to_thread(agent().invoke, request.member_id, request.thread_id, request.message)
                    yield json.dumps({"type": "answer", "reply": reply}, ensure_ascii=False) + "\n"
                    return
                except Exception as retry_exc:
                    exc = retry_exc
            yield json.dumps({"type": "error", "message": f"Processing failed: {type(exc).__name__}"}, ensure_ascii=False) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(INDEX_HTML)


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#17251e"><title>Hearth · Family AI</title><style>
*{box-sizing:border-box}:root{font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","PingFang SC",sans-serif;color:#17251e;background:#e9ede7}body{margin:0;min-height:100vh;background:radial-gradient(circle at 8% 5%,#f7e8ca 0,transparent 27%),radial-gradient(circle at 92% 90%,#cce0d1 0,transparent 30%),#edf0eb;display:grid;place-items:center;padding:24px}.shell{width:min(1180px,100%);height:min(850px,calc(100vh - 48px));background:#fbfaf6;border:1px solid #ffffffaa;border-radius:30px;box-shadow:0 30px 90px #24362b24;display:grid;grid-template-columns:270px minmax(0,1fr);overflow:hidden}.side{background:#17251e;color:#eef3eb;padding:28px 22px;display:flex;flex-direction:column}.brand{display:flex;align-items:center;gap:12px;margin-bottom:36px}.mark{width:42px;height:42px;display:grid;place-items:center;border-radius:14px;background:#e7b967;color:#17251e;font-size:22px;box-shadow:inset 0 1px #fff8}.brand b{font-size:20px;letter-spacing:-.4px}.brand small{display:block;color:#92a299;margin-top:2px}.label{font-size:11px;text-transform:uppercase;letter-spacing:1.5px;color:#82948a;margin:18px 10px 9px}.member-card{background:#22352b;border:1px solid #ffffff0f;border-radius:16px;padding:12px;display:flex;align-items:center;gap:11px}.avatar{width:38px;height:38px;border-radius:12px;background:#cfe0ce;color:#1d3326;display:grid;place-items:center;font-weight:700;text-transform:uppercase}.member-card input{min-width:0;width:100%;border:0;background:transparent;color:#fff;font:inherit;font-weight:600;outline:none}.nav{display:grid;gap:7px}.quick{border:0;color:#c5d0ca;background:transparent;padding:11px 12px;border-radius:12px;text-align:left;font:inherit;cursor:pointer;transition:.18s}.quick:hover{background:#2b4035;color:#fff;transform:translateX(2px)}.quick span{margin-right:10px}.privacy{margin-top:auto;background:#203229;border-radius:15px;padding:13px;color:#aab9b1;font-size:12px;line-height:1.55}.privacy strong{display:block;color:#dce6df;margin-bottom:3px}.main{min-width:0;min-height:0;overflow:hidden;display:flex;flex-direction:column}.top{height:88px;flex:0 0 88px;padding:0 32px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #e7e7df}.top h1{font-size:19px;margin:0}.top p{font-size:12px;color:#7d877f;margin:4px 0 0}.online{font-size:12px;color:#467254;background:#e5efe5;padding:8px 12px;border-radius:999px}.online:before{content:"";display:inline-block;width:7px;height:7px;background:#52a36a;border-radius:50%;margin-right:7px;box-shadow:0 0 0 4px #52a36a1c}.chat{flex:1 1 auto;min-height:0;overflow-y:auto;overscroll-behavior:contain;padding:30px clamp(20px,5vw,64px);display:flex;flex-direction:column;gap:18px;scroll-behavior:smooth}.welcome{background:linear-gradient(135deg,#f2ead8,#edf2e9);border:1px solid #e6dfce;border-radius:22px;padding:22px 24px;margin-bottom:4px}.welcome h2{font-size:22px;margin:0 0 7px}.welcome p{margin:0;color:#667269;line-height:1.55;font-size:14px}.msg{max-width:min(78%,680px);padding:14px 17px;border-radius:18px;white-space:pre-wrap;line-height:1.6;font-size:15px;animation:rise .24s ease-out;overflow-wrap:anywhere}.me{align-self:flex-end;background:#234534;color:#fff;border-bottom-right-radius:5px;box-shadow:0 8px 22px #23453420}.ai{align-self:flex-start;background:#fff;border:1px solid #e3e4dc;border-bottom-left-radius:5px;box-shadow:0 8px 24px #3642390d}.status{align-self:flex-start;color:#647168;background:#f3f4ef;border:1px solid #e4e6df;font-size:13px}.pulse:before{content:"";display:inline-block;width:8px;height:8px;background:#d69c43;border-radius:50%;margin-right:9px;animation:pulse .8s infinite alternate}.composer-wrap{flex:0 0 auto;padding:12px 28px 24px;background:#fbfaf6;border-top:1px solid #ecece5}.composer{max-width:800px;margin:auto;display:flex;align-items:center;gap:9px;padding:8px 8px 8px 10px;border:1px solid #d9dcd4;background:#fff;border-radius:20px;box-shadow:0 12px 35px #26332912}.composer:focus-within{border-color:#89a18e;box-shadow:0 12px 35px #26332916,0 0 0 3px #6f92751a}.composer input{min-width:0;flex:1;border:0;outline:0;padding:10px 0;font:inherit;background:transparent}.send,.mic,.speaker{flex:0 0 44px;border:0;width:44px;height:44px;border-radius:14px;cursor:pointer;transition:.18s}.send{background:#234534;color:#fff;font-size:20px}.mic{background:#d9a64e;color:#17251e;font-size:21px}.speaker{background:#edf1eb;color:#4b5d52;font-size:18px}.speaker.active{background:#dcebdd;color:#27613b}.mic.recording{background:#c75548;color:#fff;animation:pulse .65s infinite alternate}.send:hover,.mic:hover,.speaker:hover{transform:translateY(-1px)}.send:disabled,.mic:disabled{opacity:.55;cursor:wait}.hint{text-align:center;color:#7f8982;font-size:11px;margin-top:8px}@keyframes pulse{to{opacity:.55;transform:scale(.94)}}@keyframes rise{from{opacity:0;transform:translateY(7px)}}
@media(max-width:760px){body{padding:0;min-height:100dvh}.shell{height:100vh;height:100dvh;border-radius:0;grid-template-columns:minmax(0,1fr)}.side{display:none}.top{height:72px;flex-basis:72px;padding:0 18px}.chat{padding:20px 16px}.msg{max-width:90%}.composer-wrap{padding:10px 12px calc(12px + env(safe-area-inset-bottom))}.welcome{padding:18px}.online{font-size:0}.online:before{margin:0}}
</style></head><body><main class="shell"><aside class="side"><div class="brand"><div class="mark">⌂</div><div><b>Hearth</b><small>Family intelligence</small></div></div><div class="label">Current member</div><div class="member-card"><div class="avatar" id="avatar">A</div><input id="member" value="*" aria-label="Family member"></div><div class="label">Quick actions</div><nav class="nav"><button class="quick" data-prompt="/memory"><span>◉</span>Family memory</button><button class="quick" data-prompt="/todo list"><span>✓</span>Family todos</button><button class="quick" data-prompt="/web "><span>⌕</span>Live web search</button><button class="quick" data-prompt="/codex "><span>◇</span>Codex specialist</button><button class="quick" data-prompt="/agents"><span>⚙</span>Agents & permissions</button><button class="quick" data-prompt="/help"><span>?</span>Help</button></nav><div class="privacy"><strong>⌾ Local hands-free voice</strong>The * microphone is processed locally. A request is submitted only after the wake word “*.”</div></aside><section class="main"><header class="top"><div><h1>Family Voice Assistant</h1><p>Hands-free * wake word · Natural neural voice · Single-model queue</p></div><div class="online">Listening for *</div></header><section class="chat" id="chat"><div class="welcome"><h2>Hello, *</h2><p>Say “*” followed by your request—no click required. Grant microphone permission once when prompted.</p></div><div class="msg ai">Hands-free listening is ready. Try: “*, what is the weather in * tomorrow?”</div></section><div class="composer-wrap"><form class="composer" id="form"><button type="button" class="mic" id="mic" aria-label="Hands-free listening status">●</button><input id="message" autocomplete="off" placeholder="Say “*…” or type a message…"><button type="button" class="speaker active" id="speaker" aria-label="Automatic speech is on">🔊</button><button class="send" aria-label="Send">↑</button></form><div class="hint" id="hint">Starting local hands-free listening…</div></div></section></main>
<script>
const onMac=location.hostname==='127.0.0.1'||location.hostname==='localhost';
const originalFetch=window.fetch.bind(window);
window.fetch=(input,init={})=>{if(!onMac){const headers=new Headers(init.headers||{});let key=localStorage.getItem('family-ai-lan-key')||'';if(!key){key=prompt('请输入 Mac 上显示的 Family AI 局域网访问码')||'';if(key)localStorage.setItem('family-ai-lan-key',key.trim())}headers.set('X-Family-AI-Key',key.trim());init={...init,headers}}return originalFetch(input,init).then(response=>{if(response.status===401&&!onMac){localStorage.removeItem('family-ai-lan-key');hint&&(hint.textContent='访问码不正确，请刷新页面后重试。')}return response})};
const form=document.querySelector('#form'),chat=document.querySelector('#chat'),message=document.querySelector('#message'),button=form.querySelector('.send'),mic=document.querySelector('#mic'),speaker=document.querySelector('#speaker'),hint=document.querySelector('#hint'),member=document.querySelector('#member'),avatar=document.querySelector('#avatar');
let wakeEnabled=false,wakePaused=false,speaking=false,transcribing=false,audioContext,stream,processor,preRoll=[],utterance=[],silenceFrames=0,voiceFrames=0,noiseFloor=.004,currentAudio,playbackId=0,resumeTimer,followUpUntil=0,autoSpeak=localStorage.getItem('family-auto-speak')!=='false',cameraEventId=0,cameraEventsReady=false;
speaker.classList.toggle('active',autoSpeak);speaker.textContent=autoSpeak?'🔊':'🔇';
member.value=localStorage.getItem('family-member')||'*';
function memberUI(){avatar.textContent=(member.value[0]||'?').toUpperCase();localStorage.setItem('family-member',member.value)}
memberUI();member.oninput=memberUI;
document.querySelectorAll('.quick').forEach(x=>x.onclick=()=>{message.value=x.dataset.prompt;message.focus()});
function add(t,c){const d=document.createElement('div');d.className='msg '+c;d.textContent=t;chat.append(d);chat.scrollTop=chat.scrollHeight;return d}
function wavBlob(samples,rate){const b=new ArrayBuffer(44+samples.length*2),v=new DataView(b);const s=(o,t)=>[...t].forEach((c,i)=>v.setUint8(o+i,c.charCodeAt(0)));s(0,'RIFF');v.setUint32(4,36+samples.length*2,true);s(8,'WAVE');s(12,'fmt ');v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);v.setUint32(24,rate,true);v.setUint32(28,rate*2,true);v.setUint16(32,2,true);v.setUint16(34,16,true);s(36,'data');v.setUint32(40,samples.length*2,true);for(let i=0;i<samples.length;i++){const x=Math.max(-1,Math.min(1,samples[i]));v.setInt16(44+i*2,x<0?x*32768:x*32767,true)}return new Blob([b],{type:'audio/wav'})}
async function processUtterance(parts,rate){if(transcribing||wakePaused)return;transcribing=true;hint.textContent='Speech detected. Voice Turn Agent is checking…';const total=parts.reduce((n,x)=>n+x.length,0),all=new Float32Array(total);let offset=0;for(const x of parts){all.set(x,offset);offset+=x.length}try{const data=new FormData();data.append('audio',wavBlob(all,rate),'voice.wav');const r=await fetch('/api/speech/transcribe',{method:'POST',body:data});if(r.status===422){hint.textContent=Date.now()<followUpUntil?'Follow-up mode: ask your next question…':'Listening for the wake word *…';return}if(!r.ok)throw new Error((await r.json()).detail||'Recognition failed');const result=await r.json();const routed=await fetch('/api/speech/route',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({transcript:result.text,follow_up_active:Date.now()<followUpUntil})});if(!routed.ok)throw new Error('Voice routing failed');const decision=await routed.json();if(!decision.should_respond){hint.textContent=decision.reason==='wake_only'?'* is awake. Please say your request…':(Date.now()<followUpUntil?'Follow-up mode: ask your next question…':'Listening for the wake word *…');return}followUpUntil=0;message.value=decision.message;hint.textContent='Voice Turn Agent passed this to the Main Agent: '+decision.message;form.requestSubmit();}catch(e){hint.textContent='Microphone connection issue. Listening will retry automatically…';}finally{transcribing=false}}
function resetVoiceCapture(){speaking=false;utterance=[];preRoll=[];silenceFrames=0;voiceFrames=0}
async function restoreWakeListening(id=playbackId){if(id!==playbackId)return;clearTimeout(resumeTimer);resetVoiceCapture();if(!wakeEnabled)return;try{if(audioContext?.state==='suspended')await audioContext.resume();stream?.getAudioTracks().forEach(track=>{track.enabled=true});wakePaused=false;hint.textContent=Date.now()<followUpUntil?'Follow-up mode: ask your next question…':'Listening for the wake word *…';}catch(e){wakePaused=false;hint.textContent='Microphone recovery failed. Click the microphone once to reconnect.'}}
async function startWakeListening(){if(wakeEnabled)return;try{stream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true,noiseSuppression:true,autoGainControl:true}});audioContext=new AudioContext();if(audioContext.state==='suspended')await audioContext.resume();const source=audioContext.createMediaStreamSource(stream);processor=audioContext.createScriptProcessor(4096,1,1);const silent=audioContext.createGain();silent.gain.value=0;processor.onaudioprocess=e=>{if(wakePaused||transcribing)return;const chunk=new Float32Array(e.inputBuffer.getChannelData(0));let power=0;for(const value of chunk)power+=value*value;const rms=Math.sqrt(power/chunk.length);const startLevel=Math.max(.010,noiseFloor*2.8),stopLevel=Math.max(.006,noiseFloor*1.5);preRoll.push(chunk);if(preRoll.length>10)preRoll.shift();if(!speaking){noiseFloor=noiseFloor*.96+Math.min(rms,.02)*.04;if(rms>startLevel){speaking=true;utterance=preRoll.slice();silenceFrames=0;voiceFrames=1}}if(speaking){utterance.push(chunk);if(rms>startLevel)voiceFrames++;silenceFrames=rms<stopLevel?silenceFrames+1:0;const tooLong=utterance.length*4096/audioContext.sampleRate>20;if(silenceFrames>10||tooLong){const captured=utterance,hasVoice=voiceFrames>=3;speaking=false;utterance=[];silenceFrames=0;voiceFrames=0;if(hasVoice&&captured.length>7)processUtterance(captured,audioContext.sampleRate)}}};source.connect(processor);processor.connect(silent);silent.connect(audioContext.destination);wakeEnabled=true;resetVoiceCapture();mic.classList.add('recording');mic.textContent='◉';mic.setAttribute('aria-label','Hands-free listening is on');hint.textContent='Hands-free mode: listening for *…';}catch(e){wakeEnabled=false;mic.classList.remove('recording');mic.textContent='●';hint.textContent='Click the microphone once to grant permission. Listening will be automatic afterward.'}}
async function stopWakeListening(){wakeEnabled=false;wakePaused=false;speaking=false;processor?.disconnect();stream?.getTracks().forEach(t=>t.stop());if(audioContext)await audioContext.close();mic.classList.remove('recording');mic.textContent='●';hint.textContent='Hands-free listening is off. Click the microphone to turn it back on.'}
mic.onclick=()=>wakeEnabled?stopWakeListening():startWakeListening();
speaker.onclick=()=>{autoSpeak=!autoSpeak;localStorage.setItem('family-auto-speak',autoSpeak);speaker.classList.toggle('active',autoSpeak);speaker.textContent=autoSpeak?'🔊':'🔇';speaker.setAttribute('aria-label',autoSpeak?'Automatic speech is on':'Automatic speech is off');if(!autoSpeak&&currentAudio){playbackId++;currentAudio.pause();currentAudio=null;restoreWakeListening()}};
async function speakText(text){if(!autoSpeak){followUpUntil=Date.now()+45000;restoreWakeListening();return}const id=++playbackId;wakePaused=true;resetVoiceCapture();try{if(currentAudio)currentAudio.pause();hint.textContent='Generating natural speech…';const r=await fetch('/api/speech/synthesize',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({text})});if(!r.ok)throw new Error('Speech generation failed');const url=URL.createObjectURL(await r.blob());currentAudio=new Audio(url);let finished=false;const finish=()=>{if(finished||id!==playbackId)return;finished=true;URL.revokeObjectURL(url);currentAudio=null;followUpUntil=Date.now()+45000;hint.textContent='Restoring the microphone…';clearTimeout(resumeTimer);resumeTimer=setTimeout(()=>restoreWakeListening(id),350)};currentAudio.onended=finish;currentAudio.onerror=finish;currentAudio.onabort=finish;await currentAudio.play();hint.textContent='Playing continuous natural speech through *…';}catch(e){currentAudio=null;followUpUntil=Date.now()+45000;await restoreWakeListening(id);hint.textContent='The answer is displayed. Follow-up mode is ready.'}}
form.onsubmit=async e=>{e.preventDefault();const text=message.value.trim();if(!text)return;wakePaused=true;add(text,'me');message.value='';button.disabled=true;button.textContent='…';const status=add('Submitted. Loading context and family memory…','status pulse');try{const r=await fetch('/api/chat/stream',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({member_id:member.value,thread_id:'home',message:text})});if(!r.ok)throw new Error('HTTP '+r.status);const reader=r.body.getReader(),decoder=new TextDecoder();let buffer='';while(true){const {value,done}=await reader.read();if(done)break;buffer+=decoder.decode(value,{stream:true});const lines=buffer.split('\n');buffer=lines.pop();for(const line of lines){if(!line)continue;const event=JSON.parse(line);if(event.type==='status')status.textContent=event.message;else if(event.type==='answer'){status.remove();const answer=event.reply||'The model did not return a final answer.';add(answer,'ai');speakText(answer)}else if(event.type==='error'){status.classList.remove('pulse');status.textContent=event.message;wakePaused=false}}}}catch(e){status.classList.remove('pulse');status.textContent='Connection failed: '+e;wakePaused=false}finally{button.disabled=false;button.textContent='↑';if(!autoSpeak){wakePaused=false;hint.textContent='Listening for the wake word *…'}message.focus()}};
async function pollCameraEvents(){if(wakePaused||transcribing||currentAudio)return;try{const r=await fetch('/api/camera/events?after='+cameraEventId);if(!r.ok)return;const data=await r.json(),events=data.events||[];if(!cameraEventsReady){if(events.length)cameraEventId=events[events.length-1].id;cameraEventsReady=true;return}if(events.length){const event=events[0];cameraEventId=event.id;const alert=event.message_zh||event.message;add(alert,'ai');await speakText(alert)}}catch(e){}}
setInterval(pollCameraEvents,3000);pollCameraEvents();
startWakeListening();
</script></body></html>"""
