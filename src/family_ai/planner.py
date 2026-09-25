from datetime import date
from enum import Enum
import re
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from .config import settings


class ChatPlan(BaseModel):
    tool: Literal["chat"]


class WeatherPlan(BaseModel):
    tool: Literal["weather"]
    location: str = Field(description="Canonical city/place only, e.g. *")
    start_day_offset: int = Field(description="0 today, 1 tomorrow, 2 day after tomorrow", ge=0, le=15)
    days: int = Field(description="Exact requested forecast day count", ge=1, le=10)


class WebPlan(BaseModel):
    tool: Literal["web_search"]
    query: str
    result_count: int = Field(
        default=5,
        description="Exact number of recommendations/results requested; default 5",
        ge=1,
        le=10,
    )


class SearchStrategy(BaseModel):
    queries: list[str] = Field(
        description="Two or three meaningfully different web searches",
        min_length=2,
        max_length=3,
    )


class TimePlan(BaseModel):
    tool: Literal["current_time"]


class MemoryListPlan(BaseModel):
    tool: Literal["memory_list"]


class MemoryAddPlan(BaseModel):
    tool: Literal["memory_add"]
    scope: Literal["private", "family"]
    content: str


class TodoListPlan(BaseModel):
    tool: Literal["todo_list"]


class TodoAddPlan(BaseModel):
    tool: Literal["todo_add"]
    owner: Literal["me", "family"]
    content: str


class TodoCompletePlan(BaseModel):
    tool: Literal["todo_complete"]
    todo_id: str


class HelpPlan(BaseModel):
    tool: Literal["help"]


AgentPermission = Literal[
    "network_public", "family_memory_read", "database_read", "database_write",
    "files_read", "files_write", "email", "calendar", "remote_control",
]


class AgentListPlan(BaseModel):
    tool: Literal["agent_list"]


class CuratorStatusPlan(BaseModel):
    tool: Literal["curator_status"]


class CuratorAuditPlan(BaseModel):
    tool: Literal["curator_audit"]


class CuratorApprovePlan(BaseModel):
    tool: Literal["curator_approve"]
    proposal_id: str


class AgentProposePlan(BaseModel):
    tool: Literal["agent_propose"]
    name: str
    purpose: str
    permissions: list[AgentPermission]


class AgentApprovePlan(BaseModel):
    tool: Literal["agent_approve"]
    proposal_id: str


class AgentRejectPlan(BaseModel):
    tool: Literal["agent_reject"]
    proposal_id: str


class CodexRunPlan(BaseModel):
    tool: Literal["codex_run"]
    task: str
    workspace: Literal["isolated", "family_project"] = "isolated"


class CodexProposePlan(BaseModel):
    tool: Literal["codex_propose"]
    task: str
    reason: str
    workspace: Literal["isolated", "family_project"] = "isolated"


class CodexApprovePlan(BaseModel):
    tool: Literal["codex_approve"]
    request_id: str


class CodexRejectPlan(BaseModel):
    tool: Literal["codex_reject"]
    request_id: str


class ClarifyPlan(BaseModel):
    tool: Literal["clarify"]
    clarification: str


class FinancialVideoPlan(BaseModel):
    tool: Literal["financial_video_create"]
    request: str
    provider: Literal["ollama", "codex"] = "ollama"


class DelegateTaskPlan(BaseModel):
    tool: Literal["delegate_task"]
    task: str


class CameraViewPlan(BaseModel):
    tool: Literal["camera_view"]
    provider: Literal["*"]
    question: str = Field(description="What the user wants checked in the current camera view")


class GmailBackupPlan(BaseModel):
    tool: Literal["gmail_backup"]
    action: Literal["connect", "backup", "export_disk", "status", "disconnect", "pause", "resume", "cancel"]
    account: str | None = Field(
        default=None,
        description="Specific Gmail address, '*' for all connected accounts, or null",
    )


class CalendarPlan(BaseModel):
    tool: Literal["calendar"]
    action: Literal["upcoming"] = "upcoming"
    account: str | None = None
    days: int = Field(default=7, ge=1, le=31)


class SchoolEmailPlan(BaseModel):
    tool: Literal["school_email"]
    action: Literal["query", "digest", "status"] = "query"
    query: str = ""


class SystemDiagnosticsPlan(BaseModel):
    tool: Literal["system_diagnostics"]
    target: Literal["all", "database", "gmail", "drive", "ollama", "codex"] = "all"


class ToolName(str, Enum):
    chat = "chat"
    weather = "weather"
    web_search = "web_search"
    current_time = "current_time"
    memory_list = "memory_list"
    memory_add = "memory_add"
    todo_list = "todo_list"
    todo_add = "todo_add"
    todo_complete = "todo_complete"
    help = "help"
    agent_list = "agent_list"
    agent_propose = "agent_propose"
    agent_approve = "agent_approve"
    agent_reject = "agent_reject"
    codex_run = "codex_run"
    codex_propose = "codex_propose"
    codex_approve = "codex_approve"
    codex_reject = "codex_reject"
    clarify = "clarify"
    camera_view = "camera_view"
    curator_status = "curator_status"
    curator_audit = "curator_audit"
    curator_approve = "curator_approve"
    financial_video_create = "financial_video_create"
    delegate_task = "delegate_task"
    gmail_backup = "gmail_backup"
    calendar = "calendar"
    school_email = "school_email"
    system_diagnostics = "system_diagnostics"


class IntentDecision(BaseModel):
    tool: ToolName = Field(description="Single best tool for the current user request")


PLANNER_PROMPT = """You are the tool planner for a private family assistant.
Understand natural Chinese, English, mixed-language wording, typos, and conversational context.
Choose exactly one tool; never answer the user directly. Parameter extraction happens later.

Tool policy:
- weather: any current/forecast weather request. Extract canonical location, start offset, and requested number of days.
- web_search: information that can change, including news, prices, schedules, recommendations, or explicit web search.
- current_time: current date, weekday, or time.
- memory_add: user explicitly asks to remember/save a fact. Default scope=private unless they say family/shared/everyone.
- memory_list: user asks only to display/list what is remembered.
- If the user asks to USE remembered preferences to plan, recommend, write, analyze, or answer something, choose chat, not memory_list.
- todo_add/list/complete: family or personal tasks.
- help: asks what the assistant can do or sends /help.
- agent_list: asks to list existing agents or pending proposals.
- agent_propose: explicitly asks to create/configure a specialist agent, or requests a recurring unsupported capability requiring new permissions. Request only minimum permissions.
- agent_approve/agent_reject: user explicitly approves or rejects a proposal ID. Never infer approval from vague agreement.
- Ordinary one-off questions use chat or existing tools; do not propose an agent merely because specialization could help.
- No agent can grant itself permissions. New capabilities remain pending until explicit approval.
- codex_run: only when the user explicitly uses /codex; this explicit command authorizes sending that task to Codex in read-only mode.
- codex_propose: a task is unusually complex and likely beyond the local model, especially deep code/repository analysis. Explain why, but do not call Codex yet.
- codex_approve/codex_reject: explicit decision for a pending Codex request ID. Never infer approval.
- Prefer local chat for normal questions. Never send memories, credentials, database contents, or prior private conversation to Codex.
- chat: stable knowledge, reasoning, writing, planning, or ordinary conversation.
- clarify: required tool argument is truly missing. Put the question in clarification.
- camera_view: user asks to look at, inspect, or describe a home camera image, or asks whether someone is visible. Choose * or * from the request/context.
- financial_video_create: user asks to make, create, produce, or generate a financial education video. Default provider is Ollama. Use Codex only when the user explicitly names Codex for this video task.
- delegate_task: a concrete task needs a specialist and no listed tool fits. Main Agent will ask Codex to define the missing specialist and finish the original task. Do not use for ordinary questions or casual conversation.
- gmail_backup: connect, run, inspect, or disconnect the read-only local Gmail backup. Never use it to send, delete, or modify email.
- school_email: search and summarize School school email already stored in PostgreSQL, show status, or send the configured School digest.
- calendar: read upcoming Google Calendar events. It is read-only; never create, edit, or delete events.
- system_diagnostics: inspect Family AI database, Gmail worker, backup drive, or Ollama health. Return sanitized health only, never private content or credentials.

Interpret date ranges semantically:
- “未来3天 * 天气” => weather, *, offset 0, days 3.
- “明天*天气” => weather, *, offset 1, days 1.
- “从明天起3天” => weather, offset 1, days 3.
- “周末适合野餐吗” with a known location in recent context => weather using that location and relevant days.
- “根据你记得的我的饮食偏好设计早餐” => chat.
- “你记得我什么” => memory_list.

Do not put private family memories into a web query. Today is {today}."""


class ModelPlanner:
    def __init__(self, model: str, base_url: str):
        llm = ChatOllama(
            model=model,
            base_url=base_url,
            reasoning=False,
            temperature=0,
            num_ctx=settings.ollama_planner_context,
            num_predict=settings.ollama_planner_max_tokens,
            client_kwargs={"timeout": 120.0},
            validate_model_on_init=False,
        )
        self.classifier = llm.with_structured_output(IntentDecision, method="json_schema")
        self.search_strategist = llm.with_structured_output(SearchStrategy, method="json_schema")
        self.extractors = {
            ToolName.weather: llm.with_structured_output(WeatherPlan, method="json_schema"),
            ToolName.web_search: llm.with_structured_output(WebPlan, method="json_schema"),
            ToolName.memory_add: llm.with_structured_output(MemoryAddPlan, method="json_schema"),
            ToolName.todo_add: llm.with_structured_output(TodoAddPlan, method="json_schema"),
            ToolName.todo_complete: llm.with_structured_output(TodoCompletePlan, method="json_schema"),
            ToolName.clarify: llm.with_structured_output(ClarifyPlan, method="json_schema"),
            ToolName.agent_propose: llm.with_structured_output(AgentProposePlan, method="json_schema"),
            ToolName.agent_approve: llm.with_structured_output(AgentApprovePlan, method="json_schema"),
            ToolName.agent_reject: llm.with_structured_output(AgentRejectPlan, method="json_schema"),
            ToolName.codex_propose: llm.with_structured_output(CodexProposePlan, method="json_schema"),
            ToolName.codex_approve: llm.with_structured_output(CodexApprovePlan, method="json_schema"),
            ToolName.codex_reject: llm.with_structured_output(CodexRejectPlan, method="json_schema"),
            ToolName.camera_view: llm.with_structured_output(CameraViewPlan, method="json_schema"),
            ToolName.delegate_task: llm.with_structured_output(DelegateTaskPlan, method="json_schema"),
            ToolName.gmail_backup: llm.with_structured_output(GmailBackupPlan, method="json_schema"),
            ToolName.calendar: llm.with_structured_output(CalendarPlan, method="json_schema"),
            ToolName.school_email: llm.with_structured_output(SchoolEmailPlan, method="json_schema"),
            ToolName.system_diagnostics: llm.with_structured_output(SystemDiagnosticsPlan, method="json_schema"),
        }

    def plan(self, text: str, recent_context: str = ""):
        stripped = text.strip()
        school_named = bool(re.search(r"(?i)school", stripped))
        if school_named:
            if re.search(r"(?i)状态|配置|status|启用", stripped):
                return SchoolEmailPlan(tool="school_email", action="status")
            if re.search(r"(?i)发送|日报|汇总表|更新邮件|digest|send", stripped):
                return SchoolEmailPlan(tool="school_email", action="digest", query=stripped)
            return SchoolEmailPlan(tool="school_email", action="query", query=stripped)
        if stripped == "/help":
            return HelpPlan(tool="help")
        if stripped == "/memory":
            return MemoryListPlan(tool="memory_list")
        if re.fullmatch(r"/todo\s+list", stripped, re.I):
            return TodoListPlan(tool="todo_list")
        match = re.fullmatch(r"/todo\s+add\s+(me|family)\s+(.+)", stripped, re.I | re.S)
        if match:
            return TodoAddPlan(
                tool="todo_add", owner=match.group(1).lower(), content=match.group(2).strip()
            )
        match = re.fullmatch(r"/todo\s+done\s+([\w-]+)", stripped, re.I)
        if match:
            return TodoCompletePlan(tool="todo_complete", todo_id=match.group(1))
        match = re.fullmatch(r"/remember\s+(private|family)\s+(.+)", stripped, re.I | re.S)
        if match:
            return MemoryAddPlan(
                tool="memory_add", scope=match.group(1).lower(), content=match.group(2).strip()
            )
        gmail_named = bool(re.search(r"(?i)gmail|e-?mail|谷歌邮箱|邮件备份|邮件", stripped))
        if re.search(r"(?i)calendar|日历|日程|行程|安排", stripped):
            address = re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", stripped, re.I)
            number = re.search(r"(?i)(\d+)\s*(?:天|days?)", stripped)
            days = int(number.group(1)) if number else 7
            return CalendarPlan(tool="calendar", action="upcoming",
                account=(address.group(0).casefold() if address else
                         "*" if re.search(r"(?i)所有|全部|两个|all|both", stripped) else None),
                days=max(1, min(days, 31)))
        if gmail_named and re.search(r"(?i)状态|情况|进度|status|progress", stripped):
            address = re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", stripped, re.I)
            return GmailBackupPlan(tool="gmail_backup", action="status",
                account=address.group(0).casefold() if address else "*")
        if re.search(r"(?i)codex", stripped) and re.search(
            r"(?i)在运行|运行吗|执行中|状态|进度|running|status|progress", stripped):
            return SystemDiagnosticsPlan(tool="system_diagnostics", target="codex")
        if re.search(r"(?i)(检查|诊断|健康检查|check|diagnose).*(数据库|postgres|gmail|硬盘|ollama|family ai|系统)", stripped):
            if re.search(r"(?i)数据库|postgres", stripped):
                target = "database"
            elif re.search(r"(?i)gmail", stripped):
                target = "gmail"
            elif re.search(r"(?i)硬盘|drive|disk", stripped):
                target = "drive"
            elif re.search(r"(?i)ollama|模型", stripped):
                target = "ollama"
            else:
                target = "all"
            return SystemDiagnosticsPlan(tool="system_diagnostics", target=target)
        # A bare progress question in chat refers to the assistant's active
        # background job. Gmail backup is currently the only user-visible
        # background workflow, so inspect every connected account instead of
        # accidentally asking the model to start another backup.
        if re.fullmatch(
            r"(?i)\s*(?:请)?(?:查看|查询|看(?:一下)?|显示)?\s*(?:当前|现在)?\s*"
            r"(?:备份)?进度(?:怎么样|如何|呢|是多少|情况)?[？?。！!]*\s*",
            stripped,
        ):
            return GmailBackupPlan(tool="gmail_backup", action="status", account="*")
        if gmail_named:
            address = re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", stripped, re.I)
            account = address.group(0).casefold() if address else (
                "*" if re.search(r"(?i)所有|全部|两个|all|both", stripped) else None
            )
            if re.search(r"(?i)暂停|pause", stripped):
                return GmailBackupPlan(tool="gmail_backup", action="pause", account=account)
            if re.search(r"(?i)继续|恢复|resume|continue", stripped):
                return GmailBackupPlan(tool="gmail_backup", action="resume", account=account)
            if re.search(r"(?i)取消|停止|cancel|stop", stripped):
                return GmailBackupPlan(tool="gmail_backup", action="cancel", account=account)
            if re.search(r"(?i)断开|撤销授权|disconnect|revoke", stripped):
                return GmailBackupPlan(tool="gmail_backup", action="disconnect", account=account)
            if re.search(r"(?i)状态|情况|进度|status|progress", stripped):
                return GmailBackupPlan(tool="gmail_backup", action="status", account=account)
            if re.search(r"(?i)连接|授权|登录|connect|authorize|sign[ -]?in", stripped):
                return GmailBackupPlan(tool="gmail_backup", action="connect", account=account)
            if re.search(r"(?i)硬盘|磁盘|disk|drive", stripped):
                return GmailBackupPlan(tool="gmail_backup", action="export_disk", account=account or "*")
            if re.search(r"(?i)备份|同步|归档|backup|back up|archive|sync", stripped):
                if account is None and re.search(r"(?i)同步|sync", stripped):
                    account = "*"
                return GmailBackupPlan(tool="gmail_backup", action="backup", account=account)
        video_request = bool(
            re.search(r"(制作|生成|做|create|make|produce|generate)", stripped, re.I)
            and re.search(r"(视频|video)", stripped, re.I)
        )
        if video_request:
            return FinancialVideoPlan(
                tool="financial_video_create",
                request=stripped,
                provider="codex" if re.search(r"\bcodex\b", stripped, re.I) else "ollama",
            )
        if re.search(r"\bcodex\b", stripped, re.I) and not stripped.lower().startswith("/codex"):
            task = re.sub(r"(?i)(请|please)?\s*(使用|用|use)?\s*codex\s*", "", stripped, count=1).strip()
            return CodexRunPlan(tool="codex_run", task=task or stripped, workspace="isolated")
        if stripped == "/agents":
            return AgentListPlan(tool="agent_list")
        if re.search(r"(?i)(?:model\s*(?:&|and)?\s*capability\s*)?curator|模型与能力更新", stripped):
            if re.search(r"(?i)审计|检查更新|audit", stripped):
                return CuratorAuditPlan(tool="curator_audit")
            return CuratorStatusPlan(tool="curator_status")
        if re.fullmatch(r"/curator(?:\s+status)?", stripped, re.I):
            return CuratorStatusPlan(tool="curator_status")
        if re.fullmatch(r"/curator\s+audit", stripped, re.I):
            return CuratorAuditPlan(tool="curator_audit")
        match = re.fullmatch(r"/curator\s+approve\s+([\w-]+)", stripped, re.I)
        if match:
            return CuratorApprovePlan(tool="curator_approve", proposal_id=match.group(1))
        match = re.fullmatch(r"/camera\s+(\*)(?:\s+(.+))?", stripped, re.I | re.S)
        if match:
            return CameraViewPlan(tool="camera_view", provider=match.group(1).lower(), question=(match.group(2) or "Is anyone visible?").strip())
        match = re.fullmatch(r"/gmail\s+(connect|backup|export|status|disconnect|pause|resume|cancel)(?:\s+(\S+))?", stripped, re.I)
        if match:
            action = "export_disk" if match.group(1).lower() == "export" else match.group(1).lower()
            return GmailBackupPlan(tool="gmail_backup", action=action, account=match.group(2))
        match = re.fullmatch(r"/codex\s+(approve|reject)\s+([\w-]+)", stripped, re.I)
        if match:
            approved = match.group(1).lower() == "approve"
            plan_type = CodexApprovePlan if approved else CodexRejectPlan
            tool = "codex_approve" if approved else "codex_reject"
            return plan_type(tool=tool, request_id=match.group(2))
        match = re.fullmatch(r"/codex(?:\s+(project))?\s+(.+)", stripped, re.I | re.S)
        if match:
            return CodexRunPlan(
                tool="codex_run",
                task=match.group(2).strip(),
                workspace="family_project" if match.group(1) else "isolated",
            )
        match = re.fullmatch(r"/agent\s+(approve|reject)\s+([\w-]+)", stripped, re.I)
        if match:
            plan_type = AgentApprovePlan if match.group(1).lower() == "approve" else AgentRejectPlan
            tool = "agent_approve" if match.group(1).lower() == "approve" else "agent_reject"
            return plan_type(tool=tool, proposal_id=match.group(2))
        system = SystemMessage(content=PLANNER_PROMPT.format(today=date.today().isoformat()))
        human = HumanMessage(content=f"Recent conversation:\n{recent_context or '(none)'}\n\nCurrent request:\n{text}")
        decision = self.classifier.invoke([system, human])
        simple = {
            ToolName.chat: ChatPlan,
            ToolName.current_time: TimePlan,
            ToolName.memory_list: MemoryListPlan,
            ToolName.todo_list: TodoListPlan,
            ToolName.help: HelpPlan,
            ToolName.agent_list: AgentListPlan,
            ToolName.curator_status: CuratorStatusPlan,
            ToolName.curator_audit: CuratorAuditPlan,
        }
        if decision.tool in simple:
            return simple[decision.tool](tool=decision.tool.value)
        extractor = self.extractors.get(decision.tool)
        if not extractor:
            return ClarifyPlan(tool="clarify", clarification="请换一种方式说明你希望我做什么。")
        extraction_prompt = SystemMessage(content=f"""Extract and normalize arguments for the `{decision.tool.value}` tool.
Today is {date.today().isoformat()}. Preserve the user's meaning exactly.
For weather, convert Chinese number words and return the exact requested day count; 未来3天 means offset 0 and days 3, 从明天起3天 means offset 1 and days 3.
For memory/todo content, copy the fact or task itself without command words.
For web search, preserve place/category constraints and extract the exact requested result count; Chinese '3个' means result_count=3.
For agent proposals, choose a short name and minimum permissions. A proposal never activates itself.
For scope/owner, default to private/me unless the user explicitly says family/shared/everyone.
Use recent conversation to resolve follow-up references. Do not answer the request.""")
        return extractor.invoke([extraction_prompt, human])

    def search_strategy(self, original_request: str, base_query: str) -> list[str]:
        prompt = SystemMessage(content=f"""You are a web-search strategist.
Generate 2 or 3 meaningfully different search-engine queries for the request.
The searches must jointly improve freshness, authority, and local relevance.
For recommendations, include one authoritative/official or respected specialist query and one recent local-guide query.
Preserve every location, category, date, quantity, and user constraint.
Do not answer the question. Do not use private memory. Today is {date.today().isoformat()}.""")
        request = HumanMessage(
            content=f"Original request: {original_request}\nBase query: {base_query}"
        )
        strategy = self.search_strategist.invoke([prompt, request])
        unique = []
        for query in [base_query, *strategy.queries]:
            cleaned = query.strip()
            if cleaned and cleaned.casefold() not in {item.casefold() for item in unique}:
                unique.append(cleaned)
        return unique[:3]
