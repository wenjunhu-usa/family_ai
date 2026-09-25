from family_ai.planner import ModelPlanner


def planner_without_init():
    return object.__new__(ModelPlanner)


def test_explicit_codex_commands_do_not_need_model():
    planner = planner_without_init()
    direct = planner.plan("/codex explain this theorem")
    assert direct.tool == "codex_run"
    assert direct.workspace == "isolated"
    project = planner.plan("/codex project review the architecture")
    assert project.workspace == "family_project"
    approval = planner.plan("/codex approve abc123")
    assert approval.tool == "codex_approve"


def test_common_explicit_commands_do_not_load_model():
    planner = planner_without_init()
    assert planner.plan("/help").tool == "help"
    assert planner.plan("/memory").tool == "memory_list"
    assert planner.plan("/todo list").tool == "todo_list"
    assert planner.plan("/todo add family Buy milk").owner == "family"
    assert planner.plan("/todo done abc123").todo_id == "abc123"
    assert planner.plan("/remember private passport expires in May").scope == "private"


def test_explicit_camera_command_does_not_need_model():
    planner = planner_without_init()
    plan = planner.plan("/camera * Is anyone visible?")
    assert plan.tool == "camera_view"
    assert plan.provider == "*"
    assert plan.question == "Is anyone visible?"


def test_explicit_gmail_backup_commands_do_not_need_model():
    planner = planner_without_init()
    assert planner.plan("/gmail connect").action == "connect"
    assert planner.plan("/gmail backup").action == "backup"
    assert planner.plan("/gmail status").action == "status"


def test_natural_gmail_backup_requests_do_not_require_commands():
    planner = planner_without_init()
    assert planner.plan("连接我的 Gmail，然后准备备份").action == "connect"
    assert planner.plan("现在开始备份我的 Gmail").action == "backup"
    assert planner.plan("Gmail 备份进度怎么样？").action == "status"
    assert planner.plan("Please disconnect Gmail backup").action == "disconnect"
    assert planner.plan("备份所有 Gmail").account == "*"
    assert planner.plan("同步 Gmail").account == "*"
    assert planner.plan("备份 first@gmail.example").account == "first@gmail.example"
    assert planner.plan("把 Gmail 备份到 disk").action == "export_disk"
    assert planner.plan("把所有 Gmail 邮件备份到移动硬盘").account == "*"
    assert planner.plan("把email从postgresql备份进入disk").action == "export_disk"


def test_bare_progress_question_inspects_all_gmail_jobs():
    planner = planner_without_init()
    plan = planner.plan("查看进度")
    assert plan.tool == "gmail_backup"
    assert plan.action == "status"
    assert plan.account == "*"
    assert planner.plan("暂停 Gmail 备份").action == "pause"
    assert planner.plan("继续 Gmail 备份").action == "resume"
    assert planner.plan("取消 Gmail 备份").action == "cancel"


def test_system_diagnostics_routes_without_model():
    planner = planner_without_init()
    plan = planner.plan("检查 PostgreSQL 数据库")
    assert plan.tool == "system_diagnostics"
    assert plan.target == "database"
    assert planner.plan("诊断 Gmail worker").target == "gmail"


def test_workflow_and_codex_status_are_independent_observations():
    planner = planner_without_init()
    gmail = planner.plan("检查备份gmailstatus")
    assert (gmail.tool, gmail.action, gmail.account) == ("gmail_backup", "status", "*")
    codex = planner.plan("我approve了codex，是否在运行？")
    assert (codex.tool, codex.target) == ("system_diagnostics", "codex")
def test_curator_natural_language_status_routes_without_model():
    planner = ModelPlanner.__new__(ModelPlanner)
    assert planner.plan("查看 model curator 状态").tool == "curator_status"
