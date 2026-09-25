from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "local-main"
    ollama_chat_context: int = 16384
    ollama_tool_context: int = 12288
    ollama_planner_context: int = 8192
    ollama_chat_max_tokens: int = 1024
    ollama_tool_max_tokens: int = 768
    ollama_planner_max_tokens: int = 384
    family_name: str = "Our Family"
    family_ai_data_dir: Path = Path(__file__).resolve().parents[2] / "data"
    family_ai_project_dir: Path = Path("*")
    financial_video_workspace: Path = Path("*")
    database_url: str | None = None
    rag_embedding_model: str = "*"
    codex_cli_path: Path = Path(
        "*"
    )
    whisper_model_path: Path = Path("*")
    tts_model_path: Path = Path("*")
    tts_chinese_voice: str = "*"
    tts_english_voice: str = "*"
    voice_wake_word: str = "*"
    camera_capture_helper: Path = Path(
        "/Applications/Family AI Camera Bridge.app/Contents/MacOS/camera-window-capture"
    )
    camera_monitor_enabled: bool = False
    camera_monitor_interval_seconds: float = 6.0
    camera_gate_interval_seconds: float = 2.0
    camera_alert_cooldown_seconds: int = 120
    google_oauth_client_id: str = ""
    google_oauth_client_secret: str = ""
    google_oauth_redirect_uri: str = "http://127.0.0.1:8000/api/gmail/oauth/callback"
    gmail_backup_root: Path = Path("*")
    gmail_daily_sync_hour: int = 3
    school_digest_enabled: bool = True
    school_digest_hour: int = 9
    school_digest_member_id: str = "*"
    school_digest_sender: str = "*"
    school_digest_recipient: str = "*"
    school_email_lookback_days: int = 30
    family_ai_lan_access_token: str = ""
    family_ai_mcp_host: str = "0.0.0.0"
    family_ai_mcp_port: int = 8001
    family_ai_mcp_devices_file: Path | None = None
    family_ai_mcp_allowed_hosts: str = (
        "localhost,127.0.0.1,*,*"
    )

    @property
    def mcp_devices_file(self) -> Path:
        return self.family_ai_mcp_devices_file or self.family_ai_data_dir / "mcp-devices.json"


settings = Settings()
settings.family_ai_data_dir.mkdir(parents=True, exist_ok=True)
