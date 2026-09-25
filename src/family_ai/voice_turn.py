import re
from dataclasses import dataclass


WAKE_PATTERN = re.compile(
    r"^\s*(?:hey[,，\s]*)?\*[,，。:：!！?？\s]*",
    re.I,
)

NOISE_TRANSCRIPTS = {
    "嗯", "啊", "哦", "呃", "唉", "喂", "谢谢观看", "感谢观看",
    "字幕", "you", "um", "uh", "hmm", "ah", "oh",
}


@dataclass(frozen=True)
class VoiceTurnDecision:
    should_respond: bool
    message: str = ""
    reason: str = ""


class VoiceTurnAgent:
    """Permission-free gate. The Main Agent remains the semantic/tool authority."""

    name = "Voice Turn Agent"
    permissions: tuple[str, ...] = ()

    def decide(self, transcript: str, follow_up_active: bool) -> VoiceTurnDecision:
        text = re.sub(r"\s+", " ", transcript).strip()
        if not text:
            return VoiceTurnDecision(False, reason="empty")

        wake = WAKE_PATTERN.match(text)
        if wake:
            message = text[wake.end():].strip()
            if not message:
                return VoiceTurnDecision(False, reason="wake_only")
            return VoiceTurnDecision(True, message, "wake_word")

        normalized = re.sub(r"[\s,.，。!?！？:：'\"]", "", text).lower()
        if normalized in NOISE_TRANSCRIPTS or len(normalized) < 2:
            return VoiceTurnDecision(False, reason="noise")

        if follow_up_active:
            return VoiceTurnDecision(True, text, "conversation_follow_up")
        return VoiceTurnDecision(False, reason="not_directed")
