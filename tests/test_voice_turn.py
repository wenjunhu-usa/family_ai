from family_ai.voice_turn import VoiceTurnAgent


def test_wake_word_routes_request():
    decision = VoiceTurnAgent().decide("*，明天天气怎么样？", False)
    assert decision.should_respond
    assert decision.message == "明天天气怎么样？"
    assert decision.reason == "wake_word"


def test_follow_up_routes_without_wake_word():
    decision = VoiceTurnAgent().decide("那后天呢？", True)
    assert decision.should_respond
    assert decision.message == "那后天呢？"
    assert decision.reason == "conversation_follow_up"


def test_background_speech_is_ignored_outside_follow_up():
    decision = VoiceTurnAgent().decide("我们晚上去超市", False)
    assert not decision.should_respond
    assert decision.reason == "not_directed"


def test_whisper_noise_is_ignored_during_follow_up():
    decision = VoiceTurnAgent().decide("谢谢观看", True)
    assert not decision.should_respond
    assert decision.reason == "noise"
