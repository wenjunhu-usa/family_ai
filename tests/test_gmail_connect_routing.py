from family_ai.graph import should_start_gmail_oauth


def test_existing_account_does_not_repeat_oauth():
    existing = ["first@example.com"]
    assert not should_start_gmail_oauth(existing, None, "连接我的 Gmail")
    assert not should_start_gmail_oauth(existing, "first@example.com", "连接 Gmail")


def test_explicit_second_account_starts_oauth():
    existing = ["first@example.com"]
    assert should_start_gmail_oauth(existing, None, "连接我的第二个 Gmail")
    assert should_start_gmail_oauth(existing, "second@example.com", "连接 second@example.com")
