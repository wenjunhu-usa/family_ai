from family_ai.rag import chunk_text, extract_text


def test_chunk_text_preserves_content_with_overlap():
    text = "第一段是家庭保险资料。" * 80
    chunks = chunk_text(text, max_chars=120, overlap=20)
    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 121 for chunk in chunks)
    assert "家庭保险资料" in chunks[0]


def test_extract_markdown_text():
    assert extract_text("notes.md", "# *\n家庭地点".encode()) == "# *\n家庭地点"


def test_rejects_unsupported_binary_files():
    try:
        extract_text("photo.jpg", b"image")
    except ValueError as exc:
        assert "Supported files" in str(exc)
    else:
        raise AssertionError("unsupported files must be rejected")
