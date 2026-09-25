from ddgs import DDGS


def search_web(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search public webpages. Never pass family memories into this function."""
    errors = []
    for backend in ("brave", "bing", "auto"):
        try:
            results = DDGS(timeout=15).text(
                query,
                region="us-en",
                safesearch="on",
                max_results=max_results,
                backend=backend,
            )
            normalized = [
                {
                    "title": str(item.get("title", "")),
                    "url": str(item.get("href", "")),
                    "snippet": str(item.get("body", "")),
                }
                for item in results
                if item.get("href") and item.get("title")
            ]
            if normalized:
                return normalized
            errors.append(f"{backend}: empty")
        except Exception as exc:
            errors.append(f"{backend}: {type(exc).__name__}")
    raise RuntimeError("all search backends failed (" + ", ".join(errors) + ")")


def format_results(results: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"[{index}] {item['title']}\nURL: {item['url']}\n摘要: {item['snippet']}"
        for index, item in enumerate(results, 1)
    )
