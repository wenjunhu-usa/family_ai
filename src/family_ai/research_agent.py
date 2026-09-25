from __future__ import annotations

from .agent_runtime import AgentOutcome, AgentRuntime


class ResearchAgent:
    permissions = frozenset({"network_public"})

    def __init__(self, runtime: AgentRuntime, strategy, search, formatter,
                 synthesizer, diagnose=None):
        self.runtime, self.strategy, self.search = runtime, strategy, search
        self.formatter, self.synthesizer = formatter, synthesizer
        self.diagnose = diagnose

    def collect(self, original_request: str, base_query: str,
                requested: int = 5) -> AgentOutcome:
        def action():
            try:
                queries = self.strategy(original_request, base_query)
            except Exception:
                queries = [base_query]
            unique, attempts = {}, []
            for query in queries:
                try:
                    batch = self.search(query, max_results=max(5, requested))
                    attempts.append({"query": query, "status": "success", "count": len(batch)})
                    for item in batch:
                        unique.setdefault(item["url"], item)
                except Exception as exc:
                    attempts.append({"query": query, "status": "failed",
                                     "error_type": type(exc).__name__})
            results = list(unique.values())
            if not results:
                raise RuntimeError("all public search strategies returned no results")
            return {"attempts": attempts, "results": results,
                    "formatted": self.formatter(results)}

        return self.runtime.execute(
            "Research Agent", "collect and deduplicate public sources", action,
            diagnose=self.diagnose,
            verify=lambda value: bool(value["results"]) and all(
                item.get("url") for item in value["results"]),
        )

    def synthesize(self, prompt) -> AgentOutcome:
        return self.runtime.execute(
            "Research Agent", "synthesize source-grounded answer",
            lambda: self.synthesizer(prompt), diagnose=self.diagnose,
            verify=lambda value: bool(str(value.content).strip()),
        )
