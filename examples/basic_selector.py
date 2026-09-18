"""Route a query with JevSingleSelector directly (no RouterQueryEngine).

Requires a live OPENROUTER_API_KEY.
"""

from __future__ import annotations

from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.selectors.jev import JevSingleSelector

QUERY = "Will it rain in Paris tomorrow?"


class StaticQueryEngine(CustomQueryEngine):
    """Documentation-only engine that returns a fixed string."""

    answer: str

    def custom_query(self, query_str: str) -> str:
        return self.answer


def main() -> None:
    weather_tool = QueryEngineTool(
        query_engine=StaticQueryEngine(answer="Weather engine"),
        metadata=ToolMetadata(
            name="weather",
            description="Answers questions about weather and forecasts",
        ),
    )
    docs_tool = QueryEngineTool(
        query_engine=StaticQueryEngine(answer="Docs engine"),
        metadata=ToolMetadata(
            name="docs",
            description="Answers questions about product documentation",
        ),
    )

    selector = JevSingleSelector(provider="openrouter", timeout_s=30)
    result = selector.select(
        [weather_tool.metadata, docs_tool.metadata],
        QUERY,
    )
    print(f"query:      {QUERY}")
    print(f"index:      {result.ind}")
    print(f"reason:     {result.reason}")
    print(f"chose:      {[weather_tool, docs_tool][result.ind].metadata.name}")


if __name__ == "__main__":
    main()
