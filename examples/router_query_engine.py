"""Wire two toy tools through RouterQueryEngine + JevSingleSelector.

Requires a live TYPESAFE_API_KEY.
"""

from __future__ import annotations

from llama_index.core.query_engine import CustomQueryEngine, RouterQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.selectors.jev import JevSingleSelector


class StaticQueryEngine(CustomQueryEngine):
    """Documentation-only engine that returns a fixed string."""

    answer: str

    def custom_query(self, query_str: str) -> str:
        return self.answer


def main() -> None:
    weather_tool = QueryEngineTool(
        query_engine=StaticQueryEngine(
            answer="Forecast: light rain in Paris tomorrow."
        ),
        metadata=ToolMetadata(
            name="weather",
            description="Answers questions about weather and forecasts",
        ),
    )
    docs_tool = QueryEngineTool(
        query_engine=StaticQueryEngine(
            answer="See the product docs for authentication."
        ),
        metadata=ToolMetadata(
            name="docs",
            description="Answers questions about product documentation",
        ),
    )

    engine = RouterQueryEngine(
        selector=JevSingleSelector(),
        query_engine_tools=[weather_tool, docs_tool],
    )
    response = engine.query("Will it rain in Paris tomorrow?")
    print(response)


if __name__ == "__main__":
    main()
