"""Wire two toy tools through RouterQueryEngine + JevSingleSelector.

Requires a live OPENROUTER_API_KEY. LlamaIndex still constructs an LLM for
the unused multi-route summarizer; MockLLM keeps that from needing OpenAI.
"""

from __future__ import annotations

from llama_index.core.llms import MockLLM
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
        selector=JevSingleSelector(provider="openrouter", timeout_s=30),
        query_engine_tools=[weather_tool, docs_tool],
        llm=MockLLM(),
        verbose=True,
    )
    response = engine.query("Will it rain in Paris tomorrow?")
    print(response)


if __name__ == "__main__":
    main()
