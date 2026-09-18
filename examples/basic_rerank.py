"""Rerank a handful of in-memory nodes with Jev.

Requires a live OPENROUTER_API_KEY. The embedding model is a local mock so
this file does not also need an OpenAI key; retrieval quality is not the
point — the before/after ordering is.
"""

from __future__ import annotations

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.postprocessor.jev import JevRerank

Settings.embed_model = MockEmbedding(embed_dim=8)

DOCUMENTS = [
    Document(text="The Eiffel Tower is a wrought-iron tower in Paris, France."),
    Document(text="Python is a high-level programming language."),
    Document(text="The Louvre is the world's largest art museum, in Paris."),
    Document(text="Photosynthesis converts light energy into chemical energy."),
    Document(text="Paris is the capital of France."),
]

QUERY = "Where is the Eiffel Tower?"


def main() -> None:
    index = VectorStoreIndex.from_documents(DOCUMENTS)
    retriever = index.as_retriever(similarity_top_k=5)
    retrieved = retriever.retrieve(QUERY)

    print("Before JevRerank:")
    for node in retrieved:
        print(f"  {node.score:.3f}  {node.node.get_content()[:80]}")

    reranked = JevRerank(
        provider="openrouter", top_n=3, mode="score", timeout_s=30
    ).postprocess_nodes(retrieved, query_str=QUERY)

    print("\nAfter JevRerank:")
    for node in reranked:
        print(f"  {node.score:.3f}  {node.node.get_content()[:80]}")


if __name__ == "__main__":
    main()
