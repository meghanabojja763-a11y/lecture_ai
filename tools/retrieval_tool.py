from typing import List
from rag.rag_pipeline import RAGPipeline


class RetrievalTool:
    """
    LangChain-compatible retrieval tool that queries the RAG pipeline
    and tracks source segments for citation display.
    """

    name = "RetrieveFromLecture"
    description = (
        "Retrieves relevant lecture segments to answer a student's question. "
        "Input should be a question or topic string."
    )

    def __init__(self, rag: RAGPipeline):
        self.rag = rag
        self.last_sources: List[str] = []

    def run(self, question: str) -> str:
        """
        Run semantic search + QA over lecture transcript.
        Stores source references for UI display.
        """
        result = self.rag.query(question)
        self.last_sources = result.get("sources", [])
        return result["answer"]

    def get_chunks(self, query: str, k: int = 3) -> List[str]:
        """Return raw similar chunks (used for context building)."""
        docs = self.rag.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
