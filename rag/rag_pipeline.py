from typing import Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert lecture assistant. Use ONLY the following lecture transcript 
excerpts to answer the student's question. If the answer is not in the context, say so honestly.

Lecture Context:
{context}

Student Question: {question}

Answer clearly and concisely, referencing specific parts of the lecture when relevant:""",
)


def _format_docs(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


class RAGPipeline:
    """
    RAG pipeline using modern LangChain LCEL (no deprecated RetrievalQA):
      1. Split transcript into chunks
      2. Embed with HuggingFace sentence-transformers
      3. Store in FAISS vector store
      4. Retrieve + answer with LCEL chain
    """

    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        self.vectorstore: Optional[FAISS] = None
        self.retriever = None
        self.chain = None
        self._last_docs: list = []

        # Embedding model (free, runs locally)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def build_index(self, transcript: str) -> None:
        """Chunk transcript and build FAISS vector store."""
        chunks = self.splitter.split_text(transcript)

        docs = [
            Document(
                page_content=chunk,
                metadata={"chunk_id": i, "source": f"Segment {i+1}"},
            )
            for i, chunk in enumerate(chunks)
        ]

        self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )

        # Modern LCEL chain — replaces RetrievalQA entirely
        self.chain = (
            {
                "context": self.retriever | _format_docs,
                "question": RunnablePassthrough(),
            }
            | RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )

    def query(self, question: str) -> dict:
        """Query the RAG chain and return answer + source segments."""
        if not self.chain:
            return {
                "answer": "No lecture loaded. Please upload and process a lecture first.",
                "sources": [],
            }

        # Get answer
        answer = self.chain.invoke(question)

        # Get source docs separately for citations
        sources = []
        if self.retriever:
            try:
                source_docs = self.retriever.invoke(question)
                sources = list({
                    doc.metadata.get("source", "Unknown")
                    for doc in source_docs
                })
            except Exception:
                pass

        return {"answer": answer, "sources": sources}

    def similarity_search(self, query: str, k: int = 4) -> list:
        """Return top-k similar chunks for a query."""
        if not self.vectorstore:
            return []
        return self.vectorstore.similarity_search(query, k=k)