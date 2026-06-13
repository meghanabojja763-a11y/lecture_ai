from typing import Optional
from groq import Groq
from langchain_groq import ChatGroq

from tools.transcribe_tool import TranscribeTool
from tools.summarize_tool import SummarizeTool
from tools.retrieval_tool import RetrievalTool
from tools.quiz_tool import QuizTool
from rag.rag_pipeline import RAGPipeline


class LectureAgent:
    """
    Lecture assistant orchestrator.
    Uses LangChain RAG + Groq LLM directly (no AgentExecutor needed).
    """

    def __init__(
        self,
        groq_api_key: str,
        model_name: str = "llama-3.3-70b-versatile",
        transcript: Optional[str] = None,
    ):
        self.groq_api_key = groq_api_key
        self.model_name = model_name
        self.transcript = transcript

        # Groq raw client for Whisper
        self.groq_client = Groq(api_key=groq_api_key)

        # LangChain LLM via Groq
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model_name=model_name,
            temperature=0.3,
            max_tokens=2048,
        )

        # RAG pipeline
        self.rag = RAGPipeline(llm=self.llm)

        # Simple conversation memory as plain list — no langchain.memory needed
        self.chat_history = []  # [{"role": "user"/"assistant", "content": "..."}]

        # Tools
        self._summarize_tool = SummarizeTool(llm=self.llm)
        self._quiz_tool = QuizTool(llm=self.llm)
        self._retrieval_tool = RetrievalTool(rag=self.rag)

        # If transcript provided (loading saved lecture), build RAG index immediately
        if transcript:
            self.build_rag_index(transcript)

    # ── Core processing ──────────────────────────────────────────────────────

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio using Groq Whisper."""
        tool = TranscribeTool(groq_client=self.groq_client)
        transcript = tool.run(audio_path)
        self.transcript = transcript
        return transcript

    def summarize(self, transcript: str) -> dict:
        """Generate structured summary using LLM."""
        return self._summarize_tool.run(transcript)

    def build_rag_index(self, transcript: str):
        """Chunk transcript and build vector store."""
        self.rag.build_index(transcript)
        self._retrieval_tool = RetrievalTool(rag=self.rag)

    def generate_quiz(self, transcript: str, num_questions: int = 5) -> list:
        """Generate MCQ quiz from transcript."""
        return self._quiz_tool.run(transcript, num_questions=num_questions)

    # ── Q&A ──────────────────────────────────────────────────────────────────

    def ask(self, question: str) -> dict:
        """
        Answer a student question using RAG over the lecture transcript.
        Maintains conversation memory for follow-up questions.
        """
        if not self.transcript:
            return {
                "answer": "No lecture loaded. Please upload and process a lecture first.",
                "sources": [],
            }

        try:
            # Get answer from RAG pipeline
            result = self.rag.query(question)
            answer = result["answer"]
            sources = result.get("sources", [])

            # Save to memory for context
            self.chat_history.append({"role": "user", "content": question})
            self.chat_history.append({"role": "assistant", "content": answer})

            return {"answer": answer, "sources": sources}

        except Exception as e:
            return {"answer": f"Sorry, I encountered an error: {str(e)}", "sources": []}