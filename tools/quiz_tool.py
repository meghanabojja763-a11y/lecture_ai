import json
import re
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


QUIZ_PROMPT = PromptTemplate(
    input_variables=["transcript", "num_questions"],
    template="""You are an expert educator. Create a multiple-choice quiz based on this lecture transcript.

Transcript:
{transcript}

Generate exactly {num_questions} multiple-choice questions.
Respond ONLY with a valid JSON array (no markdown) in this format:
[
  {{
    "question": "Question text here?",
    "options": ["A) option 1", "B) option 2", "C) option 3", "D) option 4"],
    "answer": "A) option 1",
    "explanation": "Brief explanation of why this is correct"
  }}
]

Make questions that test understanding, not just memorization."""
)


class QuizTool:
    """Generates MCQ quizzes from lecture transcripts."""

    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        self.chain = QUIZ_PROMPT | llm | StrOutputParser()

    def run(self, transcript: str, num_questions: int = 5) -> list:
        """Generate quiz questions. Returns list of question dicts."""
        max_chars = 10000
        if len(transcript) > max_chars:
            transcript = transcript[:max_chars]

        raw = self.chain.invoke({
            "transcript": transcript,
            "num_questions": str(num_questions),
        })

        return self._parse(raw)

    def _parse(self, raw: str) -> list:
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        try:
            data = json.loads(raw)
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return [{
                "question": "Could not parse quiz. Try again.",
                "options": [],
                "answer": "",
                "explanation": raw[:200],
            }]
