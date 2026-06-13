import json
import re
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


SUMMARY_PROMPT = PromptTemplate(
    input_variables=["transcript"],
    template="""You are an expert academic assistant. Analyze the following lecture transcript 
and produce a comprehensive structured summary.

Transcript:
{transcript}

Respond with ONLY a valid JSON object (no markdown, no code blocks) in this exact format:
{{
  "overview": "2-3 sentence high-level overview of the entire lecture",
  "key_topics": ["topic 1", "topic 2", "topic 3"],
  "main_points": ["point 1", "point 2", "point 3", "point 4", "point 5"],
  "key_concepts": ["concept: brief definition", "concept: brief definition"],
  "action_items": ["thing students should review or practice"],
  "difficulty_level": "beginner/intermediate/advanced"
}}"""
)

TOPIC_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["transcript", "topic"],
    template="""From the following lecture transcript, extract and summarize everything 
related to the topic: "{topic}".

Transcript:
{transcript}

Provide a clear, structured explanation of this topic as covered in the lecture."""
)


class SummarizeTool:
    """
    LangChain tool that generates structured lecture summaries using Groq LLM.
    """

    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        self.chain = SUMMARY_PROMPT | llm | StrOutputParser()
        self.topic_chain = TOPIC_SUMMARY_PROMPT | llm | StrOutputParser()

    def run(self, transcript: str) -> dict:
        """
        Generate a structured JSON summary of the lecture.
        Falls back to a default structure on parse failure.
        """
        # Truncate very long transcripts to fit context window
        max_chars = 12000
        if len(transcript) > max_chars:
            transcript = transcript[:max_chars] + "\n...[transcript truncated]"

        raw = self.chain.invoke({"transcript": transcript})
        return self._parse_json(raw)

    def run_topic(self, transcript: str, topic: str) -> str:
        """Summarize a specific topic from the lecture."""
        if not transcript:
            return "No transcript available."
        max_chars = 12000
        if len(transcript) > max_chars:
            transcript = transcript[:max_chars]
        return self.topic_chain.invoke({"transcript": transcript, "topic": topic})

    def _parse_json(self, raw: str) -> dict:
        """Safely parse LLM JSON output."""
        # Strip markdown code fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: extract whatever we can
            return {
                "overview": raw[:500] if raw else "Summary generation failed.",
                "key_topics": [],
                "main_points": [],
                "key_concepts": [],
                "action_items": [],
                "difficulty_level": "unknown",
            }
