import os

class TranscribeTool:
    """
    LangChain-compatible tool that transcribes audio using Groq's
    Whisper large-v3 model — extremely fast inference.
    """

    name = "TranscribeTool"
    description = "Transcribes an audio file to text using Groq Whisper."

    def __init__(self, groq_client):
        self.groq_client = groq_client

    def run(self, audio_path: str) -> str:
        """
        Transcribe audio file at `audio_path`.
        Returns the full transcript as a string.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        with open(audio_path, "rb") as audio_file:
            transcription = self.groq_client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), audio_file),
                model="whisper-large-v3",
                response_format="verbose_json",  # includes timestamps
                language="en",
                temperature=0.0,
            )

        # verbose_json gives us segments with timestamps
        if hasattr(transcription, "segments") and transcription.segments:
            # Build timestamped transcript
            lines = []
            for seg in transcription.segments:
                start = self._fmt_time(seg.get("start", 0))
                text = seg.get("text", "").strip()
                lines.append(f"[{start}] {text}")
            return "\n".join(lines)

        return transcription.text

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
