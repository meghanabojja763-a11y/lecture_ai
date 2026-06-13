import os
import re
from pathlib import Path


AUDIO_DIR = Path("audio_uploads")


def save_uploaded_audio(uploaded_file, title: str) -> str:
    """
    Save a Streamlit UploadedFile to disk.
    Returns the saved file path.
    """
    AUDIO_DIR.mkdir(exist_ok=True)
    safe_title = re.sub(r"[^\w\s-]", "_", title).strip().replace(" ", "_")[:60]
    ext = Path(uploaded_file.name).suffix or ".mp3"
    filename = f"{safe_title}{ext}"
    filepath = AUDIO_DIR / filename

    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return str(filepath)


def get_audio_duration_estimate(file_size_bytes: int) -> str:
    """Rough estimate of audio duration from file size."""
    # ~1 MB per minute for typical MP3 at 128kbps
    minutes = file_size_bytes / (1024 * 1024)
    if minutes < 1:
        return "< 1 minute"
    elif minutes < 60:
        return f"~{int(minutes)} minutes"
    else:
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        return f"~{hours}h {mins}m"
