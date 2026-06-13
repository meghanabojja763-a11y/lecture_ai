import json
import os
from datetime import datetime
from pathlib import Path


STORAGE_DIR = Path("lecture_data")


class LectureStorage:
    """
    Simple JSON-based persistent storage for lectures.
    Stores transcript, summary, and metadata per lecture.
    """

    def __init__(self, base_dir: str = "lecture_data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.index_path = self.base_dir / "index.json"
        self._ensure_index()

    # ── Index management ──────────────────────────────────────────────────────

    def _ensure_index(self):
        if not self.index_path.exists():
            self._write_index([])

    def _read_index(self) -> list:
        with open(self.index_path) as f:
            return json.load(f)

    def _write_index(self, data: list):
        with open(self.index_path, "w") as f:
            json.dump(data, f, indent=2)

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def save_lecture(self, title: str, transcript: str, summary: dict) -> bool:
        """Save a processed lecture to disk."""
        safe_title = self._safe_filename(title)
        lecture_path = self.base_dir / f"{safe_title}.json"

        data = {
            "title": title,
            "transcript": transcript,
            "summary": summary,
            "created_at": datetime.now().isoformat(),
            "word_count": len(transcript.split()),
        }

        with open(lecture_path, "w") as f:
            json.dump(data, f, indent=2)

        # Update index
        index = self._read_index()
        # Remove old entry with same title if exists
        index = [l for l in index if l["title"] != title]
        index.append({
            "title": title,
            "file": f"{safe_title}.json",
            "created_at": data["created_at"],
            "word_count": data["word_count"],
        })
        self._write_index(index)
        return True

    def get_lecture(self, title: str) -> dict | None:
        """Load a lecture by title."""
        safe_title = self._safe_filename(title)
        lecture_path = self.base_dir / f"{safe_title}.json"

        if not lecture_path.exists():
            # Try searching index
            index = self._read_index()
            for entry in index:
                if entry["title"] == title:
                    lecture_path = self.base_dir / entry["file"]
                    break

        if lecture_path.exists():
            with open(lecture_path) as f:
                return json.load(f)
        return None

    def list_lectures(self) -> list:
        """Return list of all saved lecture metadata."""
        return self._read_index()

    def delete_lecture(self, title: str) -> bool:
        """Delete a lecture from storage."""
        index = self._read_index()
        entry = next((l for l in index if l["title"] == title), None)
        if not entry:
            return False

        lecture_path = self.base_dir / entry["file"]
        if lecture_path.exists():
            os.remove(lecture_path)

        index = [l for l in index if l["title"] != title]
        self._write_index(index)
        return True

    @staticmethod
    def _safe_filename(title: str) -> str:
        """Convert title to a safe filename."""
        safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
        return safe.strip().replace(" ", "_").lower()[:80]
