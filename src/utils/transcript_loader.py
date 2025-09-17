import re
from pathlib import Path
from typing import List, Optional

from src.models import Transcript, TranscriptSegment, AudioFile


def load_transcript_md(file_path: Path) -> Optional[Transcript]:
    """Parse a saved transcript .md file and reconstruct a Transcript object.

    Expected format written by WhisperTranscriber.save_transcript:
    - Header lines with language and optional source URL
    - Section '## 完整文本' followed by full text
    - Section '## 时间戳段落' with lines:
        '  i. [<start>s - <end>s] <text>'
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception:
        return None

    language = ""
    source_url: Optional[str] = None
    full_text = ""
    segments: List[TranscriptSegment] = []

    # Extract language and source
    for line in content.splitlines():
        if line.startswith("语言:"):
            language = line.split(":", 1)[1].strip()
        elif line.startswith("来源:"):
            source_url = line.split(":", 1)[1].strip()

    # Extract full text
    full_match = re.search(r"## 完整文本\n(.+?)\n\n## 时间戳段落", content, flags=re.DOTALL)
    if full_match:
        full_text = full_match.group(1).strip()

    # Extract segments
    ts_section = re.search(r"## 时间戳段落\n([\s\S]+)$", content)
    if ts_section:
        for line in ts_section.group(1).splitlines():
            m = re.match(r"\s*\d+\.\s*\[(\d+(?:\.\d+)?)s\s*-\s*(\d+(?:\.\d+)?)s\]\s*(.*)$", line)
            if m:
                start = float(m.group(1))
                end = float(m.group(2))
                text = m.group(3).strip()
                segments.append(TranscriptSegment(text=text, start_time=start, end_time=end))

    if not segments and not full_text:
        return None

    # Construct a minimal AudioFile using the md path to keep provenance
    duration = segments[-1].end_time if segments else None
    audio = AudioFile(file_path=file_path, duration=duration, format="md", source_url=source_url)

    return Transcript(
        segments=segments,
        language=language or "",
        source_audio=audio,
        full_text=full_text,
        processing_time=None,
    )


def load_transcripts_from_dir(directory: Path) -> List[Transcript]:
    """Load all .md transcripts from a directory."""
    transcripts: List[Transcript] = []
    for path in sorted(directory.glob("*.md")):
        t = load_transcript_md(path)
        if t is not None:
            transcripts.append(t)
    return transcripts

