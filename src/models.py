from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


@dataclass
class AudioFile:
    """音频文件数据类型"""
    file_path: Path
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    format: Optional[str] = None
    size_bytes: Optional[int] = None
    source_url: Optional[str] = None


@dataclass
class TranscriptSegment:
    """转录片段数据类型"""
    text: str
    start_time: float
    end_time: float
    confidence: Optional[float] = None


@dataclass
class Transcript:
    """转录结果数据类型"""
    segments: List[TranscriptSegment]
    language: str
    source_audio: AudioFile
    full_text: str
    processing_time: Optional[float] = None
    
    def __post_init__(self):
        """自动生成完整文本"""
        if not self.full_text:
            self.full_text = " ".join(segment.text for segment in self.segments)