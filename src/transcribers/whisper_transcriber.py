import logging
import time
import re
from typing import List, Optional, Union
from pathlib import Path

from faster_whisper import WhisperModel
import torch

from src.models import AudioFile, Transcript, TranscriptSegment


class WhisperTranscriber:
    """基于 faster-whisper 的语音转录器"""
    
    def __init__(
        self,
        model_name: str = "large-v3",
        language: str = "zh",
        device: str = "auto",
        batch_size: int = 8,
        compute_type: str = "float16"
    ):
        """
        初始化转录器
        
        Args:
            model_name: Whisper 模型名称 (tiny, base, small, medium, large, large-v2, large-v3)
            language: 音频语言代码 (zh, en, etc.)
            device: 计算设备 (auto, cpu, cuda)
            batch_size: 批处理大小
            compute_type: 计算精度 (float16, float32, int8)
        """
        self.model_name = model_name
        self.language = language
        self.batch_size = batch_size
        self.compute_type = compute_type
        self.logger = logging.getLogger(__name__)
        
        # 设备检测和配置
        self.device = self._setup_device(device)
        
        # 初始化模型
        self.model = self._initialize_model()
        
        # 文本清洗正则表达式
        self.text_patterns = [
            (r'\s+', ' '),  # 多个空白字符合并为一个空格
            (r'^[\s\n\r]*', ''),  # 去除开头空白
            (r'[\s\n\r]*$', ''),  # 去除结尾空白
            (r'[。]{2,}', '。'),  # 多个句号合并
            (r'[，]{2,}', '，'),  # 多个逗号合并
        ]
    
    def _setup_device(self, device: str) -> str:
        """设置和检测计算设备"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                self.logger.info("CUDA available, using GPU acceleration")
            else:
                device = "cpu"
                self.logger.info("CUDA not available, using CPU")
        else:
            self.logger.info(f"Using specified device: {device}")
        
        return device
    
    def _initialize_model(self) -> WhisperModel:
        """初始化 Whisper 模型"""
        try:
            self.logger.info(f"Loading Whisper model: {self.model_name} on {self.device}")
            
            model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type
            )
            
            self.logger.info("Whisper model loaded successfully")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {e}")
            # 降级到更小的模型
            if self.model_name == "large-v3":
                self.logger.info("Attempting to fallback to medium model")
                self.model_name = "medium"
                return self._initialize_model()
            raise
    
    def _clean_text(self, text: str) -> str:
        """清洗和标准化文本"""
        if not text:
            return ""
        
        cleaned = text
        for pattern, replacement in self.text_patterns:
            cleaned = re.sub(pattern, replacement, cleaned)
        
        return cleaned.strip()
    
    def _validate_audio_file(self, audio_file: AudioFile) -> bool:
        """验证音频文件"""
        if not audio_file.file_path.exists():
            self.logger.error(f"Audio file not found: {audio_file.file_path}")
            return False
        
        if audio_file.size_bytes is not None and audio_file.size_bytes == 0:
            self.logger.error(f"Audio file is empty: {audio_file.file_path}")
            return False
        
        return True
    
    def transcribe(self, audio_file: AudioFile) -> Transcript:
        """转录单个音频文件"""
        if not self._validate_audio_file(audio_file):
            raise ValueError(f"Invalid audio file: {audio_file.file_path}")
        
        self.logger.info(f"Transcribing: {audio_file.file_path}")
        start_time = time.time()
        
        try:
            # 执行转录
            segments, info = self.model.transcribe(
                str(audio_file.file_path),
                language=self.language,
                beam_size=5,
                word_timestamps=True,
                vad_filter=True,  # 语音活动检测
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=400
                )
            )
            
            # 转换结果
            transcript_segments = []
            for segment in segments:
                cleaned_text = self._clean_text(segment.text)
                if cleaned_text:  # 只保留非空文本
                    transcript_segments.append(TranscriptSegment(
                        text=cleaned_text,
                        start_time=segment.start,
                        end_time=segment.end,
                        confidence=getattr(segment, 'avg_logprob', None)
                    ))
            
            processing_time = time.time() - start_time
            
            # 生成完整文本
            full_text = " ".join(segment.text for segment in transcript_segments)
            
            transcript = Transcript(
                segments=transcript_segments,
                language=info.language,
                source_audio=audio_file,
                full_text=full_text,
                processing_time=processing_time
            )
            
            self.logger.info(
                f"Transcription completed in {processing_time:.2f}s, "
                f"generated {len(transcript_segments)} segments"
            )
            
            return transcript
            
        except Exception as e:
            self.logger.error(f"Transcription failed for {audio_file.file_path}: {e}")
            raise
    
    def batch_transcribe(self, audio_files: List[AudioFile]) -> List[Transcript]:
        """批量转录音频文件"""
        if not audio_files:
            return []
        
        self.logger.info(f"Starting batch transcription of {len(audio_files)} files")
        
        # 验证文件
        valid_files = [f for f in audio_files if self._validate_audio_file(f)]
        invalid_count = len(audio_files) - len(valid_files)
        
        if invalid_count > 0:
            self.logger.warning(f"Skipping {invalid_count} invalid audio files")
        
        if not valid_files:
            return []
        
        transcripts = []
        failed_count = 0
        
        # 批处理转录
        for i in range(0, len(valid_files), self.batch_size):
            batch = valid_files[i:i + self.batch_size]
            self.logger.info(f"Processing batch {i // self.batch_size + 1}: files {i+1}-{min(i + len(batch), len(valid_files))}")
            
            for audio_file in batch:
                try:
                    transcript = self.transcribe(audio_file)
                    transcripts.append(transcript)
                except Exception as e:
                    self.logger.error(f"Failed to transcribe {audio_file.file_path}: {e}")
                    failed_count += 1
        
        self.logger.info(
            f"Batch transcription completed. "
            f"Success: {len(transcripts)}, Failed: {failed_count}"
        )
        
        return transcripts
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "language": self.language,
            "device": self.device,
            "compute_type": self.compute_type,
            "batch_size": self.batch_size
        }
    
    def estimate_processing_time(self, audio_files: List[AudioFile]) -> float:
        """估算处理时间（基于音频时长）"""
        total_duration = 0.0
        
        for audio_file in audio_files:
            if audio_file.duration:
                total_duration += audio_file.duration
        
        # 估算：通常转录时间约为音频时长的 0.1-0.3 倍（取决于硬件）
        multiplier = 0.1 if self.device == "cuda" else 0.3
        estimated_time = total_duration * multiplier
        
        return estimated_time
    
    def save_transcript(self, transcript: Transcript, output_path: Path) -> None:
        """保存转录结果到文件"""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                # 写入元数据
                f.write(f"# 转录结果\n")
                f.write(f"语言: {transcript.language}\n")
                f.write(f"处理时间: {transcript.processing_time:.2f}秒\n")
                f.write(f"来源: {transcript.source_audio.source_url}\n\n")
                
                # 写入完整文本
                f.write("## 完整文本\n")
                f.write(f"{transcript.full_text}\n\n")
                
                # 写入时间戳段落
                f.write("## 时间戳段落\n")
                for i, segment in enumerate(transcript.segments, 1):
                    f.write(
                        f"{i:3d}. [{segment.start_time:6.1f}s - {segment.end_time:6.1f}s] "
                        f"{segment.text}\n"
                    )
            
            self.logger.info(f"Transcript saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save transcript to {output_path}: {e}")
            raise