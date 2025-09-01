import re
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ..models import TextChunk, TranscriptSegment, Transcript


class TextProcessor:
    """文本处理器，负责清洗和分块处理转录文本"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, min_chunk_size: int = 100):
        """
        初始化文本处理器
        
        Args:
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
            min_chunk_size: 最小文本块大小
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # 配置 LangChain 文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        
        # 填充词和停用词列表
        self.filler_words = {
            '呃', '嗯', '啊', '哎', '额', '那个', '这个', '然后', '就是',
            '比如说', '怎么说呢', '对不对', '你知道吗', '是吧', '对吧'
        }
    
    def clean_text(self, text: str) -> str:
        """
        清洗文本内容
        
        Args:
            text: 待清洗的文本
            
        Returns:
            清洗后的文本
        """
        if not text:
            return ""
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 移除填充词
        words = text.split()
        cleaned_words = [word for word in words if word not in self.filler_words]
        text = ' '.join(cleaned_words)
        
        # 标准化标点符号
        text = self._normalize_punctuation(text)
        
        # 移除重复的标点
        text = re.sub(r'([。！？，；：])\1+', r'\1', text)
        
        return text.strip()
    
    def chunk_text(self, transcript: Transcript) -> List[TextChunk]:
        """
        将转录文本切分为语义块
        
        Args:
            transcript: 转录结果对象
            
        Returns:
            文本块列表
        """
        if not transcript or not transcript.segments:
            return []
        
        # 清洗完整文本
        cleaned_text = self.clean_text(transcript.full_text)
        
        if len(cleaned_text) < self.min_chunk_size:
            # 如果文本太短，返回单个块
            return [self._create_text_chunk(
                text=cleaned_text,
                transcript=transcript,
                chunk_index=0,
                start_time=transcript.segments[0].start_time if transcript.segments else 0.0,
                end_time=transcript.segments[-1].end_time if transcript.segments else 0.0
            )]
        
        # 使用 LangChain 进行文本分割
        text_chunks = self.text_splitter.split_text(cleaned_text)
        
        # 创建 TextChunk 对象并提取时间戳信息
        result_chunks = []
        for i, chunk_text in enumerate(text_chunks):
            if len(chunk_text.strip()) >= self.min_chunk_size:
                # 查找对应的时间戳信息
                start_time, end_time = self._extract_timestamps(chunk_text, transcript)
                
                text_chunk = self._create_text_chunk(
                    text=chunk_text,
                    transcript=transcript,
                    chunk_index=i,
                    start_time=start_time,
                    end_time=end_time
                )
                result_chunks.append(text_chunk)
        
        return result_chunks
    
    def _normalize_punctuation(self, text: str) -> str:
        """标准化标点符号"""
        # 英文标点转中文标点
        punctuation_map = {
            ', ': '，',
            '. ': '。',
            '? ': '？',
            '! ': '！',
            ': ': '：',
            '; ': '；',
            ',': '，',
            '.': '。',
            '?': '？',
            '!': '！',
            ':': '：',
            ';': '；'
        }
        
        for eng, chn in punctuation_map.items():
            text = text.replace(eng, chn)
        
        return text
    
    def _extract_timestamps(self, chunk_text: str, transcript: Transcript) -> tuple[float, float]:
        """
        从转录片段中提取时间戳信息
        
        Args:
            chunk_text: 文本块内容
            transcript: 转录结果
            
        Returns:
            (开始时间, 结束时间)
        """
        if not transcript.segments:
            return 0.0, 0.0
        
        # 简化的时间戳匹配：找到文本块在完整文本中的位置
        full_text = self.clean_text(transcript.full_text)
        chunk_start_pos = full_text.find(chunk_text.strip()[:50])  # 使用前50个字符匹配
        
        if chunk_start_pos == -1:
            # 如果找不到匹配，返回整体时间范围
            return transcript.segments[0].start_time, transcript.segments[-1].end_time
        
        # 根据字符位置估算时间戳
        char_per_second = len(full_text) / (transcript.segments[-1].end_time - transcript.segments[0].start_time)
        estimated_start = transcript.segments[0].start_time + (chunk_start_pos / char_per_second)
        estimated_end = estimated_start + (len(chunk_text) / char_per_second)
        
        # 限制在有效范围内
        estimated_start = max(transcript.segments[0].start_time, estimated_start)
        estimated_end = min(transcript.segments[-1].end_time, estimated_end)
        
        return estimated_start, estimated_end
    
    def _create_text_chunk(self, text: str, transcript: Transcript, chunk_index: int, 
                          start_time: float, end_time: float) -> TextChunk:
        """创建文本块对象"""
        metadata = {
            'language': transcript.language,
            'source_url': transcript.source_audio.source_url if transcript.source_audio.source_url else '',
            'audio_file': str(transcript.source_audio.file_path),
            'total_duration': transcript.source_audio.duration,
            'chunk_length': len(text),
            'processing_time': transcript.processing_time
        }
        
        return TextChunk(
            text=text,
            metadata=metadata,
            start_time=start_time,
            end_time=end_time,
            source_file=str(transcript.source_audio.file_path),
            chunk_index=chunk_index
        )