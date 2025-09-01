import unittest
from unittest.mock import Mock, patch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.processors.text_processor import TextProcessor
from src.models import Transcript, TranscriptSegment, AudioFile, TextChunk


class TestTextProcessor(unittest.TestCase):
    """测试 TextProcessor 类"""
    
    def setUp(self):
        """测试前准备"""
        self.processor = TextProcessor(
            chunk_size=100,  # 使用较小的块大小便于测试
            chunk_overlap=20,
            min_chunk_size=30
        )
        
        # 创建模拟的音频文件
        self.mock_audio = AudioFile(
            file_path=Path("/mock/audio.wav"),
            duration=120.0,
            source_url="https://www.bilibili.com/video/BV123456789"
        )
        
        # 创建模拟的转录片段
        self.mock_segments = [
            TranscriptSegment("这是第一段转录文本内容", 0.0, 10.0, 0.95),
            TranscriptSegment("这是第二段转录文本内容", 10.0, 20.0, 0.92),
            TranscriptSegment("这是第三段转录文本内容，包含了更多的详细信息", 20.0, 35.0, 0.98),
        ]
        
        # 创建模拟的转录结果
        self.mock_transcript = Transcript(
            segments=self.mock_segments,
            language="zh",
            source_audio=self.mock_audio,
            full_text="",
            processing_time=15.5
        )
    
    def test_clean_text_basic(self):
        """测试基本的文本清洗功能"""
        dirty_text = "  这是   一个  测试文本，，，包含多余空格和   重复标点！！！  "
        cleaned = self.processor.clean_text(dirty_text)
        
        self.assertEqual(cleaned, "这是 一个 测试文本，包含多余空格和 重复标点！")
    
    def test_clean_text_filler_words(self):
        """测试填充词过滤"""
        text_with_fillers = "嗯 这个 就是 那个 很好的 呃 内容 对吧"
        cleaned = self.processor.clean_text(text_with_fillers)
        
        # 检查填充词是否被移除
        self.assertNotIn("嗯", cleaned)
        self.assertNotIn("这个", cleaned)
        self.assertNotIn("那个", cleaned)
        self.assertNotIn("呃", cleaned)
        self.assertIn("很好的", cleaned)
        self.assertIn("内容", cleaned)
    
    def test_clean_text_punctuation_normalization(self):
        """测试标点符号标准化"""
        text = "Hello, world! How are you? I'm fine."
        cleaned = self.processor.clean_text(text)
        
        # 检查英文标点是否转换为中文标点
        self.assertIn("，", cleaned)
        self.assertIn("。", cleaned)
        self.assertIn("？", cleaned)
        self.assertIn("！", cleaned)
    
    def test_clean_text_empty_input(self):
        """测试空输入"""
        self.assertEqual(self.processor.clean_text(""), "")
        self.assertEqual(self.processor.clean_text(None), "")
        self.assertEqual(self.processor.clean_text("   "), "")
    
    def test_chunk_text_basic(self):
        """测试基本的文本分块功能"""
        chunks = self.processor.chunk_text(self.mock_transcript)
        
        # 验证返回了文本块
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        
        # 验证每个块都是 TextChunk 实例
        for chunk in chunks:
            self.assertIsInstance(chunk, TextChunk)
            self.assertIsInstance(chunk.text, str)
            self.assertIsInstance(chunk.metadata, dict)
            self.assertIsNotNone(chunk.chunk_index)
    
    def test_chunk_text_metadata(self):
        """测试文本块元数据"""
        chunks = self.processor.chunk_text(self.mock_transcript)
        
        if chunks:
            chunk = chunks[0]
            
            # 验证元数据包含预期字段
            self.assertIn('language', chunk.metadata)
            self.assertIn('audio_file', chunk.metadata)
            self.assertIn('total_duration', chunk.metadata)
            self.assertIn('chunk_length', chunk.metadata)
            
            # 验证元数据值
            self.assertEqual(chunk.metadata['language'], 'zh')
            self.assertEqual(chunk.metadata['total_duration'], 120.0)
            self.assertGreater(chunk.metadata['chunk_length'], 0)
    
    def test_chunk_text_short_text(self):
        """测试短文本处理"""
        # 创建一个非常短的转录
        short_segments = [TranscriptSegment("短文本", 0.0, 5.0, 0.95)]
        short_transcript = Transcript(
            segments=short_segments,
            language="zh",
            source_audio=self.mock_audio,
            full_text="短文本",
            processing_time=1.0
        )
        
        chunks = self.processor.chunk_text(short_transcript)
        
        # 对于短于最小块大小的文本，应该返回单个块
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text.strip(), "短文本")
    
    def test_chunk_text_empty_transcript(self):
        """测试空转录处理"""
        empty_transcript = Transcript(
            segments=[],
            language="zh",
            source_audio=self.mock_audio,
            full_text="",
            processing_time=0.0
        )
        
        chunks = self.processor.chunk_text(empty_transcript)
        self.assertEqual(chunks, [])
    
    def test_extract_timestamps(self):
        """测试时间戳提取"""
        sample_text = "这是第一段转录文本内容"
        start_time, end_time = self.processor._extract_timestamps(sample_text, self.mock_transcript)
        
        # 验证返回的是数值
        self.assertIsInstance(start_time, float)
        self.assertIsInstance(end_time, float)
        
        # 验证时间范围合理
        self.assertGreaterEqual(start_time, 0.0)
        self.assertGreater(end_time, start_time)
        self.assertLessEqual(end_time, 35.0)  # 不应该超过最后一个片段的结束时间
    
    def test_create_text_chunk(self):
        """测试 TextChunk 对象创建"""
        test_text = "测试文本块"
        chunk = self.processor._create_text_chunk(
            text=test_text,
            transcript=self.mock_transcript,
            chunk_index=0,
            start_time=5.0,
            end_time=10.0
        )
        
        # 验证 TextChunk 属性
        self.assertEqual(chunk.text, test_text)
        self.assertEqual(chunk.start_time, 5.0)
        self.assertEqual(chunk.end_time, 10.0)
        self.assertEqual(chunk.chunk_index, 0)
        self.assertIn('language', chunk.metadata)
        self.assertEqual(chunk.metadata['language'], 'zh')
    
    def test_normalize_punctuation(self):
        """测试标点符号标准化私有方法"""
        text = "Hello, world! How are you? I'm fine; really."
        normalized = self.processor._normalize_punctuation(text)
        
        expected = "Hello，world！How are you？I'm fine；really。"
        self.assertEqual(normalized, expected)


class TestTextProcessorIntegration(unittest.TestCase):
    """TextProcessor 集成测试"""
    
    def test_full_processing_pipeline(self):
        """测试完整的处理流程"""
        processor = TextProcessor(chunk_size=200, chunk_overlap=50, min_chunk_size=50)
        
        # 创建较长的转录内容用于测试
        long_segments = [
            TranscriptSegment("这是一个很长的转录文本内容，包含了很多有用的信息。" * 5, 0.0, 30.0, 0.95),
            TranscriptSegment("这里是第二部分的内容，同样包含了大量的文本信息。" * 5, 30.0, 60.0, 0.92),
            TranscriptSegment("最后一部分的内容，总结了前面的所有信息。" * 5, 60.0, 90.0, 0.98),
        ]
        
        audio_file = AudioFile(
            file_path=Path("/test/long_audio.wav"),
            duration=90.0,
            source_url="https://www.bilibili.com/video/BV987654321"
        )
        
        transcript = Transcript(
            segments=long_segments,
            language="zh",
            source_audio=audio_file,
            full_text="",
            processing_time=25.0
        )
        
        # 处理转录
        chunks = processor.chunk_text(transcript)
        
        # 验证结果
        self.assertGreater(len(chunks), 1)  # 应该产生多个块
        
        # 验证所有块都满足最小大小要求
        for chunk in chunks:
            self.assertGreaterEqual(len(chunk.text), processor.min_chunk_size)
        
        # 验证块有正确的索引
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.chunk_index, i)
        
        # 验证时间戳是递增的
        for i in range(1, len(chunks)):
            if chunks[i].start_time and chunks[i-1].end_time:
                self.assertGreaterEqual(chunks[i].start_time, chunks[i-1].start_time)


if __name__ == '__main__':
    unittest.main()