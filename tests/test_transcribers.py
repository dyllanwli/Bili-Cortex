import unittest
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path
from typing import List

from src.transcribers.whisper_transcriber import WhisperTranscriber
from src.models import AudioFile, Transcript, TranscriptSegment


class MockWhisperSegment:
    """模拟 Whisper 输出段落"""
    def __init__(self, text: str, start: float, end: float, avg_logprob: float = -0.5):
        self.text = text
        self.start = start
        self.end = end
        self.avg_logprob = avg_logprob


class MockWhisperInfo:
    """模拟 Whisper 输出信息"""
    def __init__(self, language: str = "zh"):
        self.language = language


class TestWhisperTranscriber(unittest.TestCase):
    """WhisperTranscriber 类的单元测试"""
    
    def setUp(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        
        # 模拟 WhisperModel 以避免真实模型加载
        self.mock_model = MagicMock()
        with patch('faster_whisper.WhisperModel', return_value=self.mock_model):
            self.transcriber = WhisperTranscriber(
                model_name="tiny",
                language="zh",
                device="cpu"
            )
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
    
    def test_setup_device_auto_cuda_available(self):
        """测试自动设备选择 - CUDA 可用"""
        with patch('torch.cuda.is_available', return_value=True):
            device = self.transcriber._setup_device("auto")
            self.assertEqual(device, "cuda")
    
    def test_setup_device_auto_cuda_unavailable(self):
        """测试自动设备选择 - CUDA 不可用"""
        with patch('torch.cuda.is_available', return_value=False):
            device = self.transcriber._setup_device("auto")
            self.assertEqual(device, "cpu")
    
    def test_setup_device_explicit(self):
        """测试显式设备选择"""
        device = self.transcriber._setup_device("cpu")
        self.assertEqual(device, "cpu")
    
    def test_clean_text_basic(self):
        """测试基本文本清洗"""
        test_cases = [
            ("  hello world  ", "hello world"),
            ("hello\n\nworld", "hello world"),
            ("多个。。。句号", "多个。句号"),
            ("多个，，，逗号", "多个，逗号"),
            ("", ""),
            ("   \n\r   ", ""),
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = self.transcriber._clean_text(input_text)
                self.assertEqual(result, expected)
    
    def test_validate_audio_file_valid(self):
        """测试有效音频文件验证"""
        # 创建临时音频文件
        audio_path = Path(self.temp_dir) / "test_audio.wav"
        audio_path.write_bytes(b"fake audio data")
        
        audio_file = AudioFile(
            file_path=audio_path,
            duration=60.0,
            size_bytes=len(b"fake audio data")
        )
        
        self.assertTrue(self.transcriber._validate_audio_file(audio_file))
    
    def test_validate_audio_file_not_exists(self):
        """测试不存在的音频文件验证"""
        audio_file = AudioFile(
            file_path=Path(self.temp_dir) / "nonexistent.wav",
            duration=60.0
        )
        
        self.assertFalse(self.transcriber._validate_audio_file(audio_file))
    
    def test_validate_audio_file_empty(self):
        """测试空音频文件验证"""
        # 创建空文件
        audio_path = Path(self.temp_dir) / "empty_audio.wav"
        audio_path.write_bytes(b"")
        
        audio_file = AudioFile(
            file_path=audio_path,
            duration=60.0,
            size_bytes=0
        )
        
        self.assertFalse(self.transcriber._validate_audio_file(audio_file))
    
    @patch('faster_whisper.WhisperModel')
    def test_transcribe_success(self, mock_whisper_model):
        """测试转录成功"""
        # 创建模拟音频文件
        audio_path = Path(self.temp_dir) / "test_audio.wav"
        audio_path.write_bytes(b"fake audio data")
        
        audio_file = AudioFile(
            file_path=audio_path,
            duration=120.0,
            source_url="https://www.bilibili.com/video/BV1234567890"
        )
        
        # 模拟 Whisper 输出
        mock_segments = [
            MockWhisperSegment("你好", 0.0, 1.5, -0.3),
            MockWhisperSegment("世界", 1.5, 3.0, -0.4),
            MockWhisperSegment("  ", 3.0, 3.5, -1.0),  # 空白段落，应该被过滤
        ]
        
        mock_info = MockWhisperInfo("zh")
        
        # 设置模拟方法
        mock_model_instance = MagicMock()
        mock_model_instance.transcribe.return_value = (mock_segments, mock_info)
        mock_whisper_model.return_value = mock_model_instance
        
        # 更新 transcriber 的模型
        self.transcriber.model = mock_model_instance
        
        # 执行转录
        transcript = self.transcriber.transcribe(audio_file)
        
        # 验证结果
        self.assertIsInstance(transcript, Transcript)
        self.assertEqual(len(transcript.segments), 2)  # 空白段落被过滤
        self.assertEqual(transcript.segments[0].text, "你好")
        self.assertEqual(transcript.segments[1].text, "世界") 
        self.assertEqual(transcript.language, "zh")
        self.assertEqual(transcript.source_audio, audio_file)
        self.assertEqual(transcript.full_text, "你好 世界")
        self.assertIsNotNone(transcript.processing_time)
    
    def test_transcribe_invalid_file(self):
        """测试转录无效文件"""
        audio_file = AudioFile(
            file_path=Path(self.temp_dir) / "nonexistent.wav",
            duration=60.0
        )
        
        with self.assertRaises(ValueError):
            self.transcriber.transcribe(audio_file)
    
    @patch.object(WhisperTranscriber, 'transcribe')
    def test_batch_transcribe_success(self, mock_transcribe):
        """测试批量转录成功"""
        # 创建模拟音频文件
        audio_files = []
        for i in range(3):
            audio_path = Path(self.temp_dir) / f"test_audio_{i}.wav"
            audio_path.write_bytes(b"fake audio data")
            
            audio_files.append(AudioFile(
                file_path=audio_path,
                duration=60.0 + i * 10,
                source_url=f"https://www.bilibili.com/video/BV123456789{i}"
            ))
        
        # 模拟 transcribe 返回值
        mock_transcripts = []
        for i, audio_file in enumerate(audio_files):
            mock_transcript = Transcript(
                segments=[
                    TranscriptSegment(f"测试文本 {i}", 0.0, 5.0, -0.5)
                ],
                language="zh",
                source_audio=audio_file,
                full_text=f"测试文本 {i}",
                processing_time=2.0
            )
            mock_transcripts.append(mock_transcript)
        
        mock_transcribe.side_effect = mock_transcripts
        
        # 执行批量转录
        results = self.transcriber.batch_transcribe(audio_files)
        
        # 验证结果
        self.assertEqual(len(results), 3)
        self.assertEqual(mock_transcribe.call_count, 3)
        
        for i, transcript in enumerate(results):
            self.assertIsInstance(transcript, Transcript)
            self.assertEqual(transcript.full_text, f"测试文本 {i}")
    
    def test_batch_transcribe_mixed_results(self):
        """测试批量转录混合结果"""
        # 创建模拟音频文件
        audio_files = []
        for i in range(3):
            if i == 1:  # 第二个文件无效
                audio_path = Path(self.temp_dir) / f"invalid_audio_{i}.wav"
                # 不创建文件
            else:
                audio_path = Path(self.temp_dir) / f"test_audio_{i}.wav"
                audio_path.write_bytes(b"fake audio data")
            
            audio_files.append(AudioFile(
                file_path=audio_path,
                duration=60.0,
                source_url=f"https://www.bilibili.com/video/BV123456789{i}"
            ))
        
        # 执行批量转录
        with patch.object(self.transcriber, 'transcribe') as mock_transcribe:
            # 第一个和第三个成功，第二个失败
            success_transcript = Transcript(
                segments=[TranscriptSegment("成功", 0.0, 2.0)],
                language="zh",
                source_audio=audio_files[0],  # 使用第一个文件作为示例
                full_text="成功"
            )
            
            mock_transcribe.side_effect = [
                success_transcript,  # 第一个成功
                Exception("转录失败"),  # 第二个失败
                success_transcript   # 第三个成功
            ]
            
            results = self.transcriber.batch_transcribe(audio_files)
            
            # 只有 1 个有效文件被处理（第二个文件验证失败，第三个文件在转录时失败）
            # 实际上，第二个文件会在验证阶段被过滤掉
            self.assertEqual(len(results), 1)  # 只有第一个成功
    
    def test_batch_transcribe_empty_list(self):
        """测试空列表的批量转录"""
        results = self.transcriber.batch_transcribe([])
        self.assertEqual(len(results), 0)
    
    def test_get_model_info(self):
        """测试获取模型信息"""
        info = self.transcriber.get_model_info()
        
        expected_keys = ["model_name", "language", "device", "compute_type", "batch_size"]
        for key in expected_keys:
            self.assertIn(key, info)
        
        self.assertEqual(info["model_name"], "tiny")
        self.assertEqual(info["language"], "zh")
        self.assertEqual(info["device"], "cpu")
    
    def test_estimate_processing_time(self):
        """测试处理时间估算"""
        audio_files = [
            AudioFile(Path("test1.wav"), duration=60.0),  # 1分钟
            AudioFile(Path("test2.wav"), duration=120.0), # 2分钟
            AudioFile(Path("test3.wav"), duration=None),  # 无时长信息
        ]
        
        estimated_time = self.transcriber.estimate_processing_time(audio_files)
        
        # CPU 设备，估算系数为 0.3
        # 总时长 = 60 + 120 = 180 秒
        # 估算时间 = 180 * 0.3 = 54 秒
        expected_time = 180.0 * 0.3
        self.assertAlmostEqual(estimated_time, expected_time, places=1)
    
    def test_save_transcript(self):
        """测试保存转录结果"""
        # 创建测试转录结果
        audio_file = AudioFile(
            file_path=Path("test_audio.wav"),
            source_url="https://www.bilibili.com/video/BV1234567890"
        )
        
        transcript = Transcript(
            segments=[
                TranscriptSegment("第一段文本", 0.0, 2.5, -0.3),
                TranscriptSegment("第二段文本", 2.5, 5.0, -0.4),
            ],
            language="zh",
            source_audio=audio_file,
            full_text="第一段文本 第二段文本",
            processing_time=3.2
        )
        
        # 保存文件
        output_path = Path(self.temp_dir) / "transcript.md"
        self.transcriber.save_transcript(transcript, output_path)
        
        # 验证文件存在
        self.assertTrue(output_path.exists())
        
        # 验证文件内容
        content = output_path.read_text(encoding='utf-8')
        
        self.assertIn("# 转录结果", content)
        self.assertIn("语言: zh", content)
        self.assertIn("处理时间: 3.2秒", content)
        self.assertIn("第一段文本 第二段文本", content)
        self.assertIn("第一段文本", content)
        self.assertIn("第二段文本", content)
        self.assertIn("[  0.0s -   2.5s]", content)
        self.assertIn("[  2.5s -   5.0s]", content)
    
    @patch('faster_whisper.WhisperModel')
    def test_model_fallback(self, mock_whisper_model):
        """测试模型降级机制"""
        # 第一次初始化失败，第二次成功
        mock_whisper_model.side_effect = [
            Exception("Model loading failed"),  # large-v3 失败
            MagicMock()  # medium 成功
        ]
        
        with patch('torch.cuda.is_available', return_value=False):
            transcriber = WhisperTranscriber(model_name="large-v3")
            
            # 应该降级到 medium 模型
            self.assertEqual(transcriber.model_name, "medium")


if __name__ == '__main__':
    unittest.main()