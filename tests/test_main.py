import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import tempfile
import asyncio
from pathlib import Path
import json

from src.main import BiliCortexProcessor, parse_args, load_urls_from_file
from src.models import AudioFile, Transcript, TranscriptSegment


class TestBiliCortexProcessor(unittest.TestCase):
    """BiliCortexProcessor 主处理器的单元测试"""
    
    def setUp(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        
        # 模拟所有依赖组件
        with patch('src.main.get_settings') as mock_settings, \
             patch('src.main.VideoExtractor') as mock_extractor_class, \
             patch('src.main.WhisperTranscriber') as mock_transcriber_class:
            
            # 设置模拟配置
            mock_config = MagicMock()
            mock_config.audio.temp_dir = self.temp_dir
            mock_config.audio.quality = "best"
            mock_config.audio.format = "wav"
            mock_config.security.max_url_length = 2048
            mock_config.security.allowed_domains = ['bilibili.com', 'www.bilibili.com']
            mock_config.security.enable_url_validation = True
            mock_config.system.transcripts_dir = f"{self.temp_dir}/transcripts"
            mock_config.system.cleanup_temp_files = True
            mock_config.get_whisper_opts.return_value = {
                'model_name': 'tiny',
                'language': 'zh',
                'device': 'cpu'
            }
            mock_config.to_dict.return_value = {'test': 'config'}
            mock_settings.return_value = mock_config
            
            # 设置模拟组件
            self.mock_extractor = MagicMock()
            self.mock_transcriber = MagicMock()
            mock_extractor_class.return_value = self.mock_extractor
            mock_transcriber_class.return_value = self.mock_transcriber
            
            # 创建处理器
            self.processor = BiliCortexProcessor()
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
    
    def test_validate_urls_valid(self):
        """测试有效 URL 验证"""
        urls = [
            "https://www.bilibili.com/video/BV1234567890",
            "https://bilibili.com/video/BV1234567891"
        ]
        
        valid_urls = self.processor.validate_urls(urls)
        self.assertEqual(len(valid_urls), 2)
        self.assertEqual(valid_urls, urls)
    
    def test_validate_urls_invalid_domain(self):
        """测试无效域名过滤"""
        urls = [
            "https://www.bilibili.com/video/BV1234567890",  # 有效
            "https://youtube.com/watch?v=123",  # 无效域名
            "https://bilibili.com/video/BV1234567891",  # 有效
        ]
        
        valid_urls = self.processor.validate_urls(urls)
        self.assertEqual(len(valid_urls), 2)
        self.assertIn("https://www.bilibili.com/video/BV1234567890", valid_urls)
        self.assertIn("https://bilibili.com/video/BV1234567891", valid_urls)
        self.assertNotIn("https://youtube.com/watch?v=123", valid_urls)
    
    def test_validate_urls_too_long(self):
        """测试过长 URL 过滤"""
        long_url = "https://www.bilibili.com/video/" + "x" * 2050
        urls = [
            "https://www.bilibili.com/video/BV1234567890",  # 有效
            long_url,  # 过长
        ]
        
        valid_urls = self.processor.validate_urls(urls)
        self.assertEqual(len(valid_urls), 1)
        self.assertIn("https://www.bilibili.com/video/BV1234567890", valid_urls)
    
    def test_process_urls_success(self):
        """测试处理 URL 成功"""
        urls = ["https://www.bilibili.com/video/BV1234567890"]
        
        # 创建模拟音频文件
        audio_file = AudioFile(
            file_path=Path(self.temp_dir) / "test_audio.wav",
            duration=120.0,
            source_url=urls[0]
        )
        
        # 创建模拟转录结果
        transcript = Transcript(
            segments=[
                TranscriptSegment("测试文本", 0.0, 5.0, -0.5)
            ],
            language="zh",
            source_audio=audio_file,
            full_text="测试文本",
            processing_time=2.0
        )
        
        # 设置模拟返回值
        self.mock_extractor.extract_audio.return_value = audio_file
        self.mock_transcriber.batch_transcribe.return_value = [transcript]
        self.mock_transcriber.estimate_processing_time.return_value = 10.0
        
        # 执行处理
        results = asyncio.run(self.processor.process_urls(urls))
        
        # 验证结果
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], transcript)
        
        # 验证调用
        self.mock_extractor.extract_audio.assert_called_once_with(urls[0])
        self.mock_transcriber.batch_transcribe.assert_called_once_with([audio_file])
    
    def test_process_urls_batch(self):
        """测试批量处理 URL"""
        urls = [
            "https://www.bilibili.com/video/BV1234567890",
            "https://www.bilibili.com/video/BV1234567891"
        ]
        
        # 创建模拟音频文件
        audio_files = [
            AudioFile(
                file_path=Path(self.temp_dir) / f"test_audio_{i}.wav",
                duration=120.0,
                source_url=url
            ) for i, url in enumerate(urls)
        ]
        
        # 创建模拟转录结果
        transcripts = [
            Transcript(
                segments=[TranscriptSegment(f"文本 {i}", 0.0, 5.0)],
                language="zh",
                source_audio=audio_file,
                full_text=f"文本 {i}"
            ) for i, audio_file in enumerate(audio_files)
        ]
        
        # 设置模拟返回值 (batch_extract 是异步方法)
        async def mock_batch_extract(urls):
            return audio_files
        
        self.mock_extractor.batch_extract = mock_batch_extract
        self.mock_transcriber.batch_transcribe.return_value = transcripts
        self.mock_transcriber.estimate_processing_time.return_value = 20.0
        
        # 执行处理
        results = asyncio.run(self.processor.process_urls(urls))
        
        # 验证结果
        self.assertEqual(len(results), 2)
        self.assertEqual(results, transcripts)
        
        # 验证调用
        self.mock_extractor.batch_extract.assert_called_once_with(urls)
        self.mock_transcriber.batch_transcribe.assert_called_once_with(audio_files)
    
    def test_process_urls_no_valid_urls(self):
        """测试没有有效 URL 的情况"""
        invalid_urls = ["https://youtube.com/watch?v=123"]
        
        results = asyncio.run(self.processor.process_urls(invalid_urls))
        
        self.assertEqual(len(results), 0)
        # 不应该调用提取器和转录器
        self.mock_extractor.extract_audio.assert_not_called()
        self.mock_extractor.batch_extract.assert_not_called()
        self.mock_transcriber.batch_transcribe.assert_not_called()
    
    def test_process_urls_extraction_failure(self):
        """测试音频提取失败"""
        urls = ["https://www.bilibili.com/video/BV1234567890"]
        
        # 设置提取器返回空列表
        self.mock_extractor.extract_audio.return_value = []
        self.mock_transcriber.estimate_processing_time.return_value = 0.0
        
        results = asyncio.run(self.processor.process_urls(urls))
        
        self.assertEqual(len(results), 0)
        # 转录器不应该被调用
        self.mock_transcriber.batch_transcribe.assert_not_called()
    
    def test_save_transcripts(self):
        """测试保存转录结果"""
        # 创建转录结果
        audio_file = AudioFile(
            file_path=Path(self.temp_dir) / "test_audio.wav",
            source_url="https://www.bilibili.com/video/BV1234567890"
        )
        
        transcript = Transcript(
            segments=[TranscriptSegment("测试", 0.0, 2.0)],
            language="zh",
            source_audio=audio_file,
            full_text="测试"
        )
        
        # 模拟保存方法
        self.mock_transcriber.save_transcript = MagicMock()
        
        asyncio.run(self.processor._save_transcripts([transcript]))
        
        # 验证保存方法被调用
        self.mock_transcriber.save_transcript.assert_called_once()
        
        # 检查调用参数
        args, kwargs = self.mock_transcriber.save_transcript.call_args
        self.assertEqual(args[0], transcript)
        self.assertIsInstance(args[1], Path)
        self.assertTrue(args[1].name.startswith("transcript_"))
    
    def test_process_single_url_success(self):
        """测试单个 URL 处理成功"""
        url = "https://www.bilibili.com/video/BV1234567890"
        
        # 创建模拟转录结果
        transcript = Transcript(
            segments=[TranscriptSegment("测试", 0.0, 2.0)],
            language="zh",
            source_audio=AudioFile(Path("test.wav")),
            full_text="测试"
        )
        
        with patch.object(self.processor, 'process_urls', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = [transcript]
            
            result = self.processor.process_single_url(url)
            
            self.assertEqual(result, transcript)
    
    def test_process_single_url_failure(self):
        """测试单个 URL 处理失败"""
        url = "https://www.bilibili.com/video/BV1234567890"
        
        with patch.object(self.processor, 'process_urls', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = []  # 空结果
            
            result = self.processor.process_single_url(url)
            
            self.assertIsNone(result)
    
    def test_get_system_info(self):
        """测试获取系统信息"""
        self.mock_transcriber.get_model_info.return_value = {'model': 'tiny'}
        
        info = self.processor.get_system_info()
        
        self.assertIn("extractor_info", info)
        self.assertIn("transcriber_info", info)
        self.assertIn("settings", info)
        
        self.assertEqual(info["transcriber_info"], {'model': 'tiny'})
        self.assertEqual(info["settings"], {'test': 'config'})


class TestArgParsing(unittest.TestCase):
    """测试命令行参数解析"""
    
    def test_parse_args_basic(self):
        """测试基本参数解析"""
        # 模拟命令行参数
        test_args = ["https://www.bilibili.com/video/BV1234567890"]
        
        with patch('sys.argv', ['main.py'] + test_args):
            args = parse_args()
            
            self.assertEqual(args.urls, test_args)
            self.assertEqual(args.log_level, 'INFO')
            self.assertFalse(args.no_save)
            self.assertFalse(args.info)
            self.assertIsNone(args.from_file)
    
    def test_parse_args_multiple_urls(self):
        """测试多个 URL 参数"""
        test_args = [
            "https://www.bilibili.com/video/BV1234567890",
            "https://www.bilibili.com/video/BV1234567891"
        ]
        
        with patch('sys.argv', ['main.py'] + test_args):
            args = parse_args()
            
            self.assertEqual(args.urls, test_args)
    
    def test_parse_args_from_file(self):
        """测试从文件读取参数"""
        test_args = ["--from-file", "urls.txt"]
        
        with patch('sys.argv', ['main.py'] + test_args):
            args = parse_args()
            
            self.assertEqual(args.from_file, "urls.txt")
    
    def test_parse_args_no_save(self):
        """测试不保存参数"""
        test_args = ["--no-save", "https://www.bilibili.com/video/BV1234567890"]
        
        with patch('sys.argv', ['main.py'] + test_args):
            args = parse_args()
            
            self.assertTrue(args.no_save)
    
    def test_parse_args_log_level(self):
        """测试日志级别参数"""
        test_args = ["--log-level", "DEBUG", "https://www.bilibili.com/video/BV1234567890"]
        
        with patch('sys.argv', ['main.py'] + test_args):
            args = parse_args()
            
            self.assertEqual(args.log_level, 'DEBUG')
    
    def test_parse_args_info(self):
        """测试信息参数"""
        test_args = ["--info"]
        
        with patch('sys.argv', ['main.py'] + test_args):
            args = parse_args()
            
            self.assertTrue(args.info)


class TestLoadUrlsFromFile(unittest.TestCase):
    """测试从文件加载 URL"""
    
    def setUp(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
    
    def test_load_urls_from_file_success(self):
        """测试成功从文件加载 URL"""
        urls_content = """
        https://www.bilibili.com/video/BV1234567890
        https://www.bilibili.com/video/BV1234567891
        # 这是注释
        
        https://www.bilibili.com/video/BV1234567892
        """
        
        urls_file = Path(self.temp_dir) / "urls.txt"
        urls_file.write_text(urls_content, encoding='utf-8')
        
        urls = load_urls_from_file(str(urls_file))
        
        expected_urls = [
            "https://www.bilibili.com/video/BV1234567890",
            "https://www.bilibili.com/video/BV1234567891",
            "https://www.bilibili.com/video/BV1234567892"
        ]
        
        self.assertEqual(urls, expected_urls)
    
    def test_load_urls_from_file_empty(self):
        """测试从空文件加载 URL"""
        urls_file = Path(self.temp_dir) / "empty.txt"
        urls_file.write_text("", encoding='utf-8')
        
        urls = load_urls_from_file(str(urls_file))
        
        self.assertEqual(urls, [])
    
    def test_load_urls_from_file_not_exist(self):
        """测试加载不存在的文件"""
        urls = load_urls_from_file("/nonexistent/file.txt")
        
        self.assertEqual(urls, [])
    
    def test_load_urls_from_file_only_comments(self):
        """测试只有注释的文件"""
        urls_content = """
        # 注释行 1
        # 注释行 2
        
        # 更多注释
        """
        
        urls_file = Path(self.temp_dir) / "comments.txt"
        urls_file.write_text(urls_content, encoding='utf-8')
        
        urls = load_urls_from_file(str(urls_file))
        
        self.assertEqual(urls, [])


if __name__ == '__main__':
    unittest.main()