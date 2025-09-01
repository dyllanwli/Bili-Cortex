import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import tempfile
import asyncio
from pathlib import Path
import json

from src.extractors.video_extractor import VideoExtractor
from src.models import AudioFile


class TestVideoExtractor(unittest.TestCase):
    """VideoExtractor 类的单元测试"""
    
    def setUp(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.extractor = VideoExtractor(
            temp_dir=self.temp_dir,
            quality="best",
            format="wav"
        )
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
    
    def test_validate_url_valid_bilibili(self):
        """测试有效的 B站 URL 验证"""
        valid_urls = [
            "https://www.bilibili.com/video/BV1234567890",
            "https://bilibili.com/video/BV1234567890",
            "https://b23.tv/abc123",
            "https://m.bilibili.com/video/BV1234567890",
        ]
        
        for url in valid_urls:
            with self.subTest(url=url):
                self.assertTrue(
                    self.extractor._validate_url(url),
                    f"URL should be valid: {url}"
                )
    
    def test_validate_url_invalid(self):
        """测试无效 URL 验证"""
        invalid_urls = [
            "https://youtube.com/watch?v=123",
            "https://example.com/video",
            "not_a_url",
            "",
            "http://",
            "https://malicious-site.com/bilibili.com/fake"
        ]
        
        for url in invalid_urls:
            with self.subTest(url=url):
                self.assertFalse(
                    self.extractor._validate_url(url),
                    f"URL should be invalid: {url}"
                )
    
    @patch('yt_dlp.YoutubeDL')
    def test_get_video_info_success(self, mock_ydl):
        """测试获取视频信息成功"""
        # 模拟 yt-dlp 返回的信息
        mock_info = {
            'title': 'Test Video',
            'duration': 120.5,
            'uploader': 'Test Uploader'
        }
        
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = mock_info
        mock_ydl.return_value.__enter__.return_value = mock_ydl_instance
        
        url = "https://www.bilibili.com/video/BV1234567890"
        info = self.extractor._get_video_info(url)
        
        self.assertEqual(info, mock_info)
        mock_ydl_instance.extract_info.assert_called_once_with(url, download=False)
    
    @patch('yt_dlp.YoutubeDL')
    def test_get_video_info_failure(self, mock_ydl):
        """测试获取视频信息失败"""
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.side_effect = Exception("Network error")
        mock_ydl.return_value.__enter__.return_value = mock_ydl_instance
        
        url = "https://www.bilibili.com/video/BV1234567890"
        
        with self.assertRaises(Exception):
            self.extractor._get_video_info(url)
    
    @patch('src.extractors.video_extractor.VideoExtractor._download_with_retry')
    @patch('src.extractors.video_extractor.VideoExtractor._get_video_info')
    def test_extract_audio_success(self, mock_get_info, mock_download):
        """测试音频提取成功"""
        # 创建模拟文件
        mock_audio_path = Path(self.temp_dir) / "test_audio.wav"
        mock_audio_path.write_bytes(b"fake audio data")
        
        # 模拟返回值
        mock_get_info.return_value = {
            'title': 'Test Video',
            'duration': 120.5,
        }
        mock_download.return_value = mock_audio_path
        
        url = "https://www.bilibili.com/video/BV1234567890"
        
        # 使用新的事件循环来避免冲突
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            audio_file = self.extractor.extract_audio(url)
            
            self.assertIsInstance(audio_file, AudioFile)
            self.assertEqual(audio_file.file_path, mock_audio_path)
            self.assertEqual(audio_file.duration, 120.5)
            self.assertEqual(audio_file.source_url, url)
            self.assertEqual(audio_file.size_bytes, len(b"fake audio data"))
        finally:
            loop.close()
    
    def test_extract_audio_invalid_url(self):
        """测试无效 URL 的音频提取"""
        invalid_url = "https://example.com/invalid"
        
        with self.assertRaises(ValueError) as context:
            self.extractor.extract_audio(invalid_url)
        
        self.assertIn("Unsupported or invalid URL", str(context.exception))
    
    @patch('src.extractors.video_extractor.VideoExtractor.extract_audio')
    async def test_batch_extract_success(self, mock_extract):
        """测试批量提取成功"""
        # 模拟 extract_audio 返回值
        mock_audio_files = [
            AudioFile(
                file_path=Path(f"{self.temp_dir}/audio{i}.wav"),
                duration=60.0 + i,
                source_url=f"https://www.bilibili.com/video/BV123456789{i}"
            )
            for i in range(3)
        ]
        
        mock_extract.side_effect = mock_audio_files
        
        urls = [
            "https://www.bilibili.com/video/BV1234567890",
            "https://www.bilibili.com/video/BV1234567891", 
            "https://www.bilibili.com/video/BV1234567892",
        ]
        
        results = await self.extractor.batch_extract(urls)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(mock_extract.call_count, 3)
        for i, audio_file in enumerate(results):
            self.assertIsInstance(audio_file, AudioFile)
            self.assertEqual(audio_file.duration, 60.0 + i)
    
    async def test_batch_extract_mixed_results(self):
        """测试批量提取混合结果（部分成功，部分失败）"""
        with patch.object(self.extractor, 'extract_audio') as mock_extract:
            # 第一个成功，第二个失败，第三个成功
            mock_audio_file_1 = AudioFile(
                file_path=Path(f"{self.temp_dir}/audio1.wav"),
                duration=60.0,
                source_url="https://www.bilibili.com/video/BV1234567890"
            )
            mock_audio_file_3 = AudioFile(
                file_path=Path(f"{self.temp_dir}/audio3.wav"),
                duration=90.0,
                source_url="https://www.bilibili.com/video/BV1234567892"
            )
            
            mock_extract.side_effect = [
                mock_audio_file_1,
                Exception("Download failed"),
                mock_audio_file_3
            ]
            
            urls = [
                "https://www.bilibili.com/video/BV1234567890",
                "https://www.bilibili.com/video/BV1234567891",  # 这个会失败
                "https://www.bilibili.com/video/BV1234567892",
            ]
            
            results = await self.extractor.batch_extract(urls)
            
            # 应该返回 2 个成功的结果
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0], mock_audio_file_1)
            self.assertEqual(results[1], mock_audio_file_3)
    
    def test_batch_extract_invalid_urls_filtered(self):
        """测试批量提取过滤无效 URLs"""
        urls = [
            "https://www.bilibili.com/video/BV1234567890",  # 有效
            "https://youtube.com/watch?v=123",  # 无效
            "https://b23.tv/abc123",  # 有效
            "not_a_url",  # 无效
        ]
        
        # 模拟有效 URL 的处理
        with patch.object(self.extractor, 'extract_audio') as mock_extract:
            mock_extract.return_value = AudioFile(
                file_path=Path(f"{self.temp_dir}/audio.wav"),
                duration=60.0,
                source_url="test"
            )
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(self.extractor.batch_extract(urls))
                
                # 应该只处理 2 个有效的 URL
                self.assertEqual(mock_extract.call_count, 2)
                self.assertEqual(len(results), 2)
            finally:
                loop.close()
    
    def test_cleanup_temp_files(self):
        """测试临时文件清理"""
        # 创建一些模拟的临时文件
        temp_files = []
        for i in range(3):
            temp_path = Path(self.temp_dir) / f"temp_audio_{i}.wav"
            temp_path.write_bytes(b"fake data")
            temp_files.append(AudioFile(
                file_path=temp_path,
                duration=60.0,
                source_url=f"https://www.bilibili.com/video/BV123456789{i}"
            ))
        
        # 确认文件存在
        for audio_file in temp_files:
            self.assertTrue(audio_file.file_path.exists())
        
        # 清理文件
        self.extractor.cleanup_temp_files(temp_files)
        
        # 确认文件已被删除
        for audio_file in temp_files:
            self.assertFalse(audio_file.file_path.exists())
    
    def test_cleanup_temp_files_nonexistent(self):
        """测试清理不存在的文件（应该不抛异常）"""
        nonexistent_files = [
            AudioFile(
                file_path=Path(self.temp_dir) / "nonexistent.wav",
                duration=60.0,
                source_url="https://www.bilibili.com/video/BV1234567890"
            )
        ]
        
        # 这应该不抛异常
        self.extractor.cleanup_temp_files(nonexistent_files)


if __name__ == '__main__':
    unittest.main()