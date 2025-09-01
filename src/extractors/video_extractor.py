import asyncio
import logging
import re
import tempfile
import time
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import yt_dlp
from src.models import AudioFile


class VideoExtractor:
    """视频音频提取器，基于 yt-dlp"""
    
    SUPPORTED_PLATFORMS = [
        'bilibili.com',
        'www.bilibili.com',
        'b23.tv',
        'm.bilibili.com',
        'youtube.com',
        'www.youtube.com',
        'youtu.be',
        'm.youtube.com'
    ]
    
    def __init__(self, temp_dir: str = "./data/temp", quality: str = "best", format: str = "wav"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.quality = quality
        self.format = format
        self.logger = logging.getLogger(__name__)
        
        # yt-dlp 配置
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(self.temp_dir / '%(title)s.%(ext)s'),
            'noplaylist': True,
            'extractaudio': True,
            'audioformat': format,
            'audioquality': 0,  # 最高质量
            'quiet': True,
            'no_warnings': True,
        }
    
    def _validate_url(self, url: str) -> bool:
        """验证 URL 格式和平台支持"""
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
            
            # 检查是否为支持的平台
            domain = parsed.netloc.lower()
            return any(platform in domain for platform in self.SUPPORTED_PLATFORMS)
        except Exception:
            return False
    
    def _get_video_info(self, url: str) -> dict:
        """获取视频信息"""
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
                return info
            except Exception as e:
                self.logger.error(f"Failed to extract info from {url}: {e}")
                raise
    
    async def _download_audio(self, url: str) -> Optional[Path]:
        """简化的音频下载"""
        try:
            # 使用当前时间戳作为文件名前缀
            temp_opts = self.ydl_opts.copy()
            temp_opts['outtmpl'] = str(self.temp_dir / f'audio_{int(time.time())}.%(ext)s')
            
            with yt_dlp.YoutubeDL(temp_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if info:
                    expected_path = Path(ydl.prepare_filename(info))
                    if expected_path.exists():
                        return expected_path
                    
                    # 在临时目录中查找最新文件
                    audio_files = list(self.temp_dir.glob('audio_*.*'))
                    if audio_files:
                        return max(audio_files, key=lambda p: p.stat().st_mtime)
        except Exception as e:
            self.logger.error(f"Failed to download audio from {url}: {e}")
            raise
        
        return None
    
    def extract_audio(self, url: str) -> AudioFile:
        """提取单个视频的音频"""
        if not self._validate_url(url):
            raise ValueError(f"Unsupported or invalid URL: {url}")
        
        self.logger.info(f"Extracting audio from: {url}")
        
        try:
            # 获取视频信息
            info = self._get_video_info(url)
            
            # 简化的下载逻辑
            try:
                loop = asyncio.get_running_loop()
                # 在已有的事件循环中
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._download_audio(url))
                    audio_path = future.result()
            except RuntimeError:
                # 没有运行中的事件循环
                audio_path = asyncio.run(self._download_audio(url))
            
            if not audio_path or not audio_path.exists():
                raise RuntimeError(f"Failed to download audio from {url}")
            
            # 获取文件信息
            file_stats = audio_path.stat()
            
            return AudioFile(
                file_path=audio_path,
                duration=info.get('duration'),
                format=self.format,
                size_bytes=file_stats.st_size,
                source_url=url
            )
            
        except Exception as e:
            self.logger.error(f"Failed to extract audio from {url}: {e}")
            raise
    
    async def batch_extract(self, urls: List[str]) -> List[AudioFile]:
        """简化的批量提取音频"""
        self.logger.info(f"Starting batch extraction of {len(urls)} URLs")
        
        # 验证所有 URLs
        valid_urls = [url for url in urls if self._validate_url(url)]
        invalid_count = len(urls) - len(valid_urls)
        
        if invalid_count > 0:
            self.logger.warning(f"Skipping {invalid_count} invalid URLs")
        
        if not valid_urls:
            return []
        
        # 简化的序列处理 - 减少复杂性
        audio_files = []
        failed_count = 0
        
        for i, url in enumerate(valid_urls, 1):
            try:
                self.logger.info(f"Processing URL {i}/{len(valid_urls)}: {url}")
                audio_file = self.extract_audio(url)
                audio_files.append(audio_file)
            except Exception as e:
                self.logger.error(f"Failed to extract audio from {url}: {e}")
                failed_count += 1
        
        self.logger.info(
            f"Batch extraction completed. Success: {len(audio_files)}, Failed: {failed_count}"
        )
        return audio_files
    
    def cleanup_temp_files(self, audio_files: List[AudioFile]) -> None:
        """清理临时文件"""
        for audio_file in audio_files:
            try:
                if audio_file.file_path.exists():
                    audio_file.file_path.unlink()
                    self.logger.debug(f"Cleaned up temp file: {audio_file.file_path}")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup {audio_file.file_path}: {e}")
    
    def __del__(self):
        """析构函数，清理临时文件"""
        try:
            # 清理音频临时文件
            for temp_file in self.temp_dir.glob("audio_*.*"):
                temp_file.unlink()
        except Exception:
            pass  # 忽略清理错误