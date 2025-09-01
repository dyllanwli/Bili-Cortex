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
        'm.bilibili.com'
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
    
    async def _download_with_retry(self, url: str, max_retries: int = 3) -> Optional[Path]:
        """带重试机制的下载，使用指数退避"""
        for attempt in range(max_retries):
            try:
                # 使用临时文件名避免冲突
                temp_opts = self.ydl_opts.copy()
                temp_opts['outtmpl'] = str(self.temp_dir / f'temp_{int(time.time())}_{attempt}.%(ext)s')
                
                with yt_dlp.YoutubeDL(temp_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    if info:
                        # 查找下载的文件
                        expected_path = Path(ydl.prepare_filename(info))
                        if expected_path.exists():
                            return expected_path
                        
                        # 如果预期路径不存在，在临时目录中查找最新文件
                        audio_files = list(self.temp_dir.glob(f'temp_{int(time.time())}_{attempt}.*'))
                        if audio_files:
                            return max(audio_files, key=lambda p: p.stat().st_mtime)
                            
            except Exception as e:
                self.logger.warning(f"Download attempt {attempt + 1} failed for {url}: {e}")
                if attempt < max_retries - 1:
                    # 指数退避
                    wait_time = (2 ** attempt) + (attempt * 0.1)
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(f"All download attempts failed for {url}")
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
            
            # 下载音频
            loop = asyncio.get_event_loop() if asyncio.get_event_loop().is_running() else asyncio.new_event_loop()
            if loop.is_running():
                # 如果在异步环境中，创建新的事件循环
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._download_with_retry(url))
                    audio_path = future.result()
            else:
                audio_path = asyncio.run(self._download_with_retry(url))
            
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
        """批量异步提取音频"""
        self.logger.info(f"Starting batch extraction of {len(urls)} URLs")
        
        # 验证所有 URLs
        valid_urls = [url for url in urls if self._validate_url(url)]
        invalid_urls = [url for url in urls if not self._validate_url(url)]
        
        if invalid_urls:
            self.logger.warning(f"Skipping {len(invalid_urls)} invalid URLs: {invalid_urls}")
        
        if not valid_urls:
            return []
        
        # 并发下载
        semaphore = asyncio.Semaphore(3)  # 限制并发数
        
        async def extract_single(url: str) -> Optional[AudioFile]:
            async with semaphore:
                try:
                    return self.extract_audio(url)
                except Exception as e:
                    self.logger.error(f"Failed to extract audio from {url}: {e}")
                    return None
        
        # 执行并发下载
        tasks = [extract_single(url) for url in valid_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 过滤成功的结果
        audio_files = []
        for result in results:
            if isinstance(result, AudioFile):
                audio_files.append(result)
            elif isinstance(result, Exception):
                self.logger.error(f"Exception in batch extraction: {result}")
        
        self.logger.info(f"Successfully extracted {len(audio_files)} audio files")
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
            # 清理所有临时文件
            for temp_file in self.temp_dir.glob("temp_*.*"):
                temp_file.unlink()
        except Exception:
            pass  # 忽略清理错误