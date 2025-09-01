#!/usr/bin/env python3
"""
Bili-Cortex 主入口文件
视频音频提取与转录系统
"""

import asyncio
import logging
import sys
import argparse
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from config.settings import get_settings
from src.extractors.video_extractor import VideoExtractor
from src.transcribers.whisper_transcriber import WhisperTranscriber
from src.models import AudioFile, Transcript


def setup_logging(log_level: str = "INFO") -> None:
    """设置日志配置"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    # 抑制第三方库的详细日志
    logging.getLogger('yt_dlp').setLevel(logging.WARNING)
    logging.getLogger('faster_whisper').setLevel(logging.WARNING)


class BiliCortexProcessor:
    """Bili-Cortex 主处理器"""
    
    def __init__(self):
        """初始化处理器"""
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.extractor = VideoExtractor(
            temp_dir=self.settings.audio.temp_dir,
            quality=self.settings.audio.quality,
            format=self.settings.audio.format
        )
        
        self.transcriber = WhisperTranscriber(
            **self.settings.get_whisper_opts()
        )
        
        self.logger.info("Bili-Cortex processor initialized")
        self.logger.info(f"Configuration: {self.settings.to_dict()}")
    
    def validate_urls(self, urls: List[str]) -> List[str]:
        """验证 URL 列表"""
        valid_urls = []
        for url in urls:
            if len(url) > self.settings.security.max_url_length:
                self.logger.warning(f"URL too long, skipping: {url[:50]}...")
                continue
            
            if self.settings.security.enable_url_validation:
                # 简单的 URL 格式验证
                if not any(domain in url.lower() for domain in self.settings.security.allowed_domains):
                    self.logger.warning(f"Domain not allowed, skipping: {url}")
                    continue
            
            valid_urls.append(url)
        
        return valid_urls
    
    async def process_urls(self, urls: List[str], save_transcripts: bool = True) -> List[Transcript]:
        """处理 URL 列表的完整流程"""
        self.logger.info(f"Starting processing of {len(urls)} URLs")
        
        # 验证 URLs
        valid_urls = self.validate_urls(urls)
        if not valid_urls:
            self.logger.error("No valid URLs to process")
            return []
        
        try:
            # 步骤 1: 提取音频
            self.logger.info("Step 1: Extracting audio files...")
            if len(valid_urls) == 1:
                audio_files = [self.extractor.extract_audio(valid_urls[0])]
            else:
                audio_files = await self.extractor.batch_extract(valid_urls)
            
            if not audio_files:
                self.logger.error("No audio files extracted successfully")
                return []
            
            self.logger.info(f"Successfully extracted {len(audio_files)} audio files")
            
            # 步骤 2: 转录音频
            self.logger.info("Step 2: Transcribing audio files...")
            estimated_time = self.transcriber.estimate_processing_time(audio_files)
            self.logger.info(f"Estimated processing time: {estimated_time:.1f} seconds")
            
            transcripts = self.transcriber.batch_transcribe(audio_files)
            
            if not transcripts:
                self.logger.error("No transcripts generated successfully")
                return []
            
            self.logger.info(f"Successfully generated {len(transcripts)} transcripts")
            
            # 步骤 3: 保存转录结果
            if save_transcripts:
                self.logger.info("Step 3: Saving transcripts...")
                await self._save_transcripts(transcripts)
            
            # 步骤 4: 清理临时文件
            if self.settings.system.cleanup_temp_files:
                self.logger.info("Step 4: Cleaning up temporary files...")
                self.extractor.cleanup_temp_files(audio_files)
            
            return transcripts
            
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            # 确保清理临时文件
            try:
                if hasattr(self, 'extractor') and 'audio_files' in locals():
                    self.extractor.cleanup_temp_files(audio_files)
            except:
                pass
            raise
    
    async def _save_transcripts(self, transcripts: List[Transcript]) -> None:
        """保存转录结果"""
        transcripts_dir = Path(self.settings.system.transcripts_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, transcript in enumerate(transcripts, 1):
            # 生成文件名
            if transcript.source_audio.source_url:
                # 从 URL 提取标识符
                url_hash = str(abs(hash(transcript.source_audio.source_url)))[:8]
                filename = f"transcript_{timestamp}_{url_hash}.md"
            else:
                filename = f"transcript_{timestamp}_{i:03d}.md"
            
            output_path = transcripts_dir / filename
            
            try:
                self.transcriber.save_transcript(transcript, output_path)
                self.logger.info(f"Transcript saved: {output_path}")
            except Exception as e:
                self.logger.error(f"Failed to save transcript {i}: {e}")
    
    def process_single_url(self, url: str) -> Optional[Transcript]:
        """处理单个 URL（同步接口）"""
        try:
            return asyncio.run(self.process_urls([url]))[0]
        except (IndexError, Exception):
            return None
    
    def get_system_info(self) -> dict:
        """获取系统信息"""
        return {
            "extractor_info": {
                "temp_dir": str(self.extractor.temp_dir),
                "supported_platforms": self.extractor.SUPPORTED_PLATFORMS,
                "quality": self.extractor.quality,
                "format": self.extractor.format,
            },
            "transcriber_info": self.transcriber.get_model_info(),
            "settings": self.settings.to_dict()
        }


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Bili-Cortex: Bilibili 视频音频提取与转录系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 处理单个视频
  python src/main.py https://www.bilibili.com/video/BV1234567890
  
  # 处理多个视频
  python src/main.py url1 url2 url3
  
  # 从文件读取 URL 列表
  python src/main.py --from-file urls.txt
  
  # 不保存转录文件（仅处理）
  python src/main.py --no-save https://www.bilibili.com/video/BV1234567890
        """
    )
    
    parser.add_argument(
        'urls',
        nargs='*',
        help='Bilibili 视频 URL 列表'
    )
    
    parser.add_argument(
        '--from-file', '-f',
        type=str,
        help='从文件读取 URL 列表（每行一个 URL）'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='不保存转录结果到文件'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='日志级别 (default: INFO)'
    )
    
    parser.add_argument(
        '--info',
        action='store_true',
        help='显示系统信息'
    )
    
    return parser.parse_args()


def load_urls_from_file(file_path: str) -> List[str]:
    """从文件加载 URL 列表"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            urls = []
            for line in f:
                url = line.strip()
                if url and not url.startswith('#'):  # 忽略空行和注释
                    urls.append(url)
            return urls
    except Exception as e:
        print(f"Error reading URLs from file {file_path}: {e}")
        return []


async def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # 初始化处理器
    try:
        processor = BiliCortexProcessor()
    except Exception as e:
        logger.error(f"Failed to initialize processor: {e}")
        return 1
    
    # 显示系统信息
    if args.info:
        import json
        print(json.dumps(processor.get_system_info(), indent=2, ensure_ascii=False))
        return 0
    
    # 收集 URLs
    urls = []
    
    if args.from_file:
        urls.extend(load_urls_from_file(args.from_file))
    
    if args.urls:
        urls.extend(args.urls)
    
    if not urls:
        logger.error("No URLs provided. Use --help for usage information.")
        return 1
    
    # 处理 URLs
    try:
        logger.info(f"Starting Bili-Cortex processing with {len(urls)} URLs")
        transcripts = await processor.process_urls(urls, save_transcripts=not args.no_save)
        
        if transcripts:
            logger.info(f"✅ Processing completed successfully!")
            logger.info(f"Generated {len(transcripts)} transcripts")
            
            # 显示简要统计
            total_duration = sum(
                t.source_audio.duration or 0 for t in transcripts
            )
            total_segments = sum(len(t.segments) for t in transcripts)
            
            logger.info(f"Total audio duration: {total_duration:.1f} seconds")
            logger.info(f"Total transcript segments: {total_segments}")
            
            return 0
        else:
            logger.error("❌ No transcripts generated")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        sys.exit(130)