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

import yt_dlp
from config.settings import get_settings
from src.extractors.video_extractor import VideoExtractor
from src.transcribers.whisper_transcriber import WhisperTranscriber
from src.processors.text_processor import TextProcessor
from src.vectorizers.embedding_vectorizer import EmbeddingVectorizer
from src.storage.vector_store import VectorStore
from src.models import AudioFile, Transcript, TextChunk, SearchResult
from src.utils.transcript_loader import load_transcripts_from_dir


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
            temp_dir=self.settings.core.temp_dir,
            quality=self.settings.core.quality,
            format=self.settings.core.format
        )
        
        self.transcriber = WhisperTranscriber(
            **self.settings.get_whisper_opts()
        )
        
        # 初始化知识库组件
        self.text_processor = TextProcessor(
            chunk_size=self.settings.knowledge_base.chunk_size,
            chunk_overlap=self.settings.knowledge_base.chunk_overlap
        )
        
        self.vectorizer = EmbeddingVectorizer(
            model_name=self.settings.knowledge_base.embedding_model,
            device=self.settings.knowledge_base.embedding_device
        )
        
        self.vector_store = VectorStore(
            db_path=self.settings.knowledge_base.db_path,
            collection_name=self.settings.knowledge_base.collection_name
        )
        
        self.logger.info("Bili-Cortex processor initialized")
        self.logger.info(f"Configuration: {self.settings.to_dict()}")
    
    def is_channel_url(self, url: str) -> bool:
        """检测是否为频道URL"""
        channel_patterns = [
            '/channel/',
            '/c/',
            '/@',
            '/user/',
            'space.bilibili.com/'
        ]
        return any(pattern in url.lower() for pattern in channel_patterns)

    def is_playlist_url(self, url: str) -> bool:
        """检测是否为 YouTube 播放列表URL"""
        u = url.lower()
        return ('list=' in u) or ('/playlist?' in u)
    
    def extract_channel_videos(self, channel_url: str, limit: int = 10) -> List[str]:
        """从频道提取视频URLs (简化版)

        当 limit <= 0 时，提取频道下的全部可见视频。
        """
        try:
            self.logger.info(
                f"Extracting videos from channel: {channel_url} (limit: {'ALL' if limit and limit <= 0 else limit})"
            )
            
            # 使用频道视频列表URL
            if '/channel/' in channel_url:
                videos_url = channel_url + '/videos'
            elif '/@' in channel_url or '/user/' in channel_url:
                videos_url = channel_url + '/videos'
            else:
                videos_url = channel_url
            
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True,  # 只获取URL，不下载
            }
            # limit <= 0 表示提取全部
            if limit and limit > 0:
                ydl_opts['playlistend'] = limit
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(videos_url, download=False)
                
                if not info or 'entries' not in info:
                    self.logger.warning(f"No videos found in channel: {channel_url}")
                    return []
                
                video_urls = []
                for entry in info['entries']:
                    if entry and (not limit or limit <= 0 or len(video_urls) < limit):
                        video_id = entry.get('id')
                        # 确保获取的是正确的视频ID
                        if video_id and video_id != channel_url.split('/')[-1]:
                            video_urls.append(f"https://www.youtube.com/watch?v={video_id}")
                            self.logger.debug(f"Added video: {video_id}")
                
                self.logger.info(f"Extracted {len(video_urls)} video URLs from channel")
                return video_urls
                
        except Exception as e:
            self.logger.error(f"Failed to extract channel videos: {e}")
            return []

    def extract_playlist_videos(self, playlist_url: str, limit: int = 0) -> List[str]:
        """从播放列表提取视频URLs；limit<=0 表示全部

        支持两种形式：
        - https://www.youtube.com/playlist?list=XXXX
        - https://www.youtube.com/watch?v=XXXX&list=YYYY （会重写为 playlist 链接）
        """
        try:
            self.logger.info(
                f"Extracting videos from playlist: {playlist_url} (limit: {'ALL' if limit and limit <= 0 else limit or 'ALL'})"
            )
            # 规范化为 /playlist?list= 格式
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(playlist_url)
            qs = parse_qs(parsed.query)
            list_id = None
            if 'list' in qs and qs['list']:
                list_id = qs['list'][0]
            if list_id:
                normalized_url = f"https://www.youtube.com/playlist?list={list_id}"
            else:
                normalized_url = playlist_url
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True,
            }
            if limit and limit > 0:
                ydl_opts['playlistend'] = limit
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(normalized_url, download=False)
                entries = info.get('entries', []) if info else []
                video_urls = []
                for entry in entries:
                    if entry and (not limit or limit <= 0 or len(video_urls) < limit):
                        vid = entry.get('id')
                        if vid:
                            video_urls.append(f"https://www.youtube.com/watch?v={vid}")
                self.logger.info(f"Extracted {len(video_urls)} video URLs from playlist")
                return video_urls
        except Exception as e:
            self.logger.error(f"Failed to extract playlist videos: {e}")
            return []
    
    def validate_urls(self, urls: List[str], channel_limit: int = 10) -> List[str]:
        """简化的 URL 验证，支持频道扩展"""
        valid_urls = []
        allowed_domains = [
            'bilibili.com',
            'www.bilibili.com', 
            'b23.tv',
            'm.bilibili.com',
            'youtube.com',
            'www.youtube.com',
            'youtu.be',
            'm.youtube.com'
        ]
        
        for url in urls:
            # 基本的域名检查
            if any(domain in url.lower() for domain in allowed_domains):
                # 播放列表优先展开
                if self.is_playlist_url(url):
                    self.logger.info(f"Playlist URL detected: {url}")
                    pl_videos = self.extract_playlist_videos(url, channel_limit)
                    if pl_videos:
                        valid_urls.extend(pl_videos)
                    else:
                        self.logger.warning(f"No videos extracted from playlist: {url}")
                # 检查是否为频道URL
                elif self.is_channel_url(url):
                    self.logger.info(f"Channel URL detected: {url}")
                    channel_videos = self.extract_channel_videos(url, channel_limit)
                    if channel_videos:
                        valid_urls.extend(channel_videos)
                    else:
                        self.logger.warning(f"No videos extracted from channel: {url}")
                else:
                    valid_urls.append(url)
            else:
                self.logger.warning(f"Unsupported domain, skipping: {url}")
        
        return valid_urls
    
    async def process_urls(self, urls: List[str], save_transcripts: bool = True,
                          build_knowledge_base: bool = True, channel_limit: int = 10) -> List[Transcript]:
        """处理 URL 列表的完整流程"""
        self.logger.info(f"Starting processing of {len(urls)} URLs")
        
        # 验证 URLs
        valid_urls = self.validate_urls(urls, channel_limit)
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
            transcripts = self.transcriber.batch_transcribe(audio_files)
            
            if not transcripts:
                self.logger.error("No transcripts generated successfully")
                return []
            
            self.logger.info(f"Successfully generated {len(transcripts)} transcripts")
            
            # 步骤 3: 保存转录结果
            if save_transcripts:
                self.logger.info("Step 3: Saving transcripts...")
                await self._save_transcripts(transcripts)
            
            # 步骤 4: 构建知识库
            if build_knowledge_base:
                self.logger.info("Step 4: Building knowledge base...")
                await self._build_knowledge_base(transcripts)
            
            # 步骤 5: 清理临时文件
            if self.settings.core.cleanup_temp_files:
                self.logger.info("Step 5: Cleaning up temporary files...")
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
        transcripts_dir = Path(self.settings.core.transcripts_dir)
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
    
    async def _build_knowledge_base(self, transcripts: List[Transcript]) -> None:
        """构建向量化知识库"""
        try:
            all_chunks = []
            
            # 处理每个转录结果
            for transcript in transcripts:
                self.logger.info(f"Processing transcript from {transcript.source_audio.file_path}")
                
                # 文本处理和分块
                chunks = self.text_processor.chunk_text(transcript)
                all_chunks.extend(chunks)
                
                self.logger.info(f"Generated {len(chunks)} text chunks")
            
            if not all_chunks:
                self.logger.warning("No text chunks generated")
                return
            
            self.logger.info(f"Total chunks to process: {len(all_chunks)}")
            
            # 生成嵌入向量
            self.logger.info("Generating embedding vectors...")
            embedding_vectors = self.vectorizer.encode_chunks(all_chunks)
            
            if not embedding_vectors:
                self.logger.error("Failed to generate embedding vectors")
                return
            
            self.logger.info(f"Generated {len(embedding_vectors)} embedding vectors")
            
            # 存储到向量数据库
            self.logger.info("Storing vectors in knowledge base...")
            self.vector_store.add_items(embedding_vectors)
            
            # 获取存储统计信息
            stats = self.vector_store.get_collection_stats()
            self.logger.info(f"Knowledge base updated: {stats['document_count']} total documents")
            
        except Exception as e:
            self.logger.error(f"Failed to build knowledge base: {e}")
            raise
    
    def search_knowledge_base(self, query: str, k: int = 5) -> List[SearchResult]:
        """搜索知识库"""
        try:
            self.logger.info(f"Searching knowledge base for: '{query}'")
            results = self.vector_store.similarity_search(query, k=k)
            self.logger.info(f"Found {len(results)} results")
            return results
        except Exception as e:
            self.logger.error(f"Failed to search knowledge base: {e}")
            return []
    
    def demo_search(self, query: str = None) -> None:
        """演示知识库搜索功能"""
        if not query:
            query = "视频内容"
        
        self.logger.info("=== 知识库搜索演示 ===")
        results = self.search_knowledge_base(query)
        
        if not results:
            self.logger.info("No results found")
            return
        
        for i, result in enumerate(results, 1):
            self.logger.info(f"\n--- Result {i} (Score: {result.score:.3f}) ---")
            self.logger.info(f"Text: {result.text_chunk.text[:200]}...")
            if result.text_chunk.start_time:
                self.logger.info(f"Time: {result.text_chunk.start_time:.1f}s - {result.text_chunk.end_time:.1f}s")
            self.logger.info(f"Source: {result.text_chunk.source_file}")
            
        self.logger.info("=== 搜索演示完成 ===")
    
    def process_single_url(self, url: str, channel_limit: int = 10) -> Optional[Transcript]:
        """处理单个 URL（同步接口）"""
        try:
            results = asyncio.run(self.process_urls([url], channel_limit=channel_limit))
            return results[0] if results else None
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
  
  # 处理YouTube频道（默认10个视频）
  python src/main.py https://www.youtube.com/channel/UC1234567890
  
  # 处理频道并限制视频数量
  python src/main.py --channel-limit 5 https://www.youtube.com/channel/UC1234567890
  
  # 从文件读取 URL 列表
  python src/main.py --from-file urls.txt
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
        '--no-kb',
        action='store_true',
        help='不构建向量知识库'
    )
    
    parser.add_argument(
        '--search', '-s',
        type=str,
        help='搜索已构建的知识库'
    )
    
    parser.add_argument(
        '--search-demo',
        action='store_true',
        help='演示知识库搜索功能'
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
    
    parser.add_argument(
        '--channel-limit',
        type=int,
        default=10,
        help='频道/播放列表视频数量限制，0 表示全部 (default: 10)'
    )

    parser.add_argument(
        '--index-transcripts',
        action='store_true',
        help='从已保存的转录文件构建知识库（不重新转录）'
    )

    parser.add_argument(
        '--transcripts-dir',
        default=None,
        help='指定转录文件目录，未指定则使用配置中的 transcripts_dir'
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
    
    # 搜索知识库
    if args.search:
        results = processor.search_knowledge_base(args.search)
        if results:
            print(f"\n=== 搜索结果：'{args.search}' ===")
            for i, result in enumerate(results, 1):
                print(f"\n--- Result {i} (Score: {result.score:.3f}) ---")
                print(f"Text: {result.text_chunk.text[:300]}...")
                if result.text_chunk.start_time:
                    print(f"Time: {result.text_chunk.start_time:.1f}s - {result.text_chunk.end_time:.1f}s")
                print(f"Source: {result.text_chunk.source_file}")
        else:
            print(f"No results found for '{args.search}'")
        return 0
    
    # 搜索演示
    if args.search_demo:
        processor.demo_search()
        return 0

    # 仅从现有转录文件构建知识库
    if args.index_transcripts:
        transcripts_dir = Path(args.transcripts_dir) if args.transcripts_dir else Path(processor.settings.core.transcripts_dir)
        if not transcripts_dir.exists():
            logger.error(f"Transcripts directory not found: {transcripts_dir}")
            return 1
        logger.info(f"Loading transcripts from: {transcripts_dir}")
        transcripts = load_transcripts_from_dir(transcripts_dir)
        if not transcripts:
            logger.error("No transcripts loaded from directory")
            return 1
        logger.info(f"Loaded {len(transcripts)} transcripts, building knowledge base...")
        try:
            await processor._build_knowledge_base(transcripts)
            logger.info("✅ Knowledge base built from transcripts")
            return 0
        except Exception as e:
            logger.error(f"Failed to build KB from transcripts: {e}")
            return 1
    
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
        transcripts = await processor.process_urls(
            urls, 
            save_transcripts=not args.no_save,
            build_knowledge_base=not args.no_kb,
            channel_limit=args.channel_limit
        )
        
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
