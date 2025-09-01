import os
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class AudioConfig:
    """音频配置"""
    quality: str = 'best'
    format: str = 'wav'
    temp_dir: str = './data/temp'
    
    def __post_init__(self):
        """确保临时目录存在"""
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)


@dataclass 
class TranscriptionConfig:
    """转录配置"""
    model: str = 'large-v3'
    language: str = 'zh-CN'  # zh-CN: 简体中文, zh-TW: 繁体中文, zh: 简体中文
    device: str = 'auto'
    batch_size: int = 8
    compute_type: str = 'float16'
    
    def __post_init__(self):
        """设备自动检测和验证"""
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 根据设备调整计算类型
        if self.device == 'cpu' and self.compute_type == 'float16':
            self.compute_type = 'float32'


@dataclass
class SystemConfig:
    """系统配置"""
    log_level: str = 'INFO'
    max_concurrent_downloads: int = 3
    max_file_size_mb: int = 500
    max_duration_minutes: int = 120
    cleanup_temp_files: bool = True
    
    # 数据目录配置
    data_dir: str = './data'
    audio_dir: str = './data/audio'
    transcripts_dir: str = './data/transcripts'
    temp_dir: str = './data/temp'
    
    def __post_init__(self):
        """确保所有目录存在"""
        for dir_path in [self.data_dir, self.audio_dir, self.transcripts_dir, self.temp_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


@dataclass
class SecurityConfig:
    """安全配置"""
    allowed_domains: list = None
    max_url_length: int = 2048
    enable_url_validation: bool = True
    log_sensitive_info: bool = False
    
    def __post_init__(self):
        if self.allowed_domains is None:
            self.allowed_domains = [
                'bilibili.com',
                'www.bilibili.com', 
                'b23.tv',
                'm.bilibili.com'
            ]


@dataclass
class TextProcessingConfig:
    """文本处理配置"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    
    def __post_init__(self):
        """验证文本处理配置"""
        if self.chunk_overlap >= self.chunk_size:
            self.chunk_overlap = self.chunk_size // 4
        if self.min_chunk_size > self.chunk_size:
            self.min_chunk_size = self.chunk_size // 2


@dataclass
class VectorizationConfig:
    """向量化配置"""
    model: str = 'BAAI/bge-large-zh-v1.5'
    batch_size: int = 32
    max_seq_length: int = 512
    device: str = 'auto'
    
    def __post_init__(self):
        """设备自动检测"""
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class StorageConfig:
    """存储配置"""
    db_path: str = './data/knowledge_base'
    collection_name: str = 'bili_videos'
    persist_directory: str = './data/knowledge_base'
    
    def __post_init__(self):
        """确保存储目录存在"""
        Path(self.db_path).mkdir(parents=True, exist_ok=True)
        if self.persist_directory != self.db_path:
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)


class Settings:
    """统一配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化配置
        
        Args:
            config_file: 可选的配置文件路径
        """
        # 默认配置
        self.audio = AudioConfig()
        self.transcription = TranscriptionConfig()
        self.system = SystemConfig()
        self.security = SecurityConfig()
        self.text_processing = TextProcessingConfig()
        self.vectorization = VectorizationConfig()
        self.storage = StorageConfig()
        
        # 从环境变量覆盖配置
        self._load_from_env()
        
        # 从配置文件加载（如果提供）
        if config_file and os.path.exists(config_file):
            self._load_from_file(config_file)
    
    def _load_from_env(self):
        """从环境变量加载配置"""
        # 音频配置
        if os.getenv('AUDIO_QUALITY'):
            self.audio.quality = os.getenv('AUDIO_QUALITY')
        if os.getenv('AUDIO_FORMAT'):
            self.audio.format = os.getenv('AUDIO_FORMAT')
        if os.getenv('TEMP_DIR'):
            self.audio.temp_dir = os.getenv('TEMP_DIR')
        
        # 转录配置
        if os.getenv('WHISPER_MODEL'):
            self.transcription.model = os.getenv('WHISPER_MODEL')
        if os.getenv('WHISPER_LANGUAGE'):
            self.transcription.language = os.getenv('WHISPER_LANGUAGE')
        if os.getenv('WHISPER_DEVICE'):
            self.transcription.device = os.getenv('WHISPER_DEVICE')
        if os.getenv('BATCH_SIZE'):
            self.transcription.batch_size = int(os.getenv('BATCH_SIZE'))
        
        # 系统配置
        if os.getenv('LOG_LEVEL'):
            self.system.log_level = os.getenv('LOG_LEVEL')
        if os.getenv('MAX_CONCURRENT'):
            self.system.max_concurrent_downloads = int(os.getenv('MAX_CONCURRENT'))
        
        # 文本处理配置
        if os.getenv('CHUNK_SIZE'):
            self.text_processing.chunk_size = int(os.getenv('CHUNK_SIZE'))
        if os.getenv('CHUNK_OVERLAP'):
            self.text_processing.chunk_overlap = int(os.getenv('CHUNK_OVERLAP'))
        
        # 向量化配置
        if os.getenv('EMBEDDING_MODEL'):
            self.vectorization.model = os.getenv('EMBEDDING_MODEL')
        if os.getenv('EMBEDDING_BATCH_SIZE'):
            self.vectorization.batch_size = int(os.getenv('EMBEDDING_BATCH_SIZE'))
        if os.getenv('EMBEDDING_DEVICE'):
            self.vectorization.device = os.getenv('EMBEDDING_DEVICE')
        
        # 存储配置
        if os.getenv('DB_PATH'):
            self.storage.db_path = os.getenv('DB_PATH')
        if os.getenv('COLLECTION_NAME'):
            self.storage.collection_name = os.getenv('COLLECTION_NAME')
    
    def _load_from_file(self, config_file: str):
        """从 YAML 配置文件加载"""
        try:
            import yaml
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # 更新音频配置
            if 'audio' in config_data:
                audio_config = config_data['audio']
                for key, value in audio_config.items():
                    if hasattr(self.audio, key):
                        setattr(self.audio, key, value)
            
            # 更新转录配置  
            if 'transcription' in config_data:
                transcription_config = config_data['transcription']
                for key, value in transcription_config.items():
                    if hasattr(self.transcription, key):
                        setattr(self.transcription, key, value)
            
            # 更新系统配置
            if 'system' in config_data:
                system_config = config_data['system']
                for key, value in system_config.items():
                    if hasattr(self.system, key):
                        setattr(self.system, key, value)
            
            # 更新文本处理配置
            if 'text_processing' in config_data:
                text_config = config_data['text_processing']
                for key, value in text_config.items():
                    if hasattr(self.text_processing, key):
                        setattr(self.text_processing, key, value)
            
            # 更新向量化配置
            if 'vectorization' in config_data:
                vector_config = config_data['vectorization']
                for key, value in vector_config.items():
                    if hasattr(self.vectorization, key):
                        setattr(self.vectorization, key, value)
            
            # 更新存储配置
            if 'storage' in config_data:
                storage_config = config_data['storage']
                for key, value in storage_config.items():
                    if hasattr(self.storage, key):
                        setattr(self.storage, key, value)
                        
        except ImportError:
            print("Warning: PyYAML not installed, skipping YAML config file")
        except Exception as e:
            print(f"Warning: Failed to load config file {config_file}: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'audio': {
                'quality': self.audio.quality,
                'format': self.audio.format,
                'temp_dir': self.audio.temp_dir,
            },
            'transcription': {
                'model': self.transcription.model,
                'language': self.transcription.language,
                'device': self.transcription.device,
                'batch_size': self.transcription.batch_size,
                'compute_type': self.transcription.compute_type,
            },
            'system': {
                'log_level': self.system.log_level,
                'max_concurrent_downloads': self.system.max_concurrent_downloads,
                'max_file_size_mb': self.system.max_file_size_mb,
                'max_duration_minutes': self.system.max_duration_minutes,
                'cleanup_temp_files': self.system.cleanup_temp_files,
                'data_dir': self.system.data_dir,
                'audio_dir': self.system.audio_dir,
                'transcripts_dir': self.system.transcripts_dir,
                'temp_dir': self.system.temp_dir,
            },
            'security': {
                'allowed_domains': self.security.allowed_domains,
                'max_url_length': self.security.max_url_length,
                'enable_url_validation': self.security.enable_url_validation,
                'log_sensitive_info': self.security.log_sensitive_info,
            },
            'text_processing': {
                'chunk_size': self.text_processing.chunk_size,
                'chunk_overlap': self.text_processing.chunk_overlap,
                'min_chunk_size': self.text_processing.min_chunk_size,
            },
            'vectorization': {
                'model': self.vectorization.model,
                'batch_size': self.vectorization.batch_size,
                'max_seq_length': self.vectorization.max_seq_length,
                'device': self.vectorization.device,
            },
            'storage': {
                'db_path': self.storage.db_path,
                'collection_name': self.storage.collection_name,
                'persist_directory': self.storage.persist_directory,
            }
        }
    
    def validate(self) -> bool:
        """验证配置有效性"""
        try:
            # 验证音频配置
            if self.audio.quality not in ['best', 'worst', 'bestaudio', 'worstaudio']:
                print(f"Warning: Invalid audio quality: {self.audio.quality}")
            
            if self.audio.format not in ['wav', 'mp3', 'flac', 'm4a']:
                print(f"Warning: Invalid audio format: {self.audio.format}")
            
            # 验证转录配置
            valid_models = ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']
            if self.transcription.model not in valid_models:
                print(f"Warning: Invalid Whisper model: {self.transcription.model}")
            
            if self.transcription.batch_size < 1 or self.transcription.batch_size > 32:
                print(f"Warning: Invalid batch size: {self.transcription.batch_size}")
            
            # 验证系统配置
            if self.system.max_concurrent_downloads < 1 or self.system.max_concurrent_downloads > 10:
                print(f"Warning: Invalid max concurrent downloads: {self.system.max_concurrent_downloads}")
            
            return True
            
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False
    
    def get_ydl_opts(self) -> Dict[str, Any]:
        """获取 yt-dlp 配置选项"""
        return {
            'format': 'bestaudio/best',
            'outtmpl': f'{self.audio.temp_dir}/%(title)s.%(ext)s',
            'noplaylist': True,
            'extractaudio': True,
            'audioformat': self.audio.format,
            'audioquality': 0 if self.audio.quality == 'best' else 9,
            'quiet': True,
            'no_warnings': True,
        }
    
    def get_whisper_opts(self) -> Dict[str, Any]:
        """获取 Whisper 配置选项"""
        return {
            'model_name': self.transcription.model,
            'language': self.transcription.language,
            'device': self.transcription.device,
            'batch_size': self.transcription.batch_size,
            'compute_type': self.transcription.compute_type,
        }


# 全局配置实例
settings = Settings()


def get_settings() -> Settings:
    """获取全局配置实例"""
    return settings


def reload_settings(config_file: Optional[str] = None) -> Settings:
    """重新加载配置"""
    global settings
    settings = Settings(config_file)
    return settings