import os
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class CoreConfig:
    """核心配置"""
    # 转录配置
    model: str = 'large-v3'
    language: str = 'zh-CN'  # zh-CN: 简体中文, zh-TW: 繁体中文, zh: 简体中文
    device: str = 'auto'
    compute_type: str = 'float16'
    
    # 音频配置
    quality: str = 'best'
    format: str = 'wav'
    
    # 目录配置
    data_dir: str = './data'
    temp_dir: str = './data/temp'
    transcripts_dir: str = './data/transcripts'
    
    # 系统配置
    log_level: str = 'INFO'
    cleanup_temp_files: bool = True
    
    def __post_init__(self):
        """初始化配置"""
        # 设备自动检测
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 根据设备调整计算类型
        if self.device == 'cpu' and self.compute_type == 'float16':
            self.compute_type = 'float32'
        
        # 确保目录存在
        for dir_path in [self.data_dir, self.temp_dir, self.transcripts_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


@dataclass
class KnowledgeBaseConfig:
    """知识库配置"""
    # 文本处理
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # 向量化
    embedding_model: str = 'BAAI/bge-large-zh-v1.5'
    embedding_device: str = 'auto'
    
    # 存储
    db_path: str = './data/knowledge_base'
    collection_name: str = 'bili_videos'
    
    def __post_init__(self):
        """初始化知识库配置"""
        # 设备自动检测
        if self.embedding_device == 'auto':
            self.embedding_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 验证分块配置
        if self.chunk_overlap >= self.chunk_size:
            self.chunk_overlap = self.chunk_size // 4
        
        # 确保存储目录存在
        Path(self.db_path).mkdir(parents=True, exist_ok=True)


class Settings:
    """统一配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化配置
        
        Args:
            config_file: 可选的配置文件路径
        """
        # 默认配置
        self.core = CoreConfig()
        self.knowledge_base = KnowledgeBaseConfig()
        
        # 从环境变量覆盖配置
        self._load_from_env()
        
        # 从配置文件加载（如果提供）
        if config_file and os.path.exists(config_file):
            self._load_from_file(config_file)
    
    def _load_from_env(self):
        """从环境变量加载配置"""
        # 核心配置
        if os.getenv('WHISPER_MODEL'):
            self.core.model = os.getenv('WHISPER_MODEL')
        if os.getenv('WHISPER_LANGUAGE'):
            self.core.language = os.getenv('WHISPER_LANGUAGE')
        if os.getenv('WHISPER_DEVICE'):
            self.core.device = os.getenv('WHISPER_DEVICE')
        if os.getenv('AUDIO_QUALITY'):
            self.core.quality = os.getenv('AUDIO_QUALITY')
        if os.getenv('LOG_LEVEL'):
            self.core.log_level = os.getenv('LOG_LEVEL')
        if os.getenv('TEMP_DIR'):
            self.core.temp_dir = os.getenv('TEMP_DIR')
        
        # 知识库配置
        if os.getenv('CHUNK_SIZE'):
            self.knowledge_base.chunk_size = int(os.getenv('CHUNK_SIZE'))
        if os.getenv('EMBEDDING_MODEL'):
            self.knowledge_base.embedding_model = os.getenv('EMBEDDING_MODEL')
        if os.getenv('DB_PATH'):
            self.knowledge_base.db_path = os.getenv('DB_PATH')
    
    def _load_from_file(self, config_file: str):
        """从 YAML 配置文件加载"""
        try:
            import yaml
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # 直接映射配置到对应的配置类
            config_mapping = {
                # 核心配置映射
                'audio': self.core,
                'transcription': self.core, 
                'system': self.core,
                # 知识库配置映射
                'text_processing': self.knowledge_base,
                'vectorization': self.knowledge_base,
                'storage': self.knowledge_base
            }
            
            # 字段名映射
            field_mapping = {
                'vectorization': {
                    'model': 'embedding_model',
                    'device': 'embedding_device'
                }
            }
            
            for section, target_config in config_mapping.items():
                if section in config_data:
                    section_config = config_data[section]
                    for key, value in section_config.items():
                        # 应用字段名映射
                        if section in field_mapping and key in field_mapping[section]:
                            key = field_mapping[section][key]
                        
                        if hasattr(target_config, key):
                            setattr(target_config, key, value)
                        
        except ImportError:
            print("Warning: PyYAML not installed, skipping YAML config file")
        except Exception as e:
            print(f"Warning: Failed to load config file {config_file}: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'core': {
                'model': self.core.model,
                'language': self.core.language,
                'device': self.core.device,
                'compute_type': self.core.compute_type,
                'quality': self.core.quality,
                'format': self.core.format,
                'log_level': self.core.log_level,
                'temp_dir': self.core.temp_dir,
                'transcripts_dir': self.core.transcripts_dir,
            },
            'knowledge_base': {
                'chunk_size': self.knowledge_base.chunk_size,
                'chunk_overlap': self.knowledge_base.chunk_overlap,
                'embedding_model': self.knowledge_base.embedding_model,
                'embedding_device': self.knowledge_base.embedding_device,
                'db_path': self.knowledge_base.db_path,
                'collection_name': self.knowledge_base.collection_name,
            }
        }
    
    def validate(self) -> bool:
        """验证配置有效性"""
        try:
            # 验证核心配置
            valid_models = ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']
            if self.core.model not in valid_models:
                print(f"Warning: Invalid Whisper model: {self.core.model}")
            
            if self.core.quality not in ['best', 'worst', 'bestaudio', 'worstaudio']:
                print(f"Warning: Invalid audio quality: {self.core.quality}")
            
            return True
            
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False
    
    def get_ydl_opts(self) -> Dict[str, Any]:
        """获取 yt-dlp 配置选项"""
        return {
            'format': 'bestaudio/best',
            'outtmpl': f'{self.core.temp_dir}/%(title)s.%(ext)s',
            'noplaylist': True,
            'extractaudio': True,
            'audioformat': self.core.format,
            'audioquality': 0 if self.core.quality == 'best' else 9,
            'quiet': True,
            'no_warnings': True,
        }
    
    def get_whisper_opts(self) -> Dict[str, Any]:
        """获取 Whisper 配置选项"""
        return {
            'model_name': self.core.model,
            'language': self.core.language,
            'device': self.core.device,
            'compute_type': self.core.compute_type,
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


