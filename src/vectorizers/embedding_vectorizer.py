import logging
from typing import List, Optional, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from ..models import TextChunk, EmbeddingVector

logger = logging.getLogger(__name__)


class EmbeddingVectorizer:
    """嵌入向量化器，将文本转换为向量表示"""
    
    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5", 
                 batch_size: int = 32, max_seq_length: int = 512,
                 device: str = "auto"):
        """
        初始化嵌入向量化器
        
        Args:
            model_name: 使用的嵌入模型名称
            batch_size: 批处理大小
            max_seq_length: 最大序列长度
            device: 计算设备 (auto, cpu, cuda)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.device = self._determine_device(device)
        
        # 初始化模型
        self.model = None
        self.dimension = 1024  # bge-large-zh-v1.5 默认维度
        
        # 向量缓存
        self._vector_cache: Dict[str, List[float]] = {}
        
    def _determine_device(self, device: str) -> str:
        """确定计算设备"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def load_model(self) -> None:
        """加载嵌入模型"""
        try:
            logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            
            # 设置最大序列长度
            if hasattr(self.model, 'max_seq_length'):
                self.model.max_seq_length = self.max_seq_length
            
            # 获取实际维度
            self.dimension = self.model.get_sentence_embedding_dimension()
            
            logger.info(f"Model loaded successfully. Dimension: {self.dimension}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        """
        将文本列表编码为向量
        
        Args:
            texts: 文本列表
            
        Returns:
            向量列表
        """
        if not texts:
            return []
            
        if not self.model:
            self.load_model()
        
        # 检查缓存
        cached_results = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            text_key = self._get_cache_key(text)
            if text_key in self._vector_cache:
                cached_results.append((i, self._vector_cache[text_key]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # 编码未缓存的文本
        if uncached_texts:
            try:
                logger.info(f"Encoding {len(uncached_texts)} texts in batches of {self.batch_size}")
                
                # 批量编码
                embeddings = self.model.encode(
                    uncached_texts,
                    batch_size=self.batch_size,
                    show_progress_bar=len(uncached_texts) > 10,
                    convert_to_numpy=True,
                    normalize_embeddings=True  # 归一化向量
                )
                
                # 缓存结果
                for text, embedding in zip(uncached_texts, embeddings):
                    text_key = self._get_cache_key(text)
                    self._vector_cache[text_key] = embedding.tolist()
                
                # 合并缓存和新编码的结果
                all_results = [None] * len(texts)
                
                # 填入缓存结果
                for idx, vector in cached_results:
                    all_results[idx] = vector
                
                # 填入新编码结果
                for i, idx in enumerate(uncached_indices):
                    all_results[idx] = embeddings[i].tolist()
                
                return all_results
                
            except Exception as e:
                logger.error(f"Failed to encode texts: {str(e)}")
                raise
        else:
            # 所有结果都来自缓存
            result = [None] * len(texts)
            for idx, vector in cached_results:
                result[idx] = vector
            return result
    
    def encode_chunks(self, chunks: List[TextChunk]) -> List[EmbeddingVector]:
        """
        将文本块列表转换为嵌入向量
        
        Args:
            chunks: 文本块列表
            
        Returns:
            嵌入向量列表
        """
        if not chunks:
            return []
        
        # 提取文本
        texts = [chunk.text for chunk in chunks]
        
        # 编码为向量
        vectors = self.encode_texts(texts)
        
        # 创建 EmbeddingVector 对象
        embedding_vectors = []
        for chunk, vector in zip(chunks, vectors):
            embedding_vector = EmbeddingVector(
                vector=vector,
                text_chunk=chunk,
                model_name=self.model_name,
                dimension=len(vector)
            )
            embedding_vectors.append(embedding_vector)
        
        return embedding_vectors
    
    def encode_single(self, text: str) -> List[float]:
        """
        编码单个文本
        
        Args:
            text: 待编码文本
            
        Returns:
            向量表示
        """
        vectors = self.encode_texts([text])
        return vectors[0] if vectors else []
    
    def similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            vector1: 向量1
            vector2: 向量2
            
        Returns:
            相似度分数 (0-1)
        """
        try:
            v1 = np.array(vector1)
            v2 = np.array(vector2)
            
            # 计算余弦相似度
            dot_product = np.dot(v1, v2)
            norms = np.linalg.norm(v1) * np.linalg.norm(v2)
            
            if norms == 0:
                return 0.0
            
            similarity = dot_product / norms
            return float(max(0.0, min(1.0, similarity)))  # 限制在 0-1 范围内
            
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {str(e)}")
            return 0.0
    
    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        # 使用文本的哈希值作为缓存键
        import hashlib
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'device': self.device,
            'batch_size': self.batch_size,
            'max_seq_length': self.max_seq_length,
            'cache_size': len(self._vector_cache)
        }
    
    def clear_cache(self) -> None:
        """清空向量缓存"""
        self._vector_cache.clear()
        logger.info("Vector cache cleared")