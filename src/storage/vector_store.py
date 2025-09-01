import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import uuid
import chromadb
from chromadb.config import Settings

from ..models import TextChunk, SearchResult, Collection, EmbeddingVector

logger = logging.getLogger(__name__)


class VectorStore:
    """向量存储管理器，基于 ChromaDB 实现"""
    
    def __init__(self, db_path: str = "./data/knowledge_base", 
                 collection_name: str = "bili_videos"):
        """
        初始化向量存储
        
        Args:
            db_path: 数据库存储路径
            collection_name: 默认集合名称
        """
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.client = None
        self._collections_cache: Dict[str, Any] = {}
        
        # 确保目录存在
        self.db_path.mkdir(parents=True, exist_ok=True)
    
    def _initialize_client(self) -> None:
        """初始化 ChromaDB 客户端"""
        if self.client is None:
            try:
                logger.info(f"Initializing ChromaDB client at {self.db_path}")
                
                # 配置 ChromaDB 设置
                settings = Settings(
                    persist_directory=str(self.db_path),
                    chroma_db_impl="duckdb+parquet",  # 使用持久化存储
                    anonymized_telemetry=False
                )
                
                self.client = chromadb.Client(settings)
                logger.info("ChromaDB client initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
                raise
    
    def create_collection(self, name: str, description: str = None) -> Collection:
        """
        创建或获取集合
        
        Args:
            name: 集合名称
            description: 集合描述
            
        Returns:
            集合对象
        """
        if not self.client:
            self._initialize_client()
        
        try:
            # 尝试获取现有集合
            try:
                chroma_collection = self.client.get_collection(name=name)
                logger.info(f"Retrieved existing collection: {name}")
            except:
                # 创建新集合
                chroma_collection = self.client.create_collection(
                    name=name,
                    metadata={"description": description or f"Collection {name}"}
                )
                logger.info(f"Created new collection: {name}")
            
            # 缓存集合
            self._collections_cache[name] = chroma_collection
            
            # 获取文档数量
            doc_count = chroma_collection.count()
            
            return Collection(
                name=name,
                description=description,
                document_count=doc_count
            )
            
        except Exception as e:
            logger.error(f"Failed to create/get collection {name}: {str(e)}")
            raise
    
    def add_documents(self, chunks: List[TextChunk], collection_name: str = None) -> None:
        """
        添加文档到向量存储
        
        Args:
            chunks: 文本块列表
            collection_name: 目标集合名称
        """
        if not chunks:
            return
        
        collection_name = collection_name or self.collection_name
        
        try:
            # 确保集合存在
            collection_obj = self.create_collection(collection_name)
            chroma_collection = self._collections_cache[collection_name]
            
            logger.info(f"Adding {len(chunks)} documents to collection {collection_name}")
            
            # 分批处理文档
            batch_size = 100  # ChromaDB 推荐的批处理大小
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                # 准备批处理数据
                ids = []
                texts = []
                metadatas = []
                
                for chunk in batch:
                    # 生成唯一ID
                    doc_id = str(uuid.uuid4())
                    ids.append(doc_id)
                    texts.append(chunk.text)
                    
                    # 准备元数据
                    metadata = dict(chunk.metadata)
                    metadata.update({
                        'start_time': chunk.start_time or 0.0,
                        'end_time': chunk.end_time or 0.0,
                        'source_file': chunk.source_file or '',
                        'chunk_index': chunk.chunk_index or 0
                    })
                    metadatas.append(metadata)
                
                # 添加到集合（ChromaDB 会自动生成嵌入向量）
                chroma_collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas
                )
                
                logger.info(f"Added batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            
            logger.info(f"Successfully added {len(chunks)} documents to {collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to add documents to collection {collection_name}: {str(e)}")
            raise
    
    def add_embedding_vectors(self, embedding_vectors: List[EmbeddingVector], 
                             collection_name: str = None) -> None:
        """
        添加预计算的嵌入向量到存储
        
        Args:
            embedding_vectors: 嵌入向量列表
            collection_name: 目标集合名称
        """
        if not embedding_vectors:
            return
        
        collection_name = collection_name or self.collection_name
        
        try:
            # 确保集合存在
            collection_obj = self.create_collection(collection_name)
            chroma_collection = self._collections_cache[collection_name]
            
            logger.info(f"Adding {len(embedding_vectors)} embedding vectors to collection {collection_name}")
            
            # 分批处理
            batch_size = 100
            
            for i in range(0, len(embedding_vectors), batch_size):
                batch = embedding_vectors[i:i + batch_size]
                
                ids = []
                embeddings = []
                texts = []
                metadatas = []
                
                for emb_vec in batch:
                    doc_id = str(uuid.uuid4())
                    ids.append(doc_id)
                    embeddings.append(emb_vec.vector)
                    texts.append(emb_vec.text_chunk.text)
                    
                    # 准备元数据
                    metadata = dict(emb_vec.text_chunk.metadata)
                    metadata.update({
                        'start_time': emb_vec.text_chunk.start_time or 0.0,
                        'end_time': emb_vec.text_chunk.end_time or 0.0,
                        'source_file': emb_vec.text_chunk.source_file or '',
                        'chunk_index': emb_vec.text_chunk.chunk_index or 0,
                        'model_name': emb_vec.model_name,
                        'dimension': emb_vec.dimension
                    })
                    metadatas.append(metadata)
                
                # 添加到集合
                chroma_collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas
                )
                
                logger.info(f"Added embedding batch {i//batch_size + 1}/{(len(embedding_vectors) + batch_size - 1)//batch_size}")
            
            logger.info(f"Successfully added {len(embedding_vectors)} embedding vectors to {collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to add embedding vectors to collection {collection_name}: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = 5, 
                         collection_name: str = None) -> List[SearchResult]:
        """
        执行语义相似度搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            collection_name: 搜索的集合名称
            
        Returns:
            搜索结果列表
        """
        collection_name = collection_name or self.collection_name
        
        try:
            # 获取集合
            if collection_name not in self._collections_cache:
                self.create_collection(collection_name)
            
            chroma_collection = self._collections_cache[collection_name]
            
            logger.info(f"Searching for '{query}' in collection {collection_name}, k={k}")
            
            # 执行查询
            results = chroma_collection.query(
                query_texts=[query],
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # 转换结果格式
            search_results = []
            
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0] if results['distances'] else []
                
                for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
                    # 重建 TextChunk 对象
                    chunk_metadata = dict(metadata)
                    
                    # 移除我们添加的额外字段
                    start_time = chunk_metadata.pop('start_time', 0.0)
                    end_time = chunk_metadata.pop('end_time', 0.0)
                    source_file = chunk_metadata.pop('source_file', '')
                    chunk_index = chunk_metadata.pop('chunk_index', 0)
                    chunk_metadata.pop('model_name', None)
                    chunk_metadata.pop('dimension', None)
                    
                    text_chunk = TextChunk(
                        text=doc,
                        metadata=chunk_metadata,
                        start_time=start_time,
                        end_time=end_time,
                        source_file=source_file,
                        chunk_index=chunk_index
                    )
                    
                    # 计算相似度分数 (距离转换为相似度)
                    distance = distances[i] if i < len(distances) else 1.0
                    score = max(0.0, 1.0 - distance)  # 简单转换
                    
                    search_result = SearchResult(
                        text_chunk=text_chunk,
                        score=score,
                        distance=distance
                    )
                    
                    search_results.append(search_result)
            
            logger.info(f"Found {len(search_results)} results for query")
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {str(e)}")
            raise
    
    def delete_collection(self, collection_name: str) -> None:
        """
        删除集合
        
        Args:
            collection_name: 集合名称
        """
        if not self.client:
            self._initialize_client()
        
        try:
            self.client.delete_collection(name=collection_name)
            
            # 从缓存中移除
            if collection_name in self._collections_cache:
                del self._collections_cache[collection_name]
            
            logger.info(f"Deleted collection: {collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {str(e)}")
            raise
    
    def list_collections(self) -> List[Collection]:
        """
        列出所有集合
        
        Returns:
            集合列表
        """
        if not self.client:
            self._initialize_client()
        
        try:
            collections = self.client.list_collections()
            result = []
            
            for col in collections:
                collection_obj = Collection(
                    name=col.name,
                    description=col.metadata.get('description', ''),
                    document_count=col.count()
                )
                result.append(collection_obj)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to list collections: {str(e)}")
            raise
    
    def get_collection_stats(self, collection_name: str = None) -> Dict[str, Any]:
        """
        获取集合统计信息
        
        Args:
            collection_name: 集合名称
            
        Returns:
            统计信息字典
        """
        collection_name = collection_name or self.collection_name
        
        try:
            if collection_name not in self._collections_cache:
                self.create_collection(collection_name)
            
            chroma_collection = self._collections_cache[collection_name]
            
            return {
                'name': collection_name,
                'document_count': chroma_collection.count(),
                'db_path': str(self.db_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            raise