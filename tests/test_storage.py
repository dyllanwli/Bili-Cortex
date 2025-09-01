import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import tempfile
import shutil

from src.storage.vector_store import VectorStore
from src.models import TextChunk, EmbeddingVector, SearchResult, Collection


class TestVectorStore(unittest.TestCase):
    """测试 VectorStore 类"""
    
    def setUp(self):
        """测试前准备"""
        # 使用临时目录进行测试
        self.temp_dir = tempfile.mkdtemp()
        self.store = VectorStore(
            db_path=self.temp_dir,
            collection_name="test_collection"
        )
        
        # 创建测试数据
        self.test_chunks = [
            TextChunk(
                text="这是第一个测试文本块，包含机器学习相关内容。",
                metadata={"source": "video1", "topic": "AI"},
                start_time=0.0,
                end_time=10.0,
                chunk_index=0
            ),
            TextChunk(
                text="这是第二个测试文本块，讨论深度学习技术。",
                metadata={"source": "video1", "topic": "AI"},
                start_time=10.0,
                end_time=20.0,
                chunk_index=1
            )
        ]
        
        self.test_vectors = [
            EmbeddingVector(
                vector=[0.1, 0.2, 0.3, 0.4],
                text_chunk=self.test_chunks[0],
                model_name="test-model",
                dimension=4
            ),
            EmbeddingVector(
                vector=[0.5, 0.6, 0.7, 0.8],
                text_chunk=self.test_chunks[1],
                model_name="test-model",
                dimension=4
            )
        ]
    
    def tearDown(self):
        """测试后清理"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(str(self.store.db_path), self.temp_dir)
        self.assertEqual(self.store.collection_name, "test_collection")
        self.assertIsNone(self.store.client)
        self.assertEqual(len(self.store._collections_cache), 0)
        
        # 验证目录已创建
        self.assertTrue(Path(self.temp_dir).exists())
    
    @patch('src.storage.vector_store.chromadb.Client')
    def test_initialize_client(self, mock_client_class):
        """测试客户端初始化"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # 初始化客户端
        self.store._initialize_client()
        
        # 验证客户端已设置
        self.assertEqual(self.store.client, mock_client)
        mock_client_class.assert_called_once()
    
    @patch('src.storage.vector_store.chromadb.Client')
    def test_create_collection_new(self, mock_client_class):
        """测试创建新集合"""
        # 设置模拟客户端
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.count.return_value = 0
        
        mock_client.get_collection.side_effect = Exception("Collection not found")
        mock_client.create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        # 创建集合
        collection = self.store.create_collection("new_collection", "Test collection")
        
        # 验证
        self.assertIsInstance(collection, Collection)
        self.assertEqual(collection.name, "new_collection")
        self.assertEqual(collection.description, "Test collection")
        self.assertEqual(collection.document_count, 0)
        
        mock_client.create_collection.assert_called_once_with(
            name="new_collection",
            metadata={"description": "Test collection"}
        )
    
    @patch('src.storage.vector_store.chromadb.Client')
    def test_create_collection_existing(self, mock_client_class):
        """测试获取现有集合"""
        # 设置模拟客户端
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.count.return_value = 5
        
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        # 获取现有集合
        collection = self.store.create_collection("existing_collection")
        
        # 验证
        self.assertIsInstance(collection, Collection)
        self.assertEqual(collection.name, "existing_collection")
        self.assertEqual(collection.document_count, 5)
        
        mock_client.get_collection.assert_called_once_with(name="existing_collection")
        mock_client.create_collection.assert_not_called()
    
    @patch('src.storage.vector_store.chromadb.Client')
    def test_add_documents(self, mock_client_class):
        """测试添加文档"""
        # 设置模拟客户端和集合
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.count.return_value = 0
        
        mock_client.get_collection.side_effect = Exception("Not found")
        mock_client.create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        # 添加文档
        self.store.add_documents(self.test_chunks, "test_collection")
        
        # 验证集合创建和文档添加
        mock_client.create_collection.assert_called_once()
        mock_collection.add.assert_called_once()
        
        # 验证传递给 add 的参数
        call_args = mock_collection.add.call_args
        self.assertIn('ids', call_args.kwargs)
        self.assertIn('documents', call_args.kwargs)
        self.assertIn('metadatas', call_args.kwargs)
        
        # 验证文档内容
        documents = call_args.kwargs['documents']
        self.assertEqual(len(documents), 2)
        self.assertIn("机器学习", documents[0])
        self.assertIn("深度学习", documents[1])
    
    @patch('src.storage.vector_store.chromadb.Client')
    def test_add_embedding_vectors(self, mock_client_class):
        """测试添加嵌入向量"""
        # 设置模拟客户端和集合
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.count.return_value = 0
        
        mock_client.get_collection.side_effect = Exception("Not found")
        mock_client.create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        # 添加嵌入向量
        self.store.add_embedding_vectors(self.test_vectors, "test_collection")
        
        # 验证集合创建和向量添加
        mock_client.create_collection.assert_called_once()
        mock_collection.add.assert_called_once()
        
        # 验证传递给 add 的参数
        call_args = mock_collection.add.call_args
        self.assertIn('ids', call_args.kwargs)
        self.assertIn('embeddings', call_args.kwargs)
        self.assertIn('documents', call_args.kwargs)
        self.assertIn('metadatas', call_args.kwargs)
        
        # 验证向量内容
        embeddings = call_args.kwargs['embeddings']
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(embeddings[0], [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(embeddings[1], [0.5, 0.6, 0.7, 0.8])
    
    @patch('src.storage.vector_store.chromadb.Client')
    def test_similarity_search(self, mock_client_class):
        """测试相似度搜索"""
        # 设置模拟客户端和集合
        mock_client = Mock()
        mock_collection = Mock()
        
        # 模拟搜索结果
        mock_collection.query.return_value = {
            'documents': [["第一个结果文档", "第二个结果文档"]],
            'metadatas': [[
                {'source': 'video1', 'topic': 'AI', 'start_time': 0.0, 'end_time': 10.0, 'source_file': 'test.wav', 'chunk_index': 0},
                {'source': 'video2', 'topic': 'ML', 'start_time': 10.0, 'end_time': 20.0, 'source_file': 'test2.wav', 'chunk_index': 1}
            ]],
            'distances': [[0.2, 0.5]]
        }
        
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        # 执行搜索
        results = self.store.similarity_search("测试查询", k=2, collection_name="test_collection")
        
        # 验证结果
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], SearchResult)
        self.assertIsInstance(results[1], SearchResult)
        
        # 验证第一个结果
        first_result = results[0]
        self.assertEqual(first_result.text_chunk.text, "第一个结果文档")
        self.assertAlmostEqual(first_result.score, 0.8, places=1)  # 1.0 - 0.2
        self.assertEqual(first_result.distance, 0.2)
        
        # 验证查询调用
        mock_collection.query.assert_called_once_with(
            query_texts=["测试查询"],
            n_results=2,
            include=['documents', 'metadatas', 'distances']
        )
    
    @patch('src.storage.vector_store.chromadb.Client')
    def test_similarity_search_no_results(self, mock_client_class):
        """测试搜索无结果"""
        # 设置模拟客户端和集合
        mock_client = Mock()
        mock_collection = Mock()
        
        # 模拟空搜索结果
        mock_collection.query.return_value = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        # 执行搜索
        results = self.store.similarity_search("无结果查询", k=5)
        
        # 验证结果
        self.assertEqual(len(results), 0)
    
    @patch('src.storage.vector_store.chromadb.Client')
    def test_delete_collection(self, mock_client_class):
        """测试删除集合"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # 添加到缓存中
        self.store._collections_cache["test_collection"] = Mock()
        
        # 删除集合
        self.store.delete_collection("test_collection")
        
        # 验证
        mock_client.delete_collection.assert_called_once_with(name="test_collection")
        self.assertNotIn("test_collection", self.store._collections_cache)
    
    @patch('src.storage.vector_store.chromadb.Client')
    def test_list_collections(self, mock_client_class):
        """测试列出所有集合"""
        # 设置模拟客户端
        mock_client = Mock()
        
        # 模拟集合列表
        mock_col1 = Mock()
        mock_col1.name = "collection1"
        mock_col1.metadata = {"description": "First collection"}
        mock_col1.count.return_value = 10
        
        mock_col2 = Mock()
        mock_col2.name = "collection2"
        mock_col2.metadata = {}
        mock_col2.count.return_value = 5
        
        mock_client.list_collections.return_value = [mock_col1, mock_col2]
        mock_client_class.return_value = mock_client
        
        # 获取集合列表
        collections = self.store.list_collections()
        
        # 验证结果
        self.assertEqual(len(collections), 2)
        
        self.assertEqual(collections[0].name, "collection1")
        self.assertEqual(collections[0].description, "First collection")
        self.assertEqual(collections[0].document_count, 10)
        
        self.assertEqual(collections[1].name, "collection2")
        self.assertEqual(collections[1].description, "")
        self.assertEqual(collections[1].document_count, 5)
    
    @patch('src.storage.vector_store.chromadb.Client')
    def test_get_collection_stats(self, mock_client_class):
        """测试获取集合统计信息"""
        # 设置模拟客户端和集合
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.count.return_value = 25
        
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        # 获取统计信息
        stats = self.store.get_collection_stats("test_collection")
        
        # 验证结果
        self.assertEqual(stats['name'], "test_collection")
        self.assertEqual(stats['document_count'], 25)
        self.assertEqual(stats['db_path'], self.temp_dir)
    
    def test_add_documents_empty_list(self):
        """测试添加空文档列表"""
        # 这应该不会引起错误，直接返回
        self.store.add_documents([])
        # 如果没有异常，测试通过
    
    def test_add_embedding_vectors_empty_list(self):
        """测试添加空向量列表"""
        # 这应该不会引起错误，直接返回
        self.store.add_embedding_vectors([])
        # 如果没有异常，测试通过


class TestVectorStoreIntegration(unittest.TestCase):
    """VectorStore 集成测试"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.store = VectorStore(db_path=self.temp_dir, collection_name="integration_test")
    
    def tearDown(self):
        """测试后清理"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    @patch('src.storage.vector_store.chromadb.Client')
    def test_full_workflow(self, mock_client_class):
        """测试完整的工作流程"""
        # 设置完整的模拟环境
        mock_client = Mock()
        mock_collection = Mock()
        
        # 模拟集合创建
        mock_client.get_collection.side_effect = Exception("Not found")
        mock_client.create_collection.return_value = mock_collection
        mock_collection.count.return_value = 0
        
        # 模拟搜索结果
        mock_collection.query.return_value = {
            'documents': [["相关文档内容"]],
            'metadatas': [[{
                'source': 'test_video',
                'start_time': 5.0,
                'end_time': 15.0,
                'source_file': 'test.wav',
                'chunk_index': 0
            }]],
            'distances': [[0.3]]
        }
        
        mock_client_class.return_value = mock_client
        
        # 创建测试数据
        chunks = [
            TextChunk(
                text="这是一个关于人工智能的综合讨论。",
                metadata={"source": "test_video", "topic": "AI"},
                start_time=0.0,
                end_time=30.0,
                chunk_index=0
            )
        ]
        
        vectors = [
            EmbeddingVector(
                vector=[0.1, 0.2, 0.3, 0.4, 0.5],
                text_chunk=chunks[0],
                model_name="test-model",
                dimension=5
            )
        ]
        
        # 执行完整工作流程
        
        # 1. 创建集合
        collection = self.store.create_collection("ai_content", "AI related content")
        self.assertIsInstance(collection, Collection)
        
        # 2. 添加文档
        self.store.add_documents(chunks, "ai_content")
        
        # 3. 添加向量
        self.store.add_embedding_vectors(vectors, "ai_content")
        
        # 4. 执行搜索
        results = self.store.similarity_search("人工智能相关内容", k=3, collection_name="ai_content")
        
        # 验证搜索结果
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], SearchResult)
        self.assertEqual(results[0].text_chunk.text, "相关文档内容")
        
        # 5. 获取统计信息
        stats = self.store.get_collection_stats("ai_content")
        self.assertIn('name', stats)
        self.assertIn('document_count', stats)
        self.assertIn('db_path', stats)
        
        # 验证所有操作都被正确调用
        mock_client.create_collection.assert_called()
        mock_collection.add.assert_called()
        mock_collection.query.assert_called()


if __name__ == '__main__':
    unittest.main()