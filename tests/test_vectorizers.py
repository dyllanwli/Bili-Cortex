import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np

from src.vectorizers.embedding_vectorizer import EmbeddingVectorizer
from src.models import TextChunk, EmbeddingVector


class TestEmbeddingVectorizer(unittest.TestCase):
    """测试 EmbeddingVectorizer 类"""
    
    def setUp(self):
        """测试前准备"""
        self.vectorizer = EmbeddingVectorizer(
            model_name="test-model",
            batch_size=4,
            max_seq_length=128,
            device="cpu"
        )
        
        # 创建模拟的文本块
        self.mock_chunks = [
            TextChunk(
                text="这是第一个文本块，包含了一些测试内容。",
                metadata={"source": "test1", "language": "zh"},
                start_time=0.0,
                end_time=10.0,
                chunk_index=0
            ),
            TextChunk(
                text="这是第二个文本块，用于测试向量化功能。",
                metadata={"source": "test2", "language": "zh"},
                start_time=10.0,
                end_time=20.0,
                chunk_index=1
            )
        ]
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.vectorizer.model_name, "test-model")
        self.assertEqual(self.vectorizer.batch_size, 4)
        self.assertEqual(self.vectorizer.max_seq_length, 128)
        self.assertEqual(self.vectorizer.device, "cpu")
        self.assertEqual(self.vectorizer.dimension, 1024)  # 默认维度
        self.assertIsNone(self.vectorizer.model)
        self.assertEqual(len(self.vectorizer._vector_cache), 0)
    
    def test_determine_device_auto_with_cuda(self):
        """测试自动设备检测（有 CUDA）"""
        with patch('torch.cuda.is_available', return_value=True):
            device = self.vectorizer._determine_device("auto")
            self.assertEqual(device, "cuda")
    
    def test_determine_device_auto_without_cuda(self):
        """测试自动设备检测（无 CUDA）"""
        with patch('torch.cuda.is_available', return_value=False):
            device = self.vectorizer._determine_device("auto")
            self.assertEqual(device, "cpu")
    
    def test_determine_device_manual(self):
        """测试手动设备设置"""
        self.assertEqual(self.vectorizer._determine_device("cuda"), "cuda")
        self.assertEqual(self.vectorizer._determine_device("cpu"), "cpu")
    
    @patch('src.vectorizers.embedding_vectorizer.SentenceTransformer')
    def test_load_model_success(self, mock_sentence_transformer):
        """测试成功加载模型"""
        # 配置模拟对象
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_sentence_transformer.return_value = mock_model
        
        # 加载模型
        self.vectorizer.load_model()
        
        # 验证
        mock_sentence_transformer.assert_called_once_with(
            "test-model", 
            device="cpu"
        )
        self.assertEqual(self.vectorizer.model, mock_model)
        self.assertEqual(self.vectorizer.dimension, 768)
    
    @patch('src.vectorizers.embedding_vectorizer.SentenceTransformer')
    def test_load_model_failure(self, mock_sentence_transformer):
        """测试模型加载失败"""
        mock_sentence_transformer.side_effect = Exception("Model loading failed")
        
        with self.assertRaises(Exception):
            self.vectorizer.load_model()
    
    def test_encode_texts_empty_list(self):
        """测试编码空列表"""
        # 直接测试，不需要加载模型
        result = self.vectorizer.encode_texts([])
        self.assertEqual(result, [])
    
    @patch('src.vectorizers.embedding_vectorizer.SentenceTransformer')
    def test_encode_texts_success(self, mock_sentence_transformer):
        """测试成功编码文本"""
        # 设置模拟模型
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 4
        mock_model.encode.return_value = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        mock_sentence_transformer.return_value = mock_model
        
        # 加载模型并编码
        self.vectorizer.load_model()
        texts = ["文本1", "文本2"]
        result = self.vectorizer.encode_texts(texts)
        
        # 验证结果
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(result[1], [5.0, 6.0, 7.0, 8.0])
        
        # 验证模型调用
        mock_model.encode.assert_called_once_with(
            texts,
            batch_size=4,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    
    @patch('src.vectorizers.embedding_vectorizer.SentenceTransformer')
    def test_encode_texts_with_cache(self, mock_sentence_transformer):
        """测试带缓存的文本编码"""
        # 设置模拟模型
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 4
        mock_model.encode.return_value = np.array([[1.0, 2.0, 3.0, 4.0]])
        mock_sentence_transformer.return_value = mock_model
        
        # 加载模型
        self.vectorizer.load_model()
        
        # 第一次编码
        texts = ["测试文本"]
        result1 = self.vectorizer.encode_texts(texts)
        
        # 第二次编码同样的文本（应该从缓存获取）
        result2 = self.vectorizer.encode_texts(texts)
        
        # 验证结果相同
        self.assertEqual(result1, result2)
        
        # 验证模型只被调用一次（第二次使用缓存）
        self.assertEqual(mock_model.encode.call_count, 1)
    
    @patch('src.vectorizers.embedding_vectorizer.SentenceTransformer')
    def test_encode_chunks(self, mock_sentence_transformer):
        """测试编码文本块"""
        # 设置模拟模型
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 4
        mock_model.encode.return_value = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        mock_sentence_transformer.return_value = mock_model
        
        # 编码文本块
        self.vectorizer.load_model()
        result = self.vectorizer.encode_chunks(self.mock_chunks)
        
        # 验证结果
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], EmbeddingVector)
        self.assertIsInstance(result[1], EmbeddingVector)
        
        # 验证第一个向量
        self.assertEqual(result[0].vector, [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(result[0].text_chunk, self.mock_chunks[0])
        self.assertEqual(result[0].model_name, "test-model")
        self.assertEqual(result[0].dimension, 4)
    
    def test_encode_chunks_empty_list(self):
        """测试编码空文本块列表"""
        result = self.vectorizer.encode_chunks([])
        self.assertEqual(result, [])
    
    @patch('src.vectorizers.embedding_vectorizer.SentenceTransformer')
    def test_encode_single(self, mock_sentence_transformer):
        """测试编码单个文本"""
        # 设置模拟模型
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 4
        mock_model.encode.return_value = np.array([[1.0, 2.0, 3.0, 4.0]])
        mock_sentence_transformer.return_value = mock_model
        
        # 编码单个文本
        self.vectorizer.load_model()
        result = self.vectorizer.encode_single("单个文本")
        
        # 验证结果
        self.assertEqual(result, [1.0, 2.0, 3.0, 4.0])
    
    def test_similarity_calculation(self):
        """测试相似度计算"""
        vector1 = [1.0, 0.0, 0.0]
        vector2 = [0.0, 1.0, 0.0]
        vector3 = [1.0, 0.0, 0.0]
        
        # 测试正交向量（相似度为0）
        similarity = self.vectorizer.similarity(vector1, vector2)
        self.assertAlmostEqual(similarity, 0.0, places=5)
        
        # 测试相同向量（相似度为1）
        similarity = self.vectorizer.similarity(vector1, vector3)
        self.assertAlmostEqual(similarity, 1.0, places=5)
    
    def test_similarity_zero_vectors(self):
        """测试零向量的相似度计算"""
        vector1 = [0.0, 0.0, 0.0]
        vector2 = [1.0, 0.0, 0.0]
        
        similarity = self.vectorizer.similarity(vector1, vector2)
        self.assertEqual(similarity, 0.0)
    
    def test_similarity_error_handling(self):
        """测试相似度计算错误处理"""
        vector1 = [1.0, 2.0]
        vector2 = "invalid_vector"  # 无效向量类型
        
        similarity = self.vectorizer.similarity(vector1, vector2)
        self.assertEqual(similarity, 0.0)
    
    def test_get_cache_key(self):
        """测试缓存键生成"""
        text1 = "测试文本"
        text2 = "测试文本"
        text3 = "不同文本"
        
        key1 = self.vectorizer._get_cache_key(text1)
        key2 = self.vectorizer._get_cache_key(text2)
        key3 = self.vectorizer._get_cache_key(text3)
        
        # 相同文本应该生成相同的键
        self.assertEqual(key1, key2)
        
        # 不同文本应该生成不同的键
        self.assertNotEqual(key1, key3)
        
        # 键应该是字符串
        self.assertIsInstance(key1, str)
    
    def test_get_model_info(self):
        """测试获取模型信息"""
        info = self.vectorizer.get_model_info()
        
        expected_keys = ['model_name', 'dimension', 'device', 'batch_size', 'max_seq_length', 'cache_size']
        for key in expected_keys:
            self.assertIn(key, info)
        
        self.assertEqual(info['model_name'], "test-model")
        self.assertEqual(info['device'], "cpu")
        self.assertEqual(info['batch_size'], 4)
        self.assertEqual(info['max_seq_length'], 128)
        self.assertEqual(info['cache_size'], 0)
    
    def test_clear_cache(self):
        """测试清空缓存"""
        # 添加一些缓存数据
        self.vectorizer._vector_cache["test_key"] = [1.0, 2.0, 3.0]
        self.assertEqual(len(self.vectorizer._vector_cache), 1)
        
        # 清空缓存
        self.vectorizer.clear_cache()
        self.assertEqual(len(self.vectorizer._vector_cache), 0)


class TestEmbeddingVectorizerIntegration(unittest.TestCase):
    """EmbeddingVectorizer 集成测试"""
    
    @patch('src.vectorizers.embedding_vectorizer.SentenceTransformer')
    def test_full_pipeline(self, mock_sentence_transformer):
        """测试完整的向量化流程"""
        # 设置模拟模型
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = np.array([
            [0.1, 0.2, 0.3] * 256,  # 768维向量
            [0.4, 0.5, 0.6] * 256   # 768维向量
        ])
        mock_sentence_transformer.return_value = mock_model
        
        # 创建向量化器
        vectorizer = EmbeddingVectorizer(
            model_name="BAAI/bge-large-zh-v1.5",
            batch_size=32,
            device="cpu"
        )
        
        # 创建测试文本块
        chunks = [
            TextChunk(
                text="这是一个关于机器学习的文本块。",
                metadata={"topic": "AI", "source": "video1"},
                chunk_index=0
            ),
            TextChunk(
                text="这个文本块讨论深度学习的概念。",
                metadata={"topic": "AI", "source": "video1"},
                chunk_index=1
            )
        ]
        
        # 执行向量化
        embedding_vectors = vectorizer.encode_chunks(chunks)
        
        # 验证结果
        self.assertEqual(len(embedding_vectors), 2)
        
        for i, emb_vec in enumerate(embedding_vectors):
            self.assertIsInstance(emb_vec, EmbeddingVector)
            self.assertEqual(len(emb_vec.vector), 768)
            self.assertEqual(emb_vec.text_chunk, chunks[i])
            self.assertEqual(emb_vec.model_name, "BAAI/bge-large-zh-v1.5")
            self.assertEqual(emb_vec.dimension, 768)
        
        # 验证相似度计算
        similarity = vectorizer.similarity(
            embedding_vectors[0].vector,
            embedding_vectors[1].vector
        )
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
    
    @patch('src.vectorizers.embedding_vectorizer.SentenceTransformer')
    def test_batch_processing(self, mock_sentence_transformer):
        """测试批处理功能"""
        # 设置模拟模型
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 4
        
        # 模拟分批处理
        def mock_encode(texts, **kwargs):
            return np.array([[i, i+1, i+2, i+3] for i in range(len(texts))])
        
        mock_model.encode.side_effect = mock_encode
        mock_sentence_transformer.return_value = mock_model
        
        # 创建向量化器，设置较小的批处理大小
        vectorizer = EmbeddingVectorizer(batch_size=2, device="cpu")
        vectorizer.load_model()
        
        # 创建多个文本进行批处理测试
        texts = [f"测试文本 {i}" for i in range(5)]
        
        # 编码文本
        vectors = vectorizer.encode_texts(texts)
        
        # 验证结果
        self.assertEqual(len(vectors), 5)
        
        # 验证每个向量的内容
        for i, vector in enumerate(vectors):
            expected = [i, i+1, i+2, i+3]
            self.assertEqual(vector, expected)


if __name__ == '__main__':
    unittest.main()