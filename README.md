# Bili-Cortex

🎥 智能 Bilibili 视频分析与知识提取系统

将 B站视频自动转换为高质量的中文转录文本，并构建可搜索的知识库，支持语义搜索和内容分析。

## ✨ 核心特性

- 🚀 **高质量转录**: 基于 Whisper 的精准中文语音识别
- 🧠 **智能知识库**: 自动构建可搜索的向量化知识库
- 🔍 **语义搜索**: 支持自然语言查询和内容检索
- 📊 **批量处理**: 支持多视频并行处理
- 🌐 **多语言支持**: 支持简体/繁体中文自动转换
- ⚡ **GPU加速**: 自动检测并使用GPU加速处理

## 🚀 快速开始

### 1. 环境设置

```bash
# 克隆项目后，运行一键设置脚本
./setup.sh
```

### 2. 基础转录

```bash
# 转录单个视频
./bili-cortex.sh https://www.bilibili.com/video/BV1234567890

# 转录多个视频
./bili-cortex.sh url1 url2 url3

# 从文件批量转录
./bili-cortex.sh --from-file urls.txt
```

### 3. 知识库搜索

```bash
# 搜索已处理的内容
./bili-cortex.sh --search "机器学习算法"

# 演示搜索功能
./bili-cortex.sh --search-demo

# 不构建知识库，仅转录
./bili-cortex.sh --no-kb https://www.bilibili.com/video/BV1234567890
```

### 4. 批量处理文件格式

创建 `urls.txt` 文件，每行一个 URL：
```
https://www.bilibili.com/video/BV1234567890
https://www.bilibili.com/video/BV1234567891
# 这是注释行，会被忽略
https://b23.tv/abc123
```

## 📋 命令选项

### 基础选项
| 选项 | 说明 | 示例 |
|------|------|------|
| `--from-file, -f` | 从文件读取 URL 列表 | `-f urls.txt` |
| `--no-save` | 不保存转录文件 | `--no-save` |
| `--no-kb` | 不构建向量知识库 | `--no-kb` |
| `--log-level` | 设置日志级别 | `--log-level DEBUG` |
| `--info` | 显示系统信息 | `--info` |
| `--help, -h` | 显示帮助信息 | `--help` |

### 知识库选项
| 选项 | 说明 | 示例 |
|------|------|------|
| `--search, -s` | 搜索知识库内容 | `-s "深度学习"` |
| `--search-demo` | 演示搜索功能 | `--search-demo` |

## 🌐 语言设置

### 支持的语言格式
- `zh-CN`: 简体中文（自动转换为简体）
- `zh-TW`: 繁体中文（自动转换为繁体）
- `zh`: 简体中文（默认）
- `en`: 英语
- `ja`: 日语
- `ko`: 韩语

### 快速语言配置
```bash
# 方法1: 环境变量（临时）
export WHISPER_LANGUAGE="zh-CN"
./bili-cortex.sh <url>

# 方法2: 配置文件（推荐）
cp config_example.yaml config.yaml
# 编辑 config.yaml 中的 language 设置
```

### 配置文件示例
```yaml
transcription:
  model: 'large-v3'
  language: 'zh-CN'    # 简体中文
  device: 'auto'
  compute_type: 'float16'
```

### 自动转换特性
- ✅ 系统会根据语言设置自动进行繁简转换
- ✅ 无需手动后处理，转录结果已经是目标格式
- ✅ 支持混合内容的智能识别和转换

## 📄 输出文件

### 转录文件
转录结果保存在 `data/transcripts/` 目录下：

```
data/transcripts/transcript_20250901_143022_a1b2c3d4.md
```

文件包含：
- 📝 完整转录文本
- ⏰ 带时间戳的分段内容  
- ℹ️ 处理时间和来源信息

### 知识库文件
向量化知识库保存在 `data/knowledge_base/` 目录下：
- 🧠 文本向量数据库
- 🔍 支持语义搜索
- 📊 自动索引和检索

## 🌐 支持平台

- ✅ **bilibili.com** - 标准桌面版
- ✅ **www.bilibili.com** - WWW子域名  
- ✅ **b23.tv** - 短链接格式
- ✅ **m.bilibili.com** - 移动版链接

## 💻 系统要求

### 基础要求
- **Python 3.8+**
- **4GB+ 可用内存** (推荐8GB+)
- **稳定网络连接**
- **5GB+ 磁盘空间** (模型和数据存储)

### GPU 加速（可选）
- **NVIDIA GPU** 支持 CUDA 11.8+
- **6GB+ 显存** (推荐)
- 自动检测GPU并启用加速

## ⚙️ 高级使用

### 手动运行
如果不使用启动脚本：

```bash
# 激活虚拟环境
source .venv/bin/activate

# 运行程序
PYTHONPATH="$(pwd)" python src/main.py <选项>
```

### 配置自定义
```bash
# 使用自定义配置文件
cp config_example.yaml my_config.yaml
# 编辑配置后运行
PYTHONPATH="$(pwd)" python src/main.py --config my_config.yaml <url>
```

### 性能优化
```bash
# 使用更大的模型以获得更高精度
export WHISPER_MODEL="large-v3"

# 调整文本块大小以优化搜索效果  
export CHUNK_SIZE=1500

# 使用更强的嵌入模型
export EMBEDDING_MODEL="BAAI/bge-large-zh-v1.5"
```

## 🔧 故障排除

### 常见问题

**环境问题**：
```bash
# 重新设置环境
rm -rf .venv
./setup.sh
```

**权限问题**：
```bash
chmod +x setup.sh bili-cortex.sh
```

**依赖问题**：
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

**GPU问题**：
```bash
# 检查CUDA是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 强制使用CPU
export WHISPER_DEVICE="cpu"
```

**知识库问题**：
```bash
# 清理知识库重新构建
rm -rf ./data/knowledge_base
./bili-cortex.sh <url>  # 重新处理
```

**测试安装**：
```bash
./bili-cortex.sh --info
```

## ⚠️ 注意事项

- 📥 **首次运行**会下载 Whisper 模型（约 3GB）和嵌入模型（约 1.5GB）
- 🚀 **GPU 支持**会自动检测，CPU 模式作为后备
- 🎯 **转录准确率**通常达到 90-95%
- 🇨🇳 **中文优化**针对中文语音进行了特别优化
- 💾 **存储空间**每小时视频约需要 100-200MB 存储空间
- 🔍 **搜索效果**随着处理视频数量增加而提升

## 🎉 功能演示

```bash
# 1. 处理视频并构建知识库
./bili-cortex.sh https://www.bilibili.com/video/BV1234567890

# 2. 搜索相关内容
./bili-cortex.sh --search "机器学习"

# 3. 批量处理多个视频
./bili-cortex.sh url1 url2 url3
```

## 🏗️ 架构特性

### 简化设计原则
- **KISS**: 保持简单直观的接口设计
- **DRY**: 避免重复代码，统一核心功能  
- **YAGNI**: 只实现必要功能，避免过度工程化

### 技术栈
- **语音识别**: Faster-Whisper (OpenAI Whisper 优化版)
- **文本处理**: LangChain + 自定义中文优化
- **向量化**: BGE-Large-ZH (中文语义理解)
- **向量存储**: ChromaDB (轻量级向量数据库)
- **异步处理**: asyncio + 并发优化

### 性能特性  
- ⚡ GPU 自动检测和加速
- 🔄 智能批处理和缓存
- 📈 内存使用优化
- 🎯 中文语义搜索优化

---

🚀 **立即开始**: `./bili-cortex.sh <bilibili_url>`

📊 **查看演示**: `./bili-cortex.sh --search-demo`