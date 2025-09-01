# Bili-Cortex

🎥 Bilibili 视频音频提取与转录系统

将 B站视频自动转换为高质量的中文转录文本，支持批量处理和时间戳标注。

## 快速开始

### 1. 环境设置

```bash
# 克隆项目后，运行一键设置脚本
./setup.sh
```

### 2. 基本使用

```bash
# 转录单个视频
./bili-cortex.sh https://www.bilibili.com/video/BV1234567890

# 转录多个视频
./bili-cortex.sh url1 url2 url3

# 从文件批量转录
./bili-cortex.sh --from-file urls.txt
```

### 3. 文件格式

创建 `urls.txt` 文件，每行一个 URL：
```
https://www.bilibili.com/video/BV1234567890
https://www.bilibili.com/video/BV1234567891
# 这是注释行，会被忽略
https://b23.tv/abc123
```

## 命令选项

| 选项 | 说明 | 示例 |
|------|------|------|
| `--from-file, -f` | 从文件读取 URL 列表 | `-f urls.txt` |
| `--no-save` | 不保存转录文件 | `--no-save` |
| `--log-level` | 设置日志级别 | `--log-level DEBUG` |
| `--info` | 显示系统信息 | `--info` |
| `--help, -h` | 显示帮助信息 | `--help` |

## 语言设置

### 支持的语言格式
- `zh-CN`: 简体中文（自动转换为简体）
- `zh-TW`: 繁体中文（自动转换为繁体）
- `zh`: 简体中文（默认）
- `en`: 英语
- `ja`: 日语
- `ko`: 韩语

### 设置简体中文（推荐）
```bash
# 方法1: 环境变量
export WHISPER_LANGUAGE="zh-CN"
./bili-cortex.sh <url>

# 方法2: 配置文件
cp config_example.yaml config.yaml
# 编辑 config.yaml 中的 language 设置为 'zh-CN'
```

### 设置繁体中文
```bash
# 环境变量
export WHISPER_LANGUAGE="zh-TW" 
./bili-cortex.sh <url>

# 配置文件：language: 'zh-TW'
```

### 注意
- 系统会根据语言设置自动进行繁简转换
- 无需手动后处理，转录结果已经是目标格式

## 输出文件

转录结果保存在 `data/transcripts/` 目录下：

```
data/transcripts/transcript_20250901_143022_a1b2c3d4.md
```

文件包含：
- 完整转录文本
- 带时间戳的分段内容
- 处理时间和来源信息

## 支持平台

- ✅ bilibili.com
- ✅ www.bilibili.com  
- ✅ b23.tv
- ✅ m.bilibili.com

## 系统要求

- Python 3.8+
- 2GB+ 可用内存
- 网络连接

## 手动运行

如果不使用启动脚本：

```bash
# 激活虚拟环境
source .venv/bin/activate

# 运行程序
PYTHONPATH="$(pwd)" python src/main.py <选项>
```

## 故障排除

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

**测试安装**：
```bash
./bili-cortex.sh --info
```

## 注意事项

- 首次运行会下载 Whisper 模型（约 3GB）
- GPU 支持会自动检测，CPU 模式作为后备
- 转录准确率通常达到 90%+ 
- 支持中文语音优化

---

🚀 **开始使用**: `./bili-cortex.sh <bilibili_url>`