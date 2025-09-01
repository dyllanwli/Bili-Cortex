#!/bin/bash

# Bili-Cortex 项目环境设置脚本
# 自动创建虚拟环境、安装依赖、验证安装

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查 Python 版本
check_python() {
    log_info "检查 Python 版本..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        
        if [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -ge 8 ]]; then
            log_success "Python $PYTHON_VERSION 版本符合要求 (>= 3.8)"
            PYTHON_CMD="python3"
        else
            log_error "Python 版本 $PYTHON_VERSION 不符合要求，需要 Python 3.8 或更高版本"
            exit 1
        fi
    else
        log_error "未找到 Python3，请先安装 Python 3.8 或更高版本"
        exit 1
    fi
}

# 创建虚拟环境
create_venv() {
    log_info "创建虚拟环境..."
    
    if [[ -d ".venv" ]]; then
        log_warning "虚拟环境 .venv 已存在"
        read -p "是否要重新创建虚拟环境？这将删除现有环境 (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "删除现有虚拟环境..."
            rm -rf .venv
        else
            log_info "跳过虚拟环境创建"
            return 0
        fi
    fi
    
    $PYTHON_CMD -m venv .venv
    log_success "虚拟环境创建完成"
}

# 激活虚拟环境
activate_venv() {
    log_info "激活虚拟环境..."
    source .venv/bin/activate
    log_success "虚拟环境已激活"
}

# 升级 pip
upgrade_pip() {
    log_info "升级 pip..."
    python -m pip install --upgrade pip
    log_success "pip 升级完成"
}

# 安装依赖
install_dependencies() {
    log_info "安装项目依赖..."
    
    if [[ ! -f "requirements.txt" ]]; then
        log_error "未找到 requirements.txt 文件"
        exit 1
    fi
    
    pip install -r requirements.txt
    log_success "依赖安装完成"
}

# 创建必要的目录
create_directories() {
    log_info "创建必要的目录结构..."
    
    directories=(
        "data"
        "data/temp"
        "data/audio" 
        "data/transcripts"
        "logs"
    )
    
    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log_info "创建目录: $dir"
        fi
    done
    
    log_success "目录结构创建完成"
}

# 验证安装
verify_installation() {
    log_info "验证安装..."
    
    # 检查核心模块是否可以导入
    python -c "
import sys
sys.path.insert(0, '.')

try:
    from src.models import AudioFile, Transcript, TranscriptSegment
    from config.settings import Settings
    print('✓ 核心模块导入成功')
except ImportError as e:
    print(f'✗ 模块导入失败: {e}')
    sys.exit(1)

try:
    import yt_dlp
    print('✓ yt-dlp 可用')
except ImportError:
    print('✗ yt-dlp 导入失败')
    sys.exit(1)

try:
    import torch
    print('✓ PyTorch 可用')
    if torch.cuda.is_available():
        print('✓ CUDA 支持可用')
    else:
        print('! CUDA 不可用，将使用 CPU')
except ImportError:
    print('✗ PyTorch 导入失败')
    sys.exit(1)

try:
    from faster_whisper import WhisperModel
    print('✓ faster-whisper 可用')
except ImportError:
    print('✗ faster-whisper 导入失败')
    sys.exit(1)

print('\\n所有核心依赖验证通过！')
"
    
    if [[ $? -eq 0 ]]; then
        log_success "安装验证通过"
    else
        log_error "安装验证失败"
        exit 1
    fi
}

# 运行测试
run_tests() {
    log_info "运行单元测试..."
    
    if command -v pytest &> /dev/null; then
        python -m pytest tests/ -v --tb=short
        if [[ $? -eq 0 ]]; then
            log_success "所有测试通过"
        else
            log_warning "部分测试失败，但核心功能应该正常工作"
        fi
    else
        log_warning "pytest 不可用，跳过测试"
    fi
}

# 创建启动脚本
create_launcher() {
    log_info "创建启动脚本..."
    
    cat > bili-cortex.sh << 'EOF'
#!/bin/bash

# Bili-Cortex 启动脚本

# 检查虚拟环境
if [[ ! -d ".venv" ]]; then
    echo "错误: 虚拟环境不存在，请先运行 ./setup.sh"
    exit 1
fi

# 激活虚拟环境
source .venv/bin/activate

# 运行主程序
python src/main.py "$@"
EOF
    
    chmod +x bili-cortex.sh
    log_success "启动脚本创建完成: ./bili-cortex.sh"
}

# 显示使用说明
show_usage() {
    echo
    echo "=================================================="
    echo -e "${GREEN}🎉 Bili-Cortex 安装完成！${NC}"
    echo "=================================================="
    echo
    echo -e "${BLUE}使用方法:${NC}"
    echo
    echo "1. 激活虚拟环境:"
    echo "   source .venv/bin/activate"
    echo
    echo "2. 直接使用启动脚本:"
    echo "   ./bili-cortex.sh <bilibili_url>"
    echo
    echo "3. 或者手动运行:"
    echo "   python src/main.py <bilibili_url>"
    echo
    echo -e "${BLUE}示例:${NC}"
    echo "   ./bili-cortex.sh https://www.bilibili.com/video/BV1234567890"
    echo "   ./bili-cortex.sh --from-file urls.txt"
    echo "   ./bili-cortex.sh --info  # 显示系统信息"
    echo
    echo -e "${BLUE}更多选项:${NC}"
    echo "   python src/main.py --help"
    echo
    echo -e "${YELLOW}注意:${NC} 首次运行时会下载 Whisper 模型，可能需要一些时间"
    echo
}

# 主函数
main() {
    echo "=================================================="
    echo -e "${BLUE}🚀 Bili-Cortex 项目环境设置${NC}"
    echo "=================================================="
    echo
    
    # 检查是否在项目根目录
    if [[ ! -f "requirements.txt" ]] || [[ ! -d "src" ]]; then
        log_error "请在项目根目录运行此脚本"
        exit 1
    fi
    
    check_python
    create_venv
    activate_venv
    upgrade_pip
    install_dependencies
    create_directories
    verify_installation
    run_tests
    create_launcher
    show_usage
    
    log_success "环境设置完成！"
}

# 错误处理
trap 'log_error "脚本执行失败"; exit 1' ERR

# 执行主函数
main "$@"