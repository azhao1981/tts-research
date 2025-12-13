
# 1. 检查 CUDA 编译器
nvcc --version
# 如果报错 "command not found"，请执行：
# export PATH=/usr/local/cuda/bin:$PATH

# 2. 确认 ninja 已安装 (它决定了编译速度)
uv pip install ninja packaging

# 1. 强制使用所有核心进行编译 最好不要用完 有一半核心
export MAX_JOBS=8

# 2. 告诉 Flash Attention 这是一个 Ada Lovelace (L20) 架构
# 虽然通常会自动检测，但显式指定可以避免识别错误
export TORCH_CUDA_ARCH_LIST="8.9"

# 开始安装
# --no-build-isolation: 使用当前环境的 torch
# --no-cache-dir: 禁止使用缓存，强制重新下载和编译
uv pip install flash-attn --no-build-isolation --no-cache-dir

# 请按照你的环境（Python 3.10 + Torch 2.3 + CUDA 12.1）下载：
# 下载 whl 文件： (这是官方 Release 页面对应的版本，适用于你的环境)
# 注：官方包通常标 cu123，但它通常兼容 cu121
# run if
# wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# uv pip install flash_attn-2.6.3+cu123torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl