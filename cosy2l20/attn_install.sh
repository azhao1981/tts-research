
# 1. 检查 CUDA 编译器
nvcc --version
# 如果报错 "command not found"，请执行：
# export PATH=/usr/local/cuda/bin:$PATH

# 2. 确认 ninja 已安装 (它决定了编译速度)
uv pip install ninja packaging

# 1. 强制使用所有核心进行编译 最好不要用完 用一半核心,应该和内存相关，1C会到10G左右
export MAX_JOBS=8

# 2. 告诉 Flash Attention 这是一个 Ada Lovelace (L20) 架构
# 虽然通常会自动检测，但显式指定可以避免识别错误
export TORCH_CUDA_ARCH_LIST="8.9"

# 开始安装
# --no-build-isolation: 使用当前环境的 torch
# --no-cache-dir: 禁止使用缓存，强制重新下载和编译
# 使用 nohup 防止 SSH 断开导致中断
nohup uv pip install flash-attn --no-build-isolation --no-cache-dir > build_log.txt 2>&1 &

# 3. 既然用了 nohup，命令会在后台跑。
# 你可以用这就话实时查看进度：
tail -f build_log.txt

# 方案二：逃课法（直接用预编译包，不折腾了）
# 既然编译这么痛苦且容易死机，我强烈建议你放弃源码编译，直接下载对应版本的 .whl 文件安装。这是最稳妥的，几秒钟就装好了，不会死机。
# ???? 为什么不早说
# 请按照你的环境（Python 3.10 + Torch 2.3 + CUDA 12.1）下载：
# 下载 whl 文件： (这是官方 Release 页面对应的版本，适用于你的环境)
# 注：官方包通常标 cu123，但它通常兼容 cu121
# run if
# wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# uv pip install flash_attn-2.6.3+cu123torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl