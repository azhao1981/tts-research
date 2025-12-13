uv pip install ninja packaging
# 1. 强制使用所有核心进行编译 最好不要用完
export MAX_JOBS=15

# 2. 告诉 Flash Attention 这是一个 Ada Lovelace (L20) 架构
# 虽然通常会自动检测，但显式指定可以避免识别错误
export TORCH_CUDA_ARCH_LIST="8.9"

# 3. 开始安装 (指定 --no-build-isolation 以使用当前环境的 torch)
uv pip install flash-attn --no-build-isolation